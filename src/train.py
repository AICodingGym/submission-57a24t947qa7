# Single-fold trainer: cached grayscale memmap -> GPU augmentation -> timm backbone -> 11-label BCE.
# Writes per-epoch curve rows, checkpoint, val (OOF) preds and test preds every epoch.
import argparse, os, time, math, json, threading, queue
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--arch", default="tf_efficientnetv2_s.in21k_ft_in1k")
p.add_argument("--fold", type=int, default=0)
p.add_argument("--size", type=int, default=512)
p.add_argument("--cache", type=int, default=768)
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--wd", type=float, default=1e-2)
p.add_argument("--warmup", type=float, default=0.5)
p.add_argument("--ema", type=float, default=0.999)
p.add_argument("--drop_path", type=float, default=0.2)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--full", type=int, default=0)  # 1 = train on all data (no val)
p.add_argument("--max_minutes", type=float, default=1e9)
p.add_argument("--seg_w", type=float, default=0.0)  # >0: auxiliary line-segmentation loss from annotations
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dev = "cuda"

tr = pd.read_csv(f"{ROOT}/data/train.csv")
folds = pd.read_csv(f"{ROOT}/work/folds.csv")
assert (folds.StudyInstanceUID == tr.StudyInstanceUID).all()
sub = pd.read_csv(f"{ROOT}/data/sample_submission.csv")
COLS = list(tr.columns[1:12])
EVAL = list(sub.columns[1:])
EVAL_IDX = [COLS.index(c) for c in EVAL]
Y = tr[COLS].values.astype(np.float32)
X = np.load(f"{ROOT}/work/cache/train_{args.cache}.npy", mmap_mode="r")
XT = np.load(f"{ROOT}/work/cache/test_{args.cache}.npy", mmap_mode="r")
MASK_POS = np.full(len(tr), -1, np.int64)
if args.seg_w > 0:
    MASKS = torch.from_numpy(np.load(f"{ROOT}/work/cache/masks_128.npy"))
    MASK_POS[np.load(f"{ROOT}/work/cache/masks_128_idx.npy")] = np.arange(len(MASKS))
if args.full:
    tr_idx = np.arange(len(tr)); va_idx = np.array([], int)
else:
    tr_idx = np.where(folds.fold != args.fold)[0]; va_idx = np.where(folds.fold == args.fold)[0]


def batches(arr, idx, bs, shuffle):
    """Background thread gathering uint8 batches from the memmap."""
    idx = np.random.permutation(idx) if shuffle else idx
    chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle and len(chunks[-1]) < bs: chunks = chunks[:-1]
    q = queue.Queue(maxsize=6)

    def work():
        for c in chunks:
            s = np.sort(c)
            q.put((s, torch.from_numpy(np.ascontiguousarray(arr[s]))))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while True:
        it = q.get()
        if it is None: return
        yield it


def gpu_aug(x):
    # x: B,1,H,W float in [0,1]
    B = x.shape[0]
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(12)
    sc = 1 + (torch.rand(B, device=dev) * 2 - 1) * 0.15
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.08
    ty = (torch.rand(B, device=dev) * 2 - 1) * 0.08
    flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, (B, 1, args.size, args.size), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    gpu_aug.theta = theta
    # brightness / contrast / gamma
    g = torch.exp((torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.25)
    x = x.clamp(0, 1) ** g
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.2
    b = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.1
    m = x.mean((2, 3), keepdim=True)
    x = (x - m) * c + m + b
    # coarse dropout: one random box per image with p=0.5
    S = args.size
    hh = (torch.rand(B, device=dev) * 0.25 * S).long(); ww = (torch.rand(B, device=dev) * 0.25 * S).long()
    y0 = (torch.rand(B, device=dev) * S).long(); x0 = (torch.rand(B, device=dev) * S).long()
    ar = torch.arange(S, device=dev)
    my = (ar[None] >= y0[:, None]) & (ar[None] < (y0 + hh)[:, None])
    mx = (ar[None] >= x0[:, None]) & (ar[None] < (x0 + ww)[:, None])
    keep = torch.rand(B, device=dev) < 0.5
    box = (my[:, :, None] & mx[:, None, :] & keep[:, None, None])[:, None]
    x = x.masked_fill(box, 0.0)
    return x


def prep(xb, train):
    x = xb.to(dev, non_blocking=True).float().div_(255).unsqueeze(1)
    if train:
        x = gpu_aug(x)
    elif x.shape[-1] != args.size:
        x = F.interpolate(x, size=(args.size, args.size), mode="bilinear", antialias=True, align_corners=False)
    x = (x - 0.5) / 0.25
    return x.contiguous(memory_format=torch.channels_last)


model = timm.create_model(args.arch, pretrained=True, num_classes=11, in_chans=1, drop_path_rate=args.drop_path)


class SegHead(nn.Module):
    """Light FPN-ish head on stride-8 and stride-16 features -> 4 line-type logits at stride 8."""
    def __init__(self, c8, c16):
        super().__init__()
        self.l8 = nn.Conv2d(c8, 64, 1); self.l16 = nn.Conv2d(c16, 64, 1)
        self.out = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 4, 1))

    def forward(self, f8, f16):
        return self.out(self.l8(f8) + F.interpolate(self.l16(f16), size=f8.shape[-2:], mode="nearest"))


feats = {}
if args.seg_w > 0:
    fi = {f["reduction"]: f for f in model.feature_info}
    mods = dict(model.named_modules())
    for r in (8, 16):
        mods[fi[r]["module"]].register_forward_hook(lambda m, i, o, r=r: feats.__setitem__(r, o))
    model.seg_head = SegHead(fi[8]["num_chs"], fi[16]["num_chs"])
model = model.to(dev).to(memory_format=torch.channels_last)
ema = timm.utils.ModelEmaV2(model, decay=args.ema)
_ema_f = [v for v in ema.module.state_dict().values() if v.dtype.is_floating_point]
_mod_f = [v for v in model.state_dict().values() if v.dtype.is_floating_point]


@torch.no_grad()
def ema_update():  # fused foreach EMA (timm's per-tensor loop is CPU/launch bound here)
    torch._foreach_lerp_(_ema_f, _mod_f, 1 - args.ema)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
steps_per_epoch = len(tr_idx) // args.bs
total = steps_per_epoch * args.epochs; warm = int(steps_per_epoch * args.warmup)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
crit = nn.BCEWithLogitsLoss()


@torch.no_grad()
def predict(m, arr, idx):
    m.eval(); out = np.zeros((len(idx), 11), np.float32)
    pos = {v: k for k, v in enumerate(idx)}
    for s, xb in batches(arr, idx, 64, False):
        x = prep(xb, False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x).float().sigmoid() + m(torch.flip(x, dims=[3])).float().sigmoid()
        out[[pos[i] for i in s]] = (lo / 2).cpu().numpy()
    return out


def score(y, p):
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in EVAL_IDX]))


curve = f"{ROOT}/work/curves/{args.name}.csv"
with open(curve, "w") as f: f.write("step,train_metric,val_metric\n")
ck = f"{ROOT}/work/ckpt/{args.name}.pt"; pdir = f"{ROOT}/work/preds"
print(vars(args), len(tr_idx), len(va_idx), flush=True)
t0 = time.time(); step = 0
for ep in range(args.epochs):
    model.train(); tl, n = 0.0, 0
    for s, xb in batches(X, tr_idx, args.bs, True):
        x = prep(xb, True); y = torch.from_numpy(Y[s]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(model(x).float(), y)
            mp = MASK_POS[s]; has = mp >= 0
            if args.seg_w > 0:
                # fixed full-batch shapes (avoids cudnn re-tuning); loss masked to annotated samples
                mk = torch.zeros((len(s), 4, 128, 128), dtype=torch.uint8)
                if has.any(): mk[torch.from_numpy(np.where(has)[0])] = MASKS[mp[has]]
                mk = mk.to(dev, non_blocking=True).float()
                g = F.affine_grid(gpu_aug.theta, (len(s), 4, args.size // 4, args.size // 4), align_corners=False)
                mk = F.max_pool2d(F.grid_sample(mk, g, mode="bilinear", align_corners=False), 2)
                mk = (mk > 0.25).float()
                sl = model.seg_head(feats[8], feats[16]).float()
                if sl.shape[-1] != mk.shape[-1]: mk = F.adaptive_max_pool2d(mk, sl.shape[-2:])
                w = torch.from_numpy(has.astype(np.float32)).to(dev)[:, None, None, None]
                sel = F.binary_cross_entropy_with_logits(sl, mk, pos_weight=torch.tensor(5.0, device=dev), reduction="none")
                loss = loss + args.seg_w * (sel * w).sum() / (w.sum() * sel[0].numel()).clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step(); sched.step(); ema_update(); step += 1
        tl += loss.item(); n += 1
        if os.environ.get("PROF") and step == 20:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]); prof.start(); tp = time.time()
        if os.environ.get("PROF") and step == 40:
            prof.stop(); print("20 steps", time.time() - tp); print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25)); raise SystemExit
        if step % 200 == 0: print(f"ep {ep} step {step} loss {tl/n:.4f} {time.time()-t0:.0f}s", flush=True)
    val = float("nan")
    if len(va_idx):
        pv = predict(ema.module, X, va_idx); val = score(Y[va_idx], pv)
        np.save(f"{pdir}/{args.name}_val.npy", pv)
    last = ep + 1 == args.epochs or (time.time() - t0) / 60 > args.max_minutes
    if (ep + 1) % 3 == 0 or last:
        pt = predict(ema.module, XT, np.arange(len(XT)))
        np.save(f"{pdir}/{args.name}_test.npy", pt)
        torch.save(ema.module.state_dict(), ck)
    with open(curve, "a") as f: f.write(f"{ep+1},{tl/n:.5f},{val:.5f}\n")
    print(f"=== ep {ep+1} train_loss {tl/n:.4f} val_auc9 {val:.5f} elapsed {(time.time()-t0)/60:.1f}m", flush=True)
    if (time.time() - t0) / 60 > args.max_minutes:
        print("time limit hit, stopping", flush=True); break
json.dump({"name": args.name, "args": vars(args), "val_auc9": val, "minutes": (time.time() - t0) / 60, "epochs_done": ep + 1},
          open(f"{pdir}/{args.name}_meta.json", "w"))
print("done", flush=True)
os._exit(0)
