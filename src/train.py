# 2.5D UNet training, all data + augmentation on GPU (only 4 CPUs on this box).
import argparse, os, time, math, json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = f'{ROOT}/work/cache'
p = argparse.ArgumentParser()
p.add_argument('--fold', type=int, default=0)        # -1 = train on everything
p.add_argument('--nfolds', type=int, default=5)
p.add_argument('--enc', default='tu-efficientnet_b3.ra2_in1k')
p.add_argument('--arch', default='Unet')
p.add_argument('--epochs', type=int, default=15)
p.add_argument('--bs', type=int, default=24)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--offsets', default='-2,-1,0,1,2')
p.add_argument('--dev', type=int, default=0)
p.add_argument('--name', default='run')
p.add_argument('--max_steps', type=int, default=0)   # smoke test
p.add_argument('--seed', type=int, default=42)
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device(f'cuda:{a.dev}')
OUT = f'{ROOT}/work/runs/{a.name}'; os.makedirs(OUT, exist_ok=True)
offs = [int(o) for o in a.offsets.split(',')]

meta = pd.read_csv(f'{C}/train_meta.csv')
N = len(meta)
# volume bounds for neighbor lookup
start = meta.groupby('day').cumcount().values; vstart = np.arange(N) - start
vlen = meta.groupby('day').slice.transform('size').values
nb = np.stack([np.clip(np.arange(N) + o, vstart, vstart + vlen - 1) for o in offs], 1)
nb = torch.from_numpy(nb).to(dev)

folds = np.full(N, -1)
for k, (_, vi) in enumerate(GroupKFold(a.nfolds).split(meta, groups=meta.case)):
    folds[vi] = k
np.save(f'{C}/folds.npy', folds)
tr_idx = np.where(folds != a.fold)[0]; va_idx = np.where(folds == a.fold)[0]
print(f'train {len(tr_idx)} val {len(va_idx)}', flush=True)

t0 = time.time()
X = torch.from_numpy(np.load(f'{C}/train_img.npy', mmap_mode='r')[:]).to(dev)       # N,S,S fp16
Y = torch.from_numpy(np.load(f'{C}/train_mask.npy', mmap_mode='r')[:]).to(dev)      # N,S,S packed bits
S = X.shape[-1]
print(f'loaded to gpu {time.time()-t0:.0f}s', flush=True)
bits = torch.tensor([1, 2, 4], device=dev, dtype=torch.uint8).view(1, 3, 1, 1)


def get_batch(idx):
    idx = torch.as_tensor(idx, device=dev)
    x = X[nb[idx]].float()                                   # B,K,S,S
    y = ((Y[idx].unsqueeze(1) & bits) > 0).float()           # B,3,S,S
    return x, y


def augment(x, y):
    B = x.shape[0]
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(15)
    sc = torch.exp((torch.rand(B, device=dev) * 2 - 1) * math.log(1.2))
    sx = sc * torch.exp((torch.rand(B, device=dev) * 2 - 1) * 0.08)
    sy = sc * torch.exp((torch.rand(B, device=dev) * 2 - 1) * 0.08)
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.1; ty = (torch.rand(B, device=dev) * 2 - 1) * 0.1
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = torch.cos(ang) / sx; th[:, 0, 1] = -torch.sin(ang) / sx; th[:, 0, 2] = tx
    th[:, 1, 0] = torch.sin(ang) / sy; th[:, 1, 1] = torch.cos(ang) / sy; th[:, 1, 2] = ty
    keep = torch.rand(B, device=dev) < 0.2                   # some unaugmented samples
    th[keep] = torch.tensor([[1., 0, 0], [0, 1, 0]], device=dev)
    g = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, g, mode='bilinear', padding_mode='zeros', align_corners=False)
    y = (F.grid_sample(y, g, mode='bilinear', padding_mode='zeros', align_corners=False) > 0.5).float()
    # intensity: contrast/brightness/gamma/noise
    c = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.2
    b = (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.1
    gm = torch.exp((torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.3)
    x = (x.clamp(min=0) ** gm) * c + b
    x = x + torch.randn_like(x) * 0.02 * (torch.rand(B, 1, 1, 1, device=dev) < 0.3)
    return x, y


model = getattr(smp, a.arch)(a.enc, encoder_weights='imagenet', in_channels=len(offs), classes=3).to(dev)
model = model.to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
steps_per_epoch = len(tr_idx) // a.bs
total = steps_per_epoch * a.epochs if not a.max_steps else a.max_steps
warm = min(500, total // 10)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))


def loss_fn(lo, y):
    bce = F.binary_cross_entropy_with_logits(lo, y)
    pr = lo.sigmoid()
    inter = (pr * y).sum((0, 2, 3)); den = pr.sum((0, 2, 3)) + y.sum((0, 2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    return bce + dice


@torch.no_grad()
def predict(idx, bs=64):
    model.eval(); out = []
    for i in range(0, len(idx), bs):
        x, _ = get_batch(idx[i:i + bs])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pr = model(x.contiguous(memory_format=torch.channels_last)).float().sigmoid()
        out.append(pr)
    model.train()
    return torch.cat(out)


@torch.no_grad()
def val_dice(idx, thr=0.5):
    # per slice-class dice, empty-empty = 1 (Kaggle convention); 2D only, quick proxy
    ds = []
    for i in range(0, len(idx), 256):
        sub = idx[i:i + 256]
        pr = predict(sub) > thr
        _, y = get_batch(sub); y = y > 0.5
        inter = (pr & y).sum((2, 3)).float(); den = pr.sum((2, 3)) + y.sum((2, 3))
        d = torch.where(den == 0, torch.ones_like(inter), 2 * inter / den.clamp(min=1))
        ds.append(d)
    return torch.cat(ds).mean(0).cpu().numpy()


step = 0; log = []
model.train()
for ep in range(a.epochs):
    perm = np.random.permutation(tr_idx); te = time.time(); tl = 0
    for i in range(steps_per_epoch):
        x, y = get_batch(perm[i * a.bs:(i + 1) * a.bs])
        x, y = augment(x, y)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lo = model(x.contiguous(memory_format=torch.channels_last))
        loss = loss_fn(lo.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step(); step += 1; tl += loss.item() if i % 50 == 0 else 0
        if a.max_steps and step >= a.max_steps: break
    msg = dict(ep=ep, loss=tl / max(1, steps_per_epoch // 50), sec=time.time() - te)
    if len(va_idx) and (ep >= a.epochs - 4 or ep % 3 == 0 or a.max_steps):
        d = val_dice(va_idx); msg.update(dice=d.tolist(), dmean=float(d.mean()))
    print(json.dumps(msg), flush=True); log.append(msg)
    if a.max_steps and step >= a.max_steps: break
torch.save(model.state_dict(), f'{OUT}/model.pt')
json.dump(dict(vars(a), log=log), open(f'{OUT}/log.json', 'w'))
if len(va_idx):
    pr = np.concatenate([(predict(va_idx[i:i + 512]) * 255).round().byte().cpu().numpy() for i in range(0, len(va_idx), 512)])
    np.save(f'{OUT}/oof_idx.npy', va_idx)
    np.save(f'{OUT}/oof.npy', pr)
print('done', time.time() - t0, flush=True)
