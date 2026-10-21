"""Full fine-tune of an ImageNet-1k timm classifier on one CV fold (or all data with --fold -1).
Head initialised from the ImageNet head rows of the 120 dog classes, so epoch 0 == zero-shot.
Augmentation is done on GPU from the uint8 cache (CPU is only 4 cores).
Each epoch: val logloss (crop+flip TTA) -> curves csv; best-epoch val/test probs + checkpoint saved."""
import os, sys, time, math, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, 'work', 'cache')
ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('--res', type=int, default=None)
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--epochs', type=int, default=5)
ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--head_lr', type=float, default=1e-4)
ap.add_argument('--wd', type=float, default=0.05)
ap.add_argument('--ld', type=float, default=0.85, help='layer-wise lr decay')
ap.add_argument('--ls', type=float, default=0.0)
ap.add_argument('--min_scale', type=float, default=0.35)
ap.add_argument('--gc', type=int, default=1, help='grad checkpointing')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--name', default=None)
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = 'cuda'

kw = {}
if args.res is not None and any(s in args.model for s in ('eva', 'vit', 'beit')):
    kw['img_size'] = args.res
m = timm.create_model(args.model, pretrained=True, drop_path_rate=0.1, **kw)
cfg = m.pretrained_cfg
res = args.res or cfg['input_size'][-1]
dog_idx = torch.tensor(np.load(os.path.join(C, 'dog_idx.npy')))
head = m.get_classifier()
with torch.no_grad():
    w, b = head.weight[dog_idx].clone(), head.bias[dog_idx].clone()
m.reset_classifier(120)
with torch.no_grad():
    m.get_classifier().weight.copy_(w); m.get_classifier().bias.copy_(b)
if args.gc:
    m.set_grad_checkpointing(True)
m = m.to(dev)
mean = torch.tensor(cfg['mean'], device=dev).view(1, 3, 1, 1)
std = torch.tensor(cfg['std'], device=dev).view(1, 3, 1, 1)

name = args.name or f"ft_{args.model.split('.')[0]}_{res}_f{args.fold}_s{args.seed}"
out = os.path.join(ROOT, 'work', 'ft', name)
os.makedirs(out, exist_ok=True)

lab = pd.read_csv(os.path.join(C, 'train_ids.csv'))
classes = list(pd.read_csv(os.path.join(ROOT, 'data', 'sample_submission.csv'), nrows=1).columns[1:])
y_all = lab.breed.map({c: i for i, c in enumerate(classes)}).values
folds = np.load(os.path.join(ROOT, 'work', 'folds.npy'))
tr_idx = np.where(folds != args.fold)[0]
va_idx = np.where(folds == args.fold)[0]
print(name, 'res', res, 'train', len(tr_idx), 'val', len(va_idx), flush=True)

# load cache into RAM (uint8)
tr_crop = np.load(os.path.join(C, 'train_crop448.npy'), mmap_mode='r')
tr_sq = np.load(os.path.join(C, 'train_squash448.npy'), mmap_mode='r')
te_crop = np.load(os.path.join(C, 'test_crop448.npy'), mmap_mode='r')
te_sq = np.load(os.path.join(C, 'test_squash448.npy'), mmap_mode='r')


def to_gpu(arr, idx):
    idx = np.sort(idx) if False else idx
    return torch.from_numpy(np.stack([arr[i] for i in idx])).to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.


def gpu_aug(x):
    """per-sample random resized crop + flip + small colour jitter via affine grid_sample"""
    B = x.shape[0]
    scale = torch.empty(B, device=dev).uniform_(args.min_scale, 1.0)
    logr = torch.empty(B, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3))
    ratio = torch.exp(logr)
    sw = torch.sqrt(scale * ratio).clamp(max=1.0)
    sh = torch.sqrt(scale / ratio).clamp(max=1.0)
    cx = (torch.rand(B, device=dev) * 2 - 1) * (1 - sw)
    cy = (torch.rand(B, device=dev) * 2 - 1) * (1 - sh)
    flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = sw * flip
    theta[:, 0, 2] = cx
    theta[:, 1, 1] = sh
    theta[:, 1, 2] = cy
    # downsample first with antialias when target res smaller than the cache
    if res < x.shape[-1]:
        x = F.interpolate(x, size=(res, res), mode='bilinear', antialias=True, align_corners=False)
    grid = F.affine_grid(theta, (B, 3, res, res), align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    bri = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.85, 1.15)
    con = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.85, 1.15)
    mu = x.mean((1, 2, 3), keepdim=True)
    x = ((x - mu) * con + mu) * bri
    return x.clamp(0, 1)


def prep_eval(x):
    if x.shape[-1] != res:
        x = F.interpolate(x, size=(res, res), mode='bicubic', antialias=True, align_corners=False).clamp(0, 1)
    return x


# optimiser with layer-wise lr decay
def param_groups():
    groups = {}
    nl = None
    try:
        from timm.optim._param_groups import param_groups_layer_decay
        return param_groups_layer_decay(m, weight_decay=args.wd, layer_decay=args.ld, no_weight_decay_list=getattr(m, 'no_weight_decay', lambda: set())())
    except Exception as e:
        print('no layer decay', e)
        return [{'params': [p for n, p in m.named_parameters()], 'weight_decay': args.wd}]


pgs = param_groups()
head_params = set(id(p) for p in m.get_classifier().parameters())
for g in pgs:
    g['lr_scale'] = g.get('lr_scale', 1.0)
    is_head = all(id(p) in head_params for p in g['params'])
    g['base_lr'] = args.head_lr if is_head else args.lr * g['lr_scale']
    g['lr'] = g['base_lr']
opt = torch.optim.AdamW(pgs, lr=args.lr, betas=(0.9, 0.999))
steps_per_epoch = len(tr_idx) // args.bs
total = steps_per_epoch * args.epochs
warm = steps_per_epoch // 2


def set_lr(step):
    f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    for g in opt.param_groups:
        g['lr'] = g['base_lr'] * f


@torch.no_grad()
def predict(arrs, idx):
    m.eval()
    outs = []
    for i in range(0, len(idx), 64):
        bi = idx[i:i + 64]
        logit_sum = 0
        for arr in arrs:
            x = prep_eval(to_gpu(arr, bi))
            x = (x - mean) / std
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for xx in (x, x.flip(3)):
                    logit_sum = logit_sum + torch.softmax(m(xx).float(), -1)
        outs.append((logit_sum / (2 * len(arrs))).cpu())
    m.train()
    return torch.cat(outs).numpy()


def logloss(p, yy):
    return float(-np.log(np.clip(p[np.arange(len(yy)), yy], 1e-15, 1)).mean())


curve = open(os.path.join(ROOT, 'work', 'curves', name + '.csv'), 'w')
curve.write('step,train_metric,val_metric\n')
t0 = time.time()
best = 1e9
has_val = len(va_idx) > 0
if has_val:
    pv = predict([tr_crop], va_idx)
    vl0 = logloss(pv, y_all[va_idx])
    print(f'epoch 0 (zero-shot) val {vl0:.4f} acc {(pv.argmax(1) == y_all[va_idx]).mean():.4f}', flush=True)
    curve.write(f'0,nan,{vl0:.5f}\n'); curve.flush()
step = 0
m.train()
for ep in range(1, args.epochs + 1):
    perm = np.random.permutation(tr_idx)
    tl, tn = 0.0, 0
    for i in range(steps_per_epoch):
        bi = perm[i * args.bs:(i + 1) * args.bs]
        use_sq = np.random.rand(len(bi)) < 0.5
        xc = to_gpu(tr_crop, bi)
        xs = to_gpu(tr_sq, bi)
        x = torch.where(torch.tensor(use_sq, device=dev).view(-1, 1, 1, 1), xs, xc)
        x = (gpu_aug(x) - mean) / std
        yy = torch.tensor(y_all[bi], device=dev)
        set_lr(step)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = m(x)
        loss = F.cross_entropy(logits.float(), yy, label_smoothing=args.ls)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        step += 1
        tl += loss.item() * len(bi); tn += len(bi)
        if i % 100 == 0:
            print(f'ep {ep} it {i}/{steps_per_epoch} loss {tl / tn:.4f} {time.time() - t0:.0f}s', flush=True)
    if has_val:
        pv = predict([tr_crop, tr_sq], va_idx)
        vl = logloss(pv, y_all[va_idx])
    else:
        vl = float('nan')
    curve.write(f'{ep},{tl / tn:.5f},{vl:.5f}\n'); curve.flush()
    print(f'epoch {ep} train {tl / tn:.4f} val {vl:.4f} acc {(pv.argmax(1) == y_all[va_idx]).mean() if has_val else 0:.4f} {time.time() - t0:.0f}s', flush=True)
    if (not has_val) or vl < best or ep == args.epochs:
        # save preds of best epoch (or every epoch for full-data runs; last overwrites)
        if has_val and vl < best:
            best = vl
            np.save(f'{out}/val_probs.npy', pv)
            np.save(f'{out}/val_idx.npy', va_idx)
            pt = predict([te_crop, te_sq], np.arange(len(te_crop)))
            np.save(f'{out}/test_probs.npy', pt)
            torch.save(m.state_dict(), f'{out}/best.pt')
        elif not has_val:
            pt = predict([te_crop, te_sq], np.arange(len(te_crop)))
            np.save(f'{out}/test_probs.npy', pt)
print(f'RESULT {name} best_val {best:.5f} time {time.time() - t0:.0f}s', flush=True)
