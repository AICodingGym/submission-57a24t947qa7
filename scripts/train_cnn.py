"""2.5D CNN on crops (3 gray frames + helmet mask channel). One fold per process.
Usage: CUDA device chosen via --gpu index within CUDA_VISIBLE_DEVICES
python scripts/train_cnn.py --fold 0 --nfolds 2 --gpu 0 --tag cnn1
"""
import argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

ap = argparse.ArgumentParser()
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--nfolds', type=int, default=2)
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--tag', default='cnn1')
ap.add_argument('--arch', default='resnet18.a1_in1k')
ap.add_argument('--epochs', type=int, default=3)
ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--res', type=int, default=128)
ap.add_argument('--pfx', default='')
ap.add_argument('--full', type=int, default=0, help='train on all train data (no holdout)')
args = ap.parse_args()
torch.manual_seed(args.fold)
np.random.seed(args.fold)
dev = torch.device(f'cuda:{args.gpu}')
S = np.load(f'work/crops/{args.pfx}test_crops.npy', mmap_mode='r').shape[-1]
t0 = time.time()

def load(split):
    idx = pd.read_parquet(f'work/crops/{args.pfx}{split}_index.parquet')
    arr = np.load(f'work/crops/{args.pfx}{split}_crops.npy')
    s = idx['size'].values
    m = np.stack([(idx.cx1 - idx.x0) / s * S, (idx.cy1 - idx.y0) / s * S, idx.h1 / s * S,
                  (idx.cx2 - idx.x0) / s * S, (idx.cy2 - idx.y0) / s * S, idx.h2 / s * S], 1).astype(np.float32)
    m = np.nan_to_num(m, nan=-100.0)
    extra = np.stack([idx.is_g.values, (idx.view == 'Endzone').values], 1).astype(np.float32)
    return idx, arr, m, extra

tr_idx, tr_arr, tr_m, tr_x = load('train')
te_idx, te_arr, te_m, te_x = load('test')
NCH = tr_arr.shape[1] + 1
gps = np.array(sorted(tr_idx.game_play.unique()))
rng = np.random.RandomState(42)
perm = rng.permutation(len(gps))
fold_of = {g: i % args.nfolds for g, i in zip(gps[perm], range(len(gps)))}
fo = tr_idx.game_play.map(fold_of).values
if args.full:
    trn = np.arange(len(tr_idx)); val = np.array([], dtype=int)
else:
    trn = np.where(fo != args.fold)[0]; val = np.where(fo == args.fold)[0]
y = tr_idx.contact.values.astype(np.float32)
print('train', len(trn), 'val', len(val), 'pos rate', y[trn].mean(), flush=True)

yy, xx = torch.meshgrid(torch.arange(S, device=dev).float(), torch.arange(S, device=dev).float(), indexing='ij')

def make_batch(arr, m, ex, ids, aug):
    ids = np.sort(ids)
    img = torch.from_numpy(arr[ids]).to(dev).float() / 255.0
    mm = torch.from_numpy(m[ids]).to(dev)
    sig1 = (mm[:, 2] * 0.5).clamp(min=1.0)[:, None, None]
    sig2 = (mm[:, 5] * 0.5).clamp(min=1.0)[:, None, None]
    g1 = torch.exp(-((xx - mm[:, 0, None, None]) ** 2 + (yy - mm[:, 1, None, None]) ** 2) / (2 * sig1 ** 2))
    g2 = torch.exp(-((xx - mm[:, 3, None, None]) ** 2 + (yy - mm[:, 4, None, None]) ** 2) / (2 * sig2 ** 2))
    mask = torch.maximum(g1, g2)[:, None]
    x = torch.cat([(img - 0.45) / 0.25, mask * 2 - 1], 1)
    if aug:
        flip = torch.rand(len(ids), device=dev) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        a = 1 + 0.2 * (torch.rand(len(ids), 1, 1, 1, device=dev) - 0.5) * 2
        b = 0.2 * (torch.rand(len(ids), 1, 1, 1, device=dev) - 0.5) * 2
        x[:, :-1] = x[:, :-1] * a + b
        sx, sy = np.random.randint(-4, 5, 2)
        x = torch.roll(x, shifts=(int(sy), int(sx)), dims=(2, 3))
    x = F.interpolate(x, size=(args.res, args.res), mode='bilinear', align_corners=False)
    return x, torch.from_numpy(ex[ids]).to(dev), ids


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = timm.create_model(args.arch, pretrained=True, in_chans=NCH, num_classes=0)
        self.h = nn.Sequential(nn.Linear(self.b.num_features + 2, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, x, e):
        return self.h(torch.cat([self.b(x), e], 1)).squeeze(1)


net = Net().to(dev).to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
steps_per_epoch = len(trn) // args.bs
total = steps_per_epoch * args.epochs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)


@torch.no_grad()
def predict(arr, m, ex, ids):
    net.eval()
    out = np.zeros(len(ids), np.float32)
    order = np.argsort(ids)
    sids = ids[order]
    res = []
    for i in range(0, len(sids), 1024):
        x, e, _ = make_batch(arr, m, ex, sids[i:i + 1024], False)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = (torch.sigmoid(net(x, e).float()) + torch.sigmoid(net(x.flip(3), e).float())) / 2
        res.append(p.float().cpu().numpy())
    out[order] = np.concatenate(res)
    net.train()
    return out


step = 0
for ep in range(args.epochs):
    p = np.random.permutation(trn)
    tl = 0
    for i in range(steps_per_epoch):
        x, e, ids = make_batch(tr_arr, tr_m, tr_x, p[i * args.bs:(i + 1) * args.bs], True)
        yt = torch.from_numpy(y[ids]).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.binary_cross_entropy_with_logits(net(x, e).float(), yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        tl += loss.item()
        if i % 500 == 0:
            print(f'ep {ep} it {i}/{steps_per_epoch} loss {tl / (i + 1):.4f} {time.time() - t0:.0f}s', flush=True)
    if len(val):
        from sklearn.metrics import matthews_corrcoef, roc_auc_score
        pv = predict(tr_arr, tr_m, tr_x, val)
        best = max((matthews_corrcoef(y[val], pv > t), t) for t in np.arange(0.2, 0.8, 0.05))
        print(f'ep {ep} val auc {roc_auc_score(y[val], pv):.4f} mcc(rowview) {best}', flush=True)

name = f'{args.tag}_f{args.fold}' if not args.full else f'{args.tag}_full'
torch.save(net.state_dict(), f'work/{name}.pt')
if len(val):
    pv = predict(tr_arr, tr_m, tr_x, val)
    pd.DataFrame({'row': val, 'contact_id': tr_idx.contact_id.values[val], 'view': tr_idx.view.values[val], 'p': pv}).to_parquet(f'work/{name}_oof.parquet')
pt = predict(te_arr, te_m, te_x, np.arange(len(te_idx)))
pd.DataFrame({'contact_id': te_idx.contact_id.values, 'view': te_idx.view.values, 'p': pt}).to_parquet(f'work/{name}_test.parquet')
print('done', (time.time() - t0) / 60, 'min')
