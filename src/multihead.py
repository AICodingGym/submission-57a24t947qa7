"""Joint residual head over several backbones:
logits = sum_m t_m * zs_logits_m + concat_m(normed feat_m) @ W + b   (4 views as train aug / TTA)
5-fold CV, same folds as heads.py.  usage: python multihead.py --wd X --name NAME tag1 tag2 ..."""
import os, time, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE = os.path.join(ROOT, 'work', 'feats')
VIEWS = ['crop', 'crop_flip', 'squash', 'squash_flip']
ap = argparse.ArgumentParser()
ap.add_argument('tags', nargs='+')
ap.add_argument('--wd', type=float, default=1e-2)
ap.add_argument('--epochs', type=int, default=400)
ap.add_argument('--lr', type=float, default=0.01)
ap.add_argument('--name', required=True)
ap.add_argument('--nofeat', action='store_true', help='only learn per-model temperatures + bias')
args = ap.parse_args()
dev = 'cuda'
lab = pd.read_csv(os.path.join(ROOT, 'work', 'cache', 'train_ids.csv'))
classes = list(pd.read_csv(os.path.join(ROOT, 'data', 'sample_submission.csv'), nrows=1).columns[1:])
y = lab.breed.map({c: i for i, c in enumerate(classes)}).values
folds = np.load(os.path.join(ROOT, 'work', 'folds.npy'))


def load(split):
    fs, ls = [], []
    for t in args.tags:
        if not args.nofeat:
            f = torch.tensor(np.stack([np.load(f'{FE}/{t}/{split}_{v}_feat.npy').astype(np.float32) for v in VIEWS]))
            fs.append(F.normalize(f, dim=-1) * 10 / np.sqrt(len(args.tags)))
        ls.append(torch.tensor(np.stack([np.load(f'{FE}/{t}/{split}_{v}_logit.npy') for v in VIEWS])))
    X = torch.cat(fs, -1).to(dev) if fs else None
    return X, torch.stack(ls, 0).to(dev)  # M,V,N,120


Xtr, Ltr = load('train')
Xte, Lte = load('test')
yt = torch.tensor(y, device=dev)
M = Ltr.shape[0]
D = Xtr.shape[-1] if Xtr is not None else 0


def fwd(X, Lg, P):
    t, W, b = P
    z = torch.einsum('m,mvnc->vnc', t, Lg) + b
    if X is not None:
        z = z + X @ W
    return z


def fit(X, Lg, yy):
    V = Lg.shape[1]
    t = torch.full((M,), 1.0 / M, device=dev, requires_grad=True)
    W = torch.zeros(max(D, 1), 120, device=dev, requires_grad=True)
    b = torch.zeros(120, device=dev, requires_grad=True)
    opt = torch.optim.Adam([t, W, b], lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for _ in range(args.epochs):
        opt.zero_grad()
        z = fwd(X, Lg, (t, W, b))
        loss = F.cross_entropy(z.reshape(-1, 120), yy.repeat(V)) + args.wd * (W ** 2).sum()
        loss.backward(); opt.step(); sch.step()
    return (t.detach(), W.detach(), b.detach()), loss.item()


def pred(X, Lg, P):
    return torch.softmax(fwd(X, Lg, P), -1).mean(0)


curve = open(os.path.join(ROOT, 'work', 'curves', args.name + '.csv'), 'w')
curve.write('step,train_metric,val_metric\n')
t0 = time.time()
oof = np.zeros((len(y), 120), np.float32)
for k in range(5):
    tr, va = torch.tensor(folds != k, device=dev), torch.tensor(folds == k, device=dev)
    P, trl = fit(Xtr[:, tr] if D else None, Ltr[:, :, tr], yt[tr])
    p = pred(Xtr[:, va] if D else None, Ltr[:, :, va], P).cpu().numpy()
    oof[folds == k] = p
    vl = -np.log(np.clip(p[np.arange(len(p)), y[folds == k]], 1e-15, 1)).mean()
    curve.write(f'{k},{trl:.5f},{vl:.5f}\n'); curve.flush()
P, _ = fit(Xtr if D else None, Ltr, yt)
te = pred(Xte if D else None, Lte, P).cpu().numpy()
ll = -np.log(np.clip(oof[np.arange(len(y)), y], 1e-15, 1)).mean()
print('t', P[0].cpu().numpy().round(3))
np.save(os.path.join(ROOT, 'work', 'preds', args.name + '_oof.npy'), oof)
np.save(os.path.join(ROOT, 'work', 'preds', args.name + '_test.npy'), te)
print(f'RESULT {args.name} oof_logloss {ll:.5f} acc {(oof.argmax(1) == y).mean():.4f} time {time.time() - t0:.0f}s', flush=True)
