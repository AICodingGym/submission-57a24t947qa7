"""Linear heads on frozen features, 5-fold CV. Produces OOF + test probabilities per feature set.
usage: python heads.py <feat_tag> [--mode lr|zs|lr+logit] [--wd X]
 zs       : zero-shot = softmax(T * imagenet dog logits), T fit on OOF
 lr       : multinomial logistic regression on L2-normalised pooled features (all 4 views used as train aug)
 lr+logit : lr on [features, imagenet dog logits]"""
import os, sys, time, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE = os.path.join(ROOT, 'work', 'feats')
VIEWS = ['crop', 'crop_flip', 'squash', 'squash_flip']

ap = argparse.ArgumentParser()
ap.add_argument('tag')
ap.add_argument('--mode', default='lr')
ap.add_argument('--wd', type=float, default=1e-4)
ap.add_argument('--epochs', type=int, default=300)
ap.add_argument('--views', default=','.join(VIEWS))
ap.add_argument('--k', type=int, default=1000)
args = ap.parse_args()
views = args.views.split(',')

lab = pd.read_csv(os.path.join(ROOT, 'work', 'cache', 'train_ids.csv'))
classes = list(pd.read_csv(os.path.join(ROOT, 'data', 'sample_submission.csv'), nrows=1).columns[1:])
y = lab.breed.map({c: i for i, c in enumerate(classes)}).values
fold_path = os.path.join(ROOT, 'work', 'folds.npy')
if not os.path.exists(fold_path):
    folds = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(y, y)):
        folds[va] = k
    np.save(fold_path, folds)
folds = np.load(fold_path)
dev = 'cuda'


def load(split, kind):
    return np.stack([np.load(f'{FE}/{args.tag}/{split}_{v}_{kind}.npy').astype(np.float32) for v in views])  # V,N,D


COLS = None
if args.mode in ('full', 'featfull'):
    # unsupervised column selection: the K 22k classes with most mean probability mass over train+test (crop view)
    mass = sum(np.exp(np.load(f'{FE}/{args.tag}/{s}_crop_full.npy').astype(np.float32)).sum(0) for s in ('train', 'test'))
    COLS = np.argsort(-mass)[:args.k]


def feats(split):
    parts = []
    if args.mode in ('lr', 'lr+logit', 'res', 'featfull'):
        f = torch.tensor(load(split, 'feat'))
        parts.append(F.normalize(f, dim=-1) * 10)
    if args.mode == 'resmid':
        f = F.normalize(torch.tensor(load(split, 'feat')), dim=-1)
        mid = torch.tensor(load(split, 'mid'))
        chunks = [F.normalize(c, dim=-1) for c in mid.chunk(3, -1)]
        parts.append(torch.cat([f] + chunks, -1) * 10 / 2)
    if COLS is not None:
        fl = torch.tensor(np.stack([np.load(f'{FE}/{args.tag}/{split}_{v}_full.npy')[:, COLS].astype(np.float32) for v in views]))
        parts.append(fl.clamp_min(-20) / 5)
    if args.mode in ('lr+logit', 'zs', 'res', 'full', 'featfull', 'resmid'):
        parts.append(torch.tensor(load(split, 'logit')) / 10)
    return torch.cat(parts, -1)


Xtr, Xte = feats('train').to(dev), feats('test').to(dev)
yt = torch.tensor(y, device=dev)
V, N, D = Xtr.shape
name = f'{args.tag}__{args.mode}' + (f'_wd{args.wd:g}' if args.mode != 'zs' else '') + (f'_k{args.k}' if COLS is not None else '')
curve = open(os.path.join(ROOT, 'work', 'curves', name + '.csv'), 'w')
curve.write('step,train_metric,val_metric\n')
t0 = time.time()


RES = args.mode in ('res', 'full', 'featfull', 'resmid')
DF = D - 120 if RES else D  # in res mode the last 120 columns are the (scaled) imagenet logits, used as an offset


def logits_of(X, W, b, t=None):
    if RES:
        return X[..., :DF] @ W + b + X[..., DF:] * 10 * t
    return X @ W + b


def fit_temp(logits, yy):
    # logits V,n,120 -> single temperature fit by grid
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(np.log(0.05), np.log(5), 200)):
        p = torch.softmax(logits * T, -1).mean(0)
        l = F.nll_loss(torch.log(p.clamp_min(1e-15)), yy).item()
        best = min(best, (l, T))
    return best[1]


def fit_lr(X, yy):
    # X: V,n,D -> treat views as extra samples; full-batch Adam on cross-entropy + L2
    Xf, yf = X.reshape(-1, D), yy.repeat(V)
    W = torch.zeros(DF, 120, device=dev, requires_grad=True)
    b = torch.zeros(120, device=dev, requires_grad=True)
    t0 = fit_temp(X[..., DF:] * 10, yy) if RES else 1.0  # start from the zero-shot temperature (BCE-trained heads need T far from 1)
    t = torch.full((), float(t0), device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b, t] if RES else [W, b], lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for it in range(args.epochs):
        opt.zero_grad()
        loss = F.cross_entropy(logits_of(Xf, W, b, t), yf) + args.wd * (W ** 2).sum()
        loss.backward(); opt.step(); sched.step()
    return (W.detach(), b.detach(), t.detach()), loss.item()


def predict(X, P):
    return torch.softmax(logits_of(X, *P), -1).mean(0)  # average probs over views


oof = np.zeros((N, 120), np.float32)
if args.mode == 'zs':
    for k in range(5):
        tr, va = folds != k, folds == k
        T = fit_temp(Xtr[:, tr] * 10, yt[tr])
        oof[va] = torch.softmax(Xtr[:, va] * 10 * T, -1).mean(0).cpu().numpy()
        curve.write(f'{k},nan,{-np.log(np.clip(oof[va][np.arange(va.sum()), y[va]], 1e-15, 1)).mean():.5f}\n'); curve.flush()
    T = fit_temp(Xtr * 10, yt)
    te = torch.softmax(Xte * 10 * T, -1).mean(0).cpu().numpy()
    print('T', T)
else:
    for k in range(5):
        tr, va = torch.tensor(folds != k, device=dev), torch.tensor(folds == k, device=dev)
        P, trl = fit_lr(Xtr[:, tr], yt[tr])
        p = predict(Xtr[:, va], P).cpu().numpy()
        oof[folds == k] = p
        vl = -np.log(np.clip(p[np.arange(len(p)), y[folds == k]], 1e-15, 1)).mean()
        curve.write(f'{k},{trl:.5f},{vl:.5f}\n'); curve.flush()
    P, _ = fit_lr(Xtr, yt)
    te = predict(Xte, P).cpu().numpy()

ll = -np.log(np.clip(oof[np.arange(N), y], 1e-15, 1)).mean()
acc = (oof.argmax(1) == y).mean()
os.makedirs(os.path.join(ROOT, 'work', 'preds'), exist_ok=True)
np.save(os.path.join(ROOT, 'work', 'preds', name + '_oof.npy'), oof)
np.save(os.path.join(ROOT, 'work', 'preds', name + '_test.npy'), te)
print(f'RESULT {name} oof_logloss {ll:.5f} acc {acc:.4f} time {time.time() - t0:.0f}s', flush=True)
