"""Blend prediction sets by weighted geometric mean (log-prob space, non-negative weights; their sum acts as
inverse temperature). Weights fit on the validation rows common to all inputs.
Inputs: names of work/preds/<name>_{oof,test}.npy (OOF over all train rows) or work/ft/<name> dirs
(val_probs over val_idx rows). For ft runs over several folds, pass  ft:<name_f0>+<name_f1>+...  to stitch.
usage: python blend.py [--out submission.csv] [--cv] name1 name2 ..."""
import os, sys, argparse, numpy as np, pandas as pd
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('names', nargs='+')
ap.add_argument('--out', default=None)
ap.add_argument('--cv', action='store_true', help='report 5-fold CV of the blending itself')
ap.add_argument('--json', default=None, help='write blend summary json')
ap.add_argument('--save', default=None, help='save blended OOF/test probs as work/preds/<save>_{oof,test}.npy')
args = ap.parse_args()

lab = pd.read_csv(os.path.join(ROOT, 'work', 'cache', 'train_ids.csv'))
sub = pd.read_csv(os.path.join(ROOT, 'data', 'sample_submission.csv'))
classes = list(sub.columns[1:])
y = lab.breed.map({c: i for i, c in enumerate(classes)}).values
N = len(y)


def load(n):
    if n.startswith('ft:'):
        val = np.full((N, 120), np.nan, np.float32)
        tes = []
        for part in n[3:].split('+'):
            d = os.path.join(ROOT, 'work', 'ft', part)
            idx = np.load(f'{d}/val_idx.npy')
            val[idx] = np.load(f'{d}/val_probs.npy')
            tes.append(np.load(f'{d}/test_probs.npy'))
        te = np.exp(np.mean([np.log(np.clip(t, 1e-8, 1)) for t in tes], 0))
        return val, te / te.sum(1, keepdims=True)
    d = os.path.join(ROOT, 'work', 'preds')
    return np.load(f'{d}/{n}_oof.npy'), np.load(f'{d}/{n}_test.npy')


vals, tests = zip(*[load(n) for n in args.names])
L = np.stack([np.log(np.clip(v, 1e-8, 1)) for v in vals])  # M,N,120
LT = np.stack([np.log(np.clip(t, 1e-8, 1)) for t in tests])
rows = ~np.isnan(L).any((0, 2))
print('rows used', rows.sum())


def ll_of(w, Ls, yy):
    z = np.tensordot(w, Ls, 1)
    z = z - z.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return -lp[np.arange(len(yy)), yy].mean()


def fit(Ls, yy):
    # projected gradient (Adam) on non-negative weights, analytic grads via torch
    import torch
    Lt = torch.tensor(Ls, dtype=torch.float32)
    yt = torch.tensor(yy)
    M = Ls.shape[0]
    w = torch.full((M,), 1.0 / M, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.02)
    for it in range(400):
        opt.zero_grad()
        z = torch.einsum('m,mnc->nc', w, Lt)
        loss = torch.nn.functional.cross_entropy(z, yt)
        loss.backward(); opt.step()
        with torch.no_grad():
            w.clamp_(0, 5)
        if it == 300:
            for g in opt.param_groups: g['lr'] = 0.005
    return w.detach().numpy().astype(np.float64)


for n, Lm in zip(args.names, L):
    print(f'{n:70s} single {ll_of(np.ones(1), Lm[None][:, rows], y[rows]):.5f} acc {(Lm[rows].argmax(1) == y[rows]).mean():.4f}')
w = fit(L[:, rows], y[rows])
print('weights', np.round(w, 3))
print(f'BLEND in-sample logloss {ll_of(w, L[:, rows], y[rows]):.5f}')
if args.cv:
    folds = np.load(os.path.join(ROOT, 'work', 'folds.npy'))[rows]
    Lr, yr = L[:, rows], y[rows]
    tot = 0
    for k in range(5):
        wk = fit(Lr[:, folds != k], yr[folds != k])
        tot += ll_of(wk, Lr[:, folds == k], yr[folds == k]) * (folds == k).sum()
    print(f'BLEND cv logloss {tot / len(yr):.5f}')
def softmax_np(z):
    z = z - z.max(-1, keepdims=True); p = np.exp(z); return p / p.sum(-1, keepdims=True)


if args.save:
    np.save(os.path.join(ROOT, 'work', 'preds', args.save + '_oof.npy'), softmax_np(np.tensordot(w, L, 1)).astype(np.float32))
    np.save(os.path.join(ROOT, 'work', 'preds', args.save + '_test.npy'), softmax_np(np.tensordot(w, LT, 1)).astype(np.float32))
if args.json:
    import json
    singles = {n: float(ll_of(np.ones(1), Lm[None][:, rows], y[rows])) for n, Lm in zip(args.names, L)}
    json.dump({'weights': {n: float(wi) for n, wi in zip(args.names, w) if wi > 1e-3},
               'in_sample': float(ll_of(w, L[:, rows], y[rows])), 'cv': float(tot / len(yr)) if args.cv else None,
               'best_single': min(singles.values()), 'singles': singles}, open(args.json, 'w'), indent=1)
if args.out:
    z = np.tensordot(w, LT, 1)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    out = pd.DataFrame(p, columns=classes)
    out.insert(0, 'id', sub.id.values)
    out.to_csv(args.out, index=False)
    print('wrote', args.out, out.shape)
