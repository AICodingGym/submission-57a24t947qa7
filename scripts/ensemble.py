"""Score runs from saved OOF preds, ensemble them, snap to the pressure grid and write submission.csv.

usage: python scripts/ensemble.py run1 run2 ... [--method mean|median] [--out submission.csv]
A run contributes only the folds it has finished; the ensemble score uses breaths covered by every run.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('runs', nargs='+')
p.add_argument('--method', default='median')
p.add_argument('--out', default=None)
p.add_argument('--folds', default=None, help='restrict to these folds, e.g. 0,1')
args = p.parse_args()

D = 'work/data'
tr = np.load(f'{D}/train_arr.npy'); y = tr[..., 5]; m = 1 - tr[..., 4]
grid = np.load(f'{D}/pressure_grid.npy')
te_ids = np.load(f'{D}/test_ids.npy')
N = len(y)


def snap(x):
    i = np.clip(np.searchsorted(grid, x), 1, len(grid) - 1)
    lo, hi = grid[i - 1], grid[i]
    return np.where(x - lo < hi - x, lo, hi)


def mae(pred, idx):
    return float((np.abs(pred - y[idx]) * m[idx]).sum() / m[idx].sum())


oofs, tests, covers = [], [], []
for r in args.runs:
    oof = np.full(y.shape, np.nan, np.float32); tp = []
    for f in sorted(glob.glob(f'work/preds/{r}/oof_f*.npy')):
        k = f.split('_f')[-1][:-4]
        if args.folds and k not in args.folds.split(','):
            continue
        idx = np.load(f'work/preds/{r}/oofidx_f{k}.npy')
        oof[idx] = np.load(f)
        tp.append(np.load(f'work/preds/{r}/test_f{k}.npy'))
    cov = ~np.isnan(oof[:, 0])
    info = [json.load(open(f)) for f in sorted(glob.glob(f'work/preds/{r}/info_f*.json'))]
    print(f'{r}: folds={len(tp)} oof={mae(oof[cov], cov):.4f} snapped={mae(snap(oof[cov]), cov):.4f} '
          f'per-fold={[round(i["best"], 4) for i in info]}')
    oofs.append(oof); tests.append(np.mean(tp, 0) if args.method == 'mean' else np.median(tp, 0)); covers.append(cov)

cov = np.all(covers, 0)
agg = np.mean if args.method == 'mean' else np.median
ens = agg(np.stack([o[cov] for o in oofs]), 0)
print(f'ENSEMBLE ({args.method}) on {cov.sum()} breaths: {mae(ens, cov):.4f} snapped {mae(snap(ens), cov):.4f}')
if args.out:
    t = snap(agg(np.stack(tests), 0))
    sub = pd.DataFrame({'id': te_ids.ravel(), 'pressure': t.ravel()}).sort_values('id')
    ss = pd.read_csv('data/sample_submission.csv')
    assert len(sub) == len(ss) and (sub['id'].values == ss['id'].values).all()
    sub.to_csv(args.out, index=False)
    print('wrote', args.out, sub.shape)
