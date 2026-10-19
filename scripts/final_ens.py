"""Pooled ensemble: every (run, fold) model is one member.
OOF: per breath, median over all runs that have an OOF prediction for it.
Test: median over all fold-model test predictions. Snap to pressure grid.
usage: python scripts/final_ens.py run1 run2 ... [--out submission.csv]
"""
import argparse, glob
import numpy as np, pandas as pd

p = argparse.ArgumentParser(); p.add_argument('runs', nargs='+'); p.add_argument('--out'); p.add_argument('--method', default='median')
args = p.parse_args()
D = 'work/data'
tr = np.load(f'{D}/train_arr.npy'); y = tr[..., 5]; m = 1 - tr[..., 4]
grid = np.load(f'{D}/pressure_grid.npy'); te_ids = np.load(f'{D}/test_ids.npy')
agg = np.nanmedian if args.method == 'median' else np.nanmean

def snap(x):
    i = np.clip(np.searchsorted(grid, x), 1, len(grid) - 1)
    return np.where(x - grid[i - 1] < grid[i] - x, grid[i - 1], grid[i])

oof = np.full((len(args.runs),) + y.shape, np.nan, np.float32); tests = []
for j, r in enumerate(args.runs):
    for f in sorted(glob.glob(f'work/preds/{r}/oof_f*.npy')):
        k = f.split('_f')[-1][:-4]
        oof[j, np.load(f'work/preds/{r}/oofidx_f{k}.npy')] = np.load(f)
        tests.append(np.load(f'work/preds/{r}/test_f{k}.npy'))
cnt = (~np.isnan(oof[:, :, 0])).sum(0)
e = snap(agg(oof, 0))
cov = cnt > 0
print(f'members={len(tests)} breaths covered={cov.sum()} mean runs/breath={cnt[cov].mean():.2f}')
print(f'OOF {args.method} snapped MAE: {(np.abs(e[cov] - y[cov]) * m[cov]).sum() / m[cov].sum():.4f}')
if args.out:
    t = snap(agg(np.stack(tests), 0))
    sub = pd.DataFrame({'id': te_ids.ravel(), 'pressure': t.ravel()}).sort_values('id')
    ss = pd.read_csv('data/sample_submission.csv')
    assert len(sub) == len(ss) and (sub['id'].values == ss['id'].values).all() and np.isfinite(sub.pressure).all()
    sub.to_csv(args.out, index=False); print('wrote', args.out)
