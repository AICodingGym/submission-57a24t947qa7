"""Blend run predictions (work/runs/<name>/pred.npy), score on the shared validation molecules,
and write submission.csv. Usage: python scripts/make_sub.py run1 run2 ... [--per_type]
--per_type: choose per-type non-negative blend weights on validation (coarse grid), else plain mean."""
import sys, math, json
import numpy as np
import pandas as pd
import torch

runs = [a for a in sys.argv[1:] if not a.startswith('--')]
per_type = '--per_type' in sys.argv
D = torch.load('work/prep.pt')
TYPES = D['types']
pm, pt, Y, is_test = D['pm'].numpy(), D['ptype'].numpy(), D['y'][:, 0].numpy(), D['is_test'].numpy().astype(bool)
preds = np.stack([np.load(f'work/runs/{r}/pred.npy') for r in runs])  # (K, R)
val = ~is_test & ~np.isnan(preds[0])
for k in range(1, len(runs)):
    val &= ~np.isnan(preds[k])


def score(p, mask):
    return np.mean([math.log((np.abs(p[mask & (pt == t)] - Y[mask & (pt == t)])).mean()) for t in range(len(TYPES))])


for r, p in zip(runs, preds):
    print(f'{r:30s} {score(p, val):.4f}')
blend = preds.mean(0)
print('mean blend', round(score(blend, val), 4))
if per_type and len(runs) > 1:
    # per type: greedy forward selection of runs with replacement (Caruana) on validation
    blend = np.zeros(preds.shape[1])
    for t in range(len(TYPES)):
        m = val & (pt == t)
        cnt = np.zeros(len(runs))
        cur = None
        for it in range(20):
            best = None
            for k in range(len(runs)):
                c = cnt.copy(); c[k] += 1
                b = (c[:, None] * preds[:, m]).sum(0) / c.sum()
                e = np.abs(b - Y[m]).mean()
                if best is None or e < best[0]:
                    best = (e, k)
            cnt[best[1]] += 1
        w = cnt / cnt.sum()
        tm = pt == t
        blend[tm] = (w[:, None] * preds[:, tm]).sum(0)
        print(TYPES[t], np.round(w, 2))
    print('per-type blend (optimistic, weights fit on val)', round(score(blend, val), 4))
sub = pd.read_csv('data/sample_submission.csv')
idx = pd.Series(np.arange(len(pm)), index=D['pid'].numpy())
vals = blend[idx.loc[sub.id.values].values]
assert is_test[idx.loc[sub.id.values].values].all() and not np.isnan(vals).any()
sub['scalar_coupling_constant'] = vals
sub.to_csv('submission.csv', index=False)
print('wrote submission.csv', sub.shape, 'val', round(score(blend, val), 4))
