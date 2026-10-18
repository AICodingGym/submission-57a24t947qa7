# Evaluate OOF predictions of one or more run groups and tune thresholds / min-area post-processing.
import argparse, os, json, time, numpy as np, pandas as pd
import metric
from metric import dice2d


def score(pr, gt, days):
    return metric.score(pr, gt, days, ds=1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = f'{ROOT}/work/cache'
p = argparse.ArgumentParser()
p.add_argument('--groups', required=True)   # e.g. "effb3_f{},v2s_f{}" ; folds averaged across groups
p.add_argument('--folds', default='0,1,2,3')
p.add_argument('--out', default='')
a = p.parse_args()
groups = a.groups.split(','); folds = [int(f) for f in a.folds.split(',')]

meta = pd.read_csv(f'{C}/train_meta.csv')
Y = np.load(f'{C}/train_mask.npy', mmap_mode='r')
idx, P = [], []
for f in folds:
    # work at 192x192 (stride 2) so tuning stays cheap on the 4-CPU box
    ps = [np.load(f'{ROOT}/work/runs/{g.format(f)}/oof.npy', mmap_mode='r')[:, :, ::2, ::2].astype(np.uint16) for g in groups]
    idx.append(np.load(f'{ROOT}/work/runs/{groups[0].format(f)}/oof_idx.npy'))
    P.append((sum(ps) // len(ps)).astype(np.uint8))
idx = np.concatenate(idx); o = np.argsort(idx); idx = idx[o]; P = np.concatenate(P)[o]
gt = (Y[idx][:, None, ::2, ::2] & np.array([1, 2, 4], np.uint8).reshape(1, 3, 1, 1)) > 0
days = meta.day.values[idx]
print('n', len(idx), flush=True)


from post import apply as _apply


def apply(P, thr, mina, cc=(0, 0, 0)):
    return _apply(P, days, thr, mina, cc)


t = time.time()
base = score(apply(P, [0.5] * 3, [0] * 3), gt, days)
print('base', base, f'{time.time()-t:.0f}s', flush=True)
# per-class threshold via 2D dice (cheap), then min-area grid on full score
best_thr = []
for c in range(3):
    res = {th: dice2d(P[:, c:c + 1] > th * 255, gt[:, c:c + 1]).mean() for th in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]}
    best_thr.append(max(res, key=res.get)); print(c, {k: round(v, 4) for k, v in res.items()}, flush=True)
best = dict(thr=best_thr, min_area=[0, 0, 0], **score(apply(P, best_thr, [0] * 3), gt, days))
print('thr', best, flush=True)
for ma in [12, 25, 50, 100]:  # in 192x192 pixels (x4 at 384)
    r = score(apply(P, best_thr, [ma] * 3), gt, days)
    print('min_area', ma, r, flush=True)
    if r['score'] > best['score']:
        best = dict(thr=best_thr, min_area=[ma] * 3, **r)
for cc in [0.01, 0.03, 0.1, 0.2]:
    r = score(apply(P, best_thr, best['min_area'], [cc] * 3), gt, days)
    print('cc_frac', cc, r, flush=True)
    if r['score'] > best['score']:
        best = dict(thr=best_thr, min_area=best['min_area'], cc_frac=[cc] * 3, **r)
best.setdefault('cc_frac', [0, 0, 0])
print('BEST', best)
if a.out:
    best['min_area'] = [m * 4 for m in best['min_area']]  # infer applies it at 384x384
    json.dump(best, open(a.out, 'w'))
