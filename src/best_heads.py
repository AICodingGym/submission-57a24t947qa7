"""print, for each feature tag, the best-OOF prediction set (excluding stacks) -> used as blend inputs"""
import os, glob, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lab = pd.read_csv(f'{ROOT}/work/cache/train_ids.csv'); cls = list(pd.read_csv(f'{ROOT}/data/sample_submission.csv', nrows=1).columns[1:])
y = lab.breed.map({c: i for i, c in enumerate(cls)}).values
best = {}
for f in glob.glob(f'{ROOT}/work/preds/*__*_oof.npy'):
    n = os.path.basename(f)[:-8]; tag = n.split('__')[0]
    p = np.load(f); ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()
    if tag not in best or ll < best[tag][0]: best[tag] = (ll, n)
for ll, n in sorted(best.values()): print(f'{ll:.5f} {n}')
