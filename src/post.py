"""Post-processing of a blend: p' = (1-a-e) p + a * p@C + e/120, C = row-normalised OOF confusion (soft) matrix.
Nested 5-fold CV over the blend OOF to check it honestly."""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lab = pd.read_csv(f'{ROOT}/work/cache/train_ids.csv'); cls = list(pd.read_csv(f'{ROOT}/data/sample_submission.csv', nrows=1).columns[1:])
y = lab.breed.map({c: i for i, c in enumerate(cls)}).values
folds = np.load(f'{ROOT}/work/folds.npy')
oof = np.load(sys.argv[1]); te = np.load(sys.argv[2]) if len(sys.argv) > 2 else None
Y = np.eye(120)[y]


def conf(p, yy):
    Cm = np.eye(120)[yy].T @ p + 1e-3  # true -> predicted soft counts
    # we need P(true=j | predicted dist) -> use transpose of predicted->true
    Cm = (p.T @ np.eye(120)[yy]) + 1e-3  # pred i -> true j
    return Cm / Cm.sum(1, keepdims=True)


def apply(p, C, a, e):
    return (1 - a - e) * p + a * p @ C + e / 120


def ll(p, yy):
    return -np.log(np.clip(p[np.arange(len(yy)), yy], 1e-15, 1)).mean()


def fit(p, yy):
    C = conf(p, yy)
    r = minimize(lambda v: ll(apply(p, C, abs(v[0]), abs(v[1])), yy), [0.05, 0.001], method='Nelder-Mead')
    return C, abs(r.x[0]), abs(r.x[1])


print('base', ll(oof, y))
tot = 0
for k in range(5):
    tr, va = folds != k, folds == k
    C, a, e = fit(oof[tr], y[tr])
    tot += ll(apply(oof[va], C, a, e), y[va]) * va.sum()
    print(k, round(a, 4), round(e, 5))
print('nested cv', tot / len(y))
