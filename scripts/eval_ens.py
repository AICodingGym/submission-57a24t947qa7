"""Fold-0 validation accuracy (12-label) of single runs and their probability average.
usage: python eval_ens.py run1 run2 ..."""
import sys, numpy as np, pandas as pd
CMDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
meta = pd.read_csv('work/cache/train_meta.csv')
WORDS = sorted(meta.word.unique())
w2l = np.array([CMDS.index(w) if w in CMDS else 10 for w in WORDS])
y = meta.word.map({w: i for i, w in enumerate(WORDS)}).values
def acc12(p):
    p12 = np.zeros((len(p), 11))
    for i in range(len(WORDS)): p12[:, w2l[i]] += p[:, i]
    return (p12.argmax(1) == w2l[y[m]]).mean()
oofs = [np.load(f'work/runs/{r}/oof.npy') for r in sys.argv[1:]]
m = ~np.isnan(oofs[0][:, 0])
for r, o in zip(sys.argv[1:], oofs): print(r, round(acc12(o[m]), 5))
print('ensemble', round(acc12(np.mean([o[m] for o in oofs], 0)), 5))
