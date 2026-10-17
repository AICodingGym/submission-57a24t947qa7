"""Average test probabilities from run files, map 30 words -> 12 labels, apply rules, write submission.csv.
usage: python make_sub.py out.csv run_file1.npy [run_file2.npy ...]"""
import sys, hashlib, numpy as np, pandas as pd
CMDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
C = 'work/cache'
meta = pd.read_csv(f'{C}/train_meta.csv'); tm = pd.read_csv(f'{C}/test_meta.csv')
WORDS = sorted(meta.word.unique())
lab = lambda w: w if w in CMDS else 'unknown'
P = np.mean([np.load(f) for f in sys.argv[2:]], 0)
L = [CMDS + ['unknown']][0]
P12 = np.zeros((len(P), 11))
for i, w in enumerate(WORDS): P12[:, L.index(lab(w))] += P[:, i]
pred = np.array([L[i] for i in P12.argmax(1)], dtype=object)
# exact-duplicate clips present in train with a consistent label: take the train label
tr = np.load(f'{C}/train_x.npy'); te = np.load(f'{C}/test_x.npy')
d = {}
for i, x in enumerate(tr): d.setdefault(hashlib.md5(x.tobytes()).hexdigest(), set()).add(lab(meta.word[i]))
n_dup = n_chg = 0
for j, x in enumerate(te):
    s = d.get(hashlib.md5(x.tobytes()).hexdigest())
    if s and len(s) == 1:
        n_dup += 1; v = next(iter(s)); n_chg += pred[j] != v; pred[j] = v
# long clips (>1s) are background-noise recordings -> silence
pred[tm.len.values > 16000] = 'silence'
sub = pd.DataFrame({'fname': tm.fname, 'label': pred})
sub.to_csv(sys.argv[1], index=False)
print(f'dups overridden {n_dup} (changed {n_chg}); silence {(pred == "silence").sum()}')
print(sub.label.value_counts().to_dict())
