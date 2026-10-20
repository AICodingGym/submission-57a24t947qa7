# Combine runs: for each "model family" (name prefix before _f<k>), assemble OOF over available folds and
# average test preds over folds. Then ensemble families (mean of probabilities) on the folds they share.
import sys, os, glob, json, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(f"{ROOT}/data/train.csv"); folds = pd.read_csv(f"{ROOT}/work/folds.csv").fold.values
sub = pd.read_csv(f"{ROOT}/data/sample_submission.csv")
COLS = list(tr.columns[1:12]); EVAL = list(sub.columns[1:]); EI = [COLS.index(c) for c in EVAL]
Y = tr[COLS].values
fams = sys.argv[1].split(",")          # family prefixes, e.g. v2s_640seg,cnxs_512
weights = [float(w) for w in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] else [1.0] * len(fams)
write = "--write" in sys.argv

def auc(y, p): return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in EI]))

oof, test, cover = {}, {}, {}
for fam in fams:
    o = np.full((len(tr), 11), np.nan, np.float32); ts = []; fs = []
    for k in range(5):
        for pre in fam.split("|"):  # "a|b": fold k taken from first prefix that has it
            f = f"{ROOT}/work/preds/{pre}_f{k}_val.npy"
            if os.path.exists(f) and os.path.exists(f"{ROOT}/work/preds/{pre}_f{k}_meta.json"):
                o[folds == k] = np.load(f); ts.append(np.load(f"{ROOT}/work/preds/{pre}_f{k}_test.npy")); fs.append(k); break
    if not fs:
        print(f"{fam}: no finished folds, skipped"); continue
    full = sorted(glob.glob(f"{ROOT}/work/preds/{fam}_full*_test.npy"))
    ts += [np.load(f) for f in full]
    m = ~np.isnan(o[:, 0])
    print(f"{fam}: folds {fs} full {len(full)} oof AUC9 {auc(Y[m], o[m]):.5f}" if m.any() else f"{fam}: no val")
    for k in fs: print(f"   f{k}: {auc(Y[folds==k], o[folds==k]):.5f}")
    oof[fam], test[fam], cover[fam] = o, np.mean(ts, 0), m
w0 = dict(zip(fams, weights)); fams = [f for f in fams if f in oof]; weights = [w0[f] for f in fams]
m = np.all([cover[f] for f in fams], 0)
ens_o = sum(w * oof[f] for f, w in zip(fams, weights)) / sum(weights)
ens_t = sum(w * test[f] for f, w in zip(fams, weights)) / sum(weights)
score = auc(Y[m], ens_o[m])
print(f"ENSEMBLE {fams} w={weights} on {m.sum()} shared rows: AUC9 {score:.5f}")
print("per-col:", {c: round(roc_auc_score(Y[m, j], ens_o[m, j]), 4) for c, j in zip(EVAL, EI)})
if write:
    out = sub.copy(); out[EVAL] = ens_t[:, EI]
    out.to_csv(f"{ROOT}/submission.csv", index=False); print("wrote submission.csv", out.shape)
