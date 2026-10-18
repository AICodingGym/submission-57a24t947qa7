"""Evaluate temporal smoothing of OOF probs within pair. Usage: python scripts/postproc.py tag1[,tag2...]"""
import sys, numpy as np, pandas as pd
tags = sys.argv[1].split(',')
N_ALL, P_ALL = 4258375, 57891
o = pd.read_parquet(f'work/oof_{tags[0]}.parquet')
if len(tags) > 1:
    o['oof'] = np.mean([pd.read_parquet(f'work/oof_{t}.parquet').oof.values for t in tags], axis=0)
parts = o.contact_id.str.rsplit('_', n=3, expand=True)
o['key'] = parts[0] + '_' + parts[2] + '_' + parts[3]
o['step'] = parts[1].astype(int)
o = o.sort_values(['key', 'step']).reset_index(drop=True)

def mcc(pred, y):
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = P_ALL - tp; tn = N_ALL - tp - fp - fn
    return (tp * tn - fp * fn) / np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

y = o.contact.values
for w in [1, 3, 5]:
    p = o.oof if w == 1 else o.groupby('key').oof.transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
    best = max((mcc((p.values > t).astype(int), y), t) for t in np.arange(0.2, 0.6, 0.01))
    print('win', w, best)
