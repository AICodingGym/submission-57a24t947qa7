"""Average OOF/test probs of one or more GBDT tags, smooth within pair (window W), tune threshold on OOF, write submission.csv.
Usage: python scripts/make_sub.py tag1[,tag2] [W]"""
import sys, json, numpy as np, pandas as pd
tags = sys.argv[1].split(',')
W = int(sys.argv[2]) if len(sys.argv) > 2 else 3
N_ALL, P_ALL = 4258375, 57891

def prep(kind, col):
    d = pd.read_parquet(f'work/{kind}_{tags[0]}.parquet')
    d['p'] = np.mean([pd.read_parquet(f'work/{kind}_{t}.parquet')[col].values for t in tags], axis=0)
    parts = d.contact_id.str.rsplit('_', n=3, expand=True)
    d['key'] = parts[0] + '_' + parts[2] + '_' + parts[3]
    d['step'] = parts[1].astype(int)
    d = d.sort_values(['key', 'step']).reset_index(drop=True)
    d['ps'] = d.groupby('key').p.transform(lambda s: s.rolling(W, center=True, min_periods=1).mean()) if W > 1 else d.p
    return d

def mcc(pred, y):
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = P_ALL - tp; tn = N_ALL - tp - fp - fn
    return (tp * tn - fp * fn) / np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

o = prep('oof', 'oof')
score, thr = max((mcc((o.ps.values > t).astype(int), o.contact.values), t) for t in np.arange(0.2, 0.6, 0.005))
t = prep('test', 'pred')
t['contact'] = (t.ps > thr).astype(int)
sub = pd.read_csv('data/sample_submission.csv')[['contact_id']]
sub = sub.merge(t[['contact_id', 'contact']], on='contact_id', how='left').fillna({'contact': 0})
sub['contact'] = sub.contact.astype(int)
assert len(sub) == 463243 and sub.contact_id.is_unique
sub.to_csv('submission.csv', index=False)
json.dump({'tags': tags, 'W': W, 'thr': float(thr), 'oof_mcc': float(score)}, open('work/sub_info.json', 'w'))
print(tags, 'W', W, 'thr', round(thr, 3), 'oof mcc', round(score, 4), 'test pos rate', sub.contact.mean())
