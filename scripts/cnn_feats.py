"""Turn per-(contact_id, view) CNN probs into per-contact_id features incl. temporal context.
Usage: python scripts/cnn_feats.py <tag>   (reads work/<tag>_f*_oof.parquet and *_test.parquet)
Writes work/cnnfeat_<tag>_train.parquet, work/cnnfeat_<tag>_test.parquet
"""
import sys, glob
import numpy as np
import pandas as pd

tag = sys.argv[1]


def feats(p):
    w = p.pivot_table(index='contact_id', columns='view', values='p', aggfunc='mean')
    w = w.rename(columns={'Endzone': 'cnn_E', 'Sideline': 'cnn_S'}).reset_index()
    for c in ['cnn_E', 'cnn_S']:
        if c not in w:
            w[c] = np.nan
    w['cnn_max'] = w[['cnn_E', 'cnn_S']].max(axis=1)
    w['cnn_mean'] = w[['cnn_E', 'cnn_S']].mean(axis=1)
    parts = w.contact_id.str.rsplit('_', n=3, expand=True)
    w['key'] = parts[0] + '_' + parts[2] + '_' + parts[3]
    w['step'] = parts[1].astype(int)
    w = w.sort_values(['key', 'step']).reset_index(drop=True)
    g = w.groupby('key')
    for k in [1, 2, 4]:
        w[f'cnn_mean_lag{k}'] = g.cnn_mean.shift(k)
        w[f'cnn_mean_lead{k}'] = g.cnn_mean.shift(-k)
    for win in [3, 5, 9]:
        w[f'cnn_mean_rm{win}'] = g.cnn_mean.transform(lambda s: s.rolling(win, center=True, min_periods=1).mean())
        w[f'cnn_max_rmax{win}'] = g.cnn_max.transform(lambda s: s.rolling(win, center=True, min_periods=1).max())
    w = w.drop(columns=['key', 'step'])
    for c in w.columns:
        if c != 'contact_id':
            w[c] = w[c].astype(np.float32)
    return w


tags = tag.split('+')
def load(kind):
    ds = []
    for t in tags:
        fs = sorted(glob.glob(f'work/{t}_f*_{kind}.parquet'))
        if kind == 'oof':
            d = pd.concat([pd.read_parquet(f) for f in fs]).sort_values('row').reset_index(drop=True)
        else:
            xs = [pd.read_parquet(f) for f in fs]
            d = xs[0].copy(); d['p'] = np.mean([x.p.values for x in xs], axis=0)
        ds.append(d)
    out = ds[0].copy()
    out['p'] = np.mean([d.p.values for d in ds], axis=0)
    return out
name = tag.replace('+', '')
feats(load('oof')).to_parquet(f'work/cnnfeat_{name}_train.parquet')
feats(load('test')).to_parquet(f'work/cnnfeat_{name}_test.parquet')
print('ok', name)
