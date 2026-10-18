"""Cross-pair context from CNN probs: per (play, step, player) aggregates of other contacts.
Usage: python scripts/ctx_feats.py out_name cnnfeat_tag1,cnnfeat_tag2"""
import sys, numpy as np, pandas as pd
out, tags = sys.argv[1], sys.argv[2].split(',')
for split in ['train', 'test']:
    fs = [pd.read_parquet(f'work/cnnfeat_{t}_{split}.parquet', columns=['contact_id', 'cnn_mean', 'cnn_max']) for t in tags]
    w = fs[0][['contact_id']].copy()
    w['c'] = np.nanmean([f.set_index('contact_id').loc[w.contact_id, 'cnn_mean'].values for f in fs], axis=0)
    parts = w.contact_id.str.split('_', expand=True)
    w['gp'] = parts[0] + '_' + parts[1]; w['step'] = parts[2].astype(int)
    w['a'] = parts[3]; w['b'] = parts[4]
    pp = w[w.b != 'G']
    long = pd.concat([pp[['gp', 'step', 'a', 'c']].rename(columns={'a': 'pl'}), pp[['gp', 'step', 'b', 'c']].rename(columns={'b': 'pl'})])
    agg = long.groupby(['gp', 'step', 'pl']).c.agg(['max', 'sum', 'count']).reset_index()
    gg = w[w.b == 'G'][['gp', 'step', 'a', 'c']].rename(columns={'a': 'pl', 'c': 'gc'})
    agg = agg.merge(gg, on=['gp', 'step', 'pl'], how='outer')
    # second max (excluding the row itself is approximated via sum - c)
    r = w[['contact_id', 'gp', 'step', 'a', 'b', 'c']]
    for side in ['a', 'b']:
        r = r.merge(agg.rename(columns={'pl': side, 'max': f'ctx_max_{side}', 'sum': f'ctx_sum_{side}', 'count': f'ctx_n_{side}', 'gc': f'ctx_g_{side}'}), on=['gp', 'step', side], how='left')
    r['ctx_other_a'] = r.ctx_sum_a - np.where(r.b == 'G', 0, r.c)
    r['ctx_other_b'] = r.ctx_sum_b - r.c
    r = r.drop(columns=['gp', 'step', 'a', 'b', 'c'])
    for c in r.columns[1:]:
        r[c] = r[c].astype(np.float32)
    r.to_parquet(f'work/{out}_{split}.parquet')
    print(split, r.shape)
