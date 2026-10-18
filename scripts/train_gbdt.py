"""GroupKFold XGBoost (GPU) on tabular features. Writes OOF + test preds.
Usage: python scripts/train_gbdt.py <tag> [extra_feature_parquet_prefix]
"""
import sys, time, json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import matthews_corrcoef

tag = sys.argv[1]
extra = sys.argv[2] if len(sys.argv) > 2 else None
DIST_MAX = 3.0
t0 = time.time()

tr = pd.read_parquet('work/feats/train.parquet')
te = pd.read_parquet('work/feats/test.parquet')
if extra:
    for ex in extra.split(','):
        sfx = ex.split('_')[-1]
        e1 = pd.read_parquet(f'{ex}_train.parquet'); e2 = pd.read_parquet(f'{ex}_test.parquet')
        ren = {c: f'{c}_{sfx}' for c in e1.columns if c != 'contact_id'}
        tr = tr.merge(e1.rename(columns=ren), on='contact_id', how='left')
        te = te.merge(e2.rename(columns=ren), on='contact_id', how='left')

def keep(d):
    return (d.is_g == 1) | (d.dist <= DIST_MAX) | d.dist.isna()

drop = ['contact_id', 'game_play', 'contact', 'p1', 'p2']
feats = [c for c in tr.columns if c not in drop]
print('n feats', len(feats))

trk = tr[keep(tr)].reset_index(drop=True)
print('train rows kept', len(trk), 'of', len(tr), 'pos kept', trk.contact.sum(), 'of', tr.contact.sum())
tek = te[keep(te)].reset_index(drop=True)

params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device='cuda',
              learning_rate=float(__import__('os').environ.get('LR', '0.05')), max_depth=int(__import__('os').environ.get('DEPTH', '8')), subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
              reg_lambda=1.0)
gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(trk), dtype=np.float32)
tpred = np.zeros(len(tek), dtype=np.float32)
dte = xgb.DMatrix(tek[feats])
for f, (a, b) in enumerate(gkf.split(trk, trk.contact, trk.game_play)):
    if __import__('os').environ.get('ALGO') == 'lgb':
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, num_leaves=127, min_child_samples=50, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, reg_lambda=1.0, n_jobs=16, verbose=-1)
        m.fit(trk.loc[a, feats], trk.contact.values[a], eval_set=[(trk.loc[b, feats], trk.contact.values[b])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[b] = m.predict_proba(trk.loc[b, feats])[:, 1]
        tpred += m.predict_proba(tek[feats])[:, 1] / 5
        print(f'fold {f} best_it {m.best_iteration_} mcc@0.3 {matthews_corrcoef(trk.contact.values[b], oof[b] > 0.3):.4f}', flush=True)
        continue
    da = xgb.DMatrix(trk.loc[a, feats], label=trk.contact.values[a])
    db = xgb.DMatrix(trk.loc[b, feats], label=trk.contact.values[b])
    m = xgb.train(params, da, num_boost_round=5000, evals=[(db, 'val')], early_stopping_rounds=200, verbose_eval=False)
    oof[b] = m.predict(db, iteration_range=(0, m.best_iteration + 1))
    tpred += m.predict(dte, iteration_range=(0, m.best_iteration + 1)) / 5
    print(f'fold {f} best_it {m.best_iteration} mcc@0.3 {matthews_corrcoef(trk.contact.values[b], oof[b] > 0.3):.4f}', flush=True)
    if f == 0:
        imp = pd.Series(m.get_score(importance_type='gain')).sort_values(ascending=False)
        print(imp.head(30))

# threshold search over all train rows (dropped rows count as negatives)
y_all = tr.contact.values
n_missing_pos = y_all.sum() - trk.contact.sum()
def full_mcc(thr_p, thr_g):
    pred = np.where(trk.is_g == 1, oof > thr_g, oof > thr_p)
    y = trk.contact.values
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum() + n_missing_pos
    tn = len(y_all) - tp - fp - fn
    num = tp * tn - fp * fn
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den if den > 0 else 0
ths = np.arange(0.1, 0.8, 0.01)
best = max((full_mcc(t, t), t) for t in ths)
print('single thr', best)
bp = max((full_mcc(tp_, tg), tp_, tg) for tp_ in ths[::2] for tg in ths[::2])
print('separate thr', bp)
pd.DataFrame({'contact_id': trk.contact_id, 'oof': oof, 'contact': trk.contact, 'is_g': trk.is_g, 'game_play': trk.game_play}).to_parquet(f'work/oof_{tag}.parquet')
pd.DataFrame({'contact_id': tek.contact_id, 'pred': tpred, 'is_g': tek.is_g}).to_parquet(f'work/test_{tag}.parquet')
json.dump({'tag': tag, 'mcc_single': best[0], 'thr': best[1], 'mcc_sep': bp[0], 'thr_p': bp[1], 'thr_g': bp[2],
           'minutes': (time.time() - t0) / 60}, open(f'work/res_{tag}.json', 'w'))
print('done in', (time.time() - t0) / 60, 'min')
