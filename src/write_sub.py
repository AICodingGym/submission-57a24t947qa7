import sys, numpy as np, pandas as pd
me=pd.read_csv('work/meta_te.csv'); ss=pd.read_csv('work/raw/sample_submission.csv')
p=np.load(sys.argv[1]); clip=float(sys.argv[2]) if len(sys.argv)>2 else 0.001
s=pd.DataFrame({'id':me.id,'is_iceberg':np.clip(p,clip,1-clip)}).set_index('id').loc[ss.id].reset_index()
assert len(s)==len(ss) and s.is_iceberg.notna().all(); s.to_csv('submission.csv',index=False); print(s.describe())
