import pandas as pd, numpy as np
R='work/raw/'
tr=pd.read_json(R+'train.json'); te=pd.read_json(R+'test.json')
def arr(df):
    b1=np.stack(df.band_1.map(np.array)).reshape(-1,75,75); b2=np.stack(df.band_2.map(np.array)).reshape(-1,75,75)
    return np.stack([b1,b2],1).astype(np.float32)
np.save('work/Xtr.npy',arr(tr)); np.save('work/Xte.npy',arr(te))
for d,n in [(tr,'tr'),(te,'te')]:
    m=d[['id']].copy(); m['inc']=pd.to_numeric(d.inc_angle,errors='coerce')
    if 'is_iceberg' in d: m['y']=d.is_iceberg
    m.to_csv(f'work/meta_{n}.csv',index=False)
