import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
m=pd.read_csv('work/meta_tr.csv'); f=np.zeros(len(m),int)
for k,(_,v) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(m,m.y)): f[v]=k
np.save('work/folds.npy',f)
