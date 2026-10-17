import pandas as pd, numpy as np, sys, json
from scipy.stats import rankdata
from scipy.optimize import minimize
sys.path.insert(0,'src'); from metric import score, IDS
names=sys.argv[1].split(','); out=sys.argv[2] if len(sys.argv)>2 else None
tr=pd.read_parquet('work/train.parquet',columns=['target','fold']+IDS); va=tr[tr.fold==0]
te=pd.read_parquet('work/test.parquet',columns=['id'])
R=lambda v:rankdata(v)/len(v)
V=np.stack([R(np.load(f'work/preds/{n}_val.npy')) for n in names],1)
T=np.stack([R(np.load(f'work/preds/{n}_test.npy')) for n in names],1)
for i,n in enumerate(names): print(n, round(score(va,V[:,i])['final'],5))
eq=np.ones(len(names))/len(names); print('equal', json.dumps({k:round(v,5) for k,v in score(va,V@eq).items()}))
f=lambda w: -score(va,V@(np.abs(w)/np.abs(w).sum()))['final']
r=minimize(f,eq,method='Nelder-Mead',options={'maxiter':200,'xatol':1e-3,'fatol':1e-6})
w=np.abs(r.x)/np.abs(r.x).sum(); print('opt w',dict(zip(names,w.round(3))), -r.fun)
# shrink optimized weights halfway toward equal to limit holdout overfitting
wf=0.8*w+0.2*eq; s=score(va,V@wf,True); print('final w',dict(zip(names,wf.round(3))), json.dumps(s))
if out:
    pd.DataFrame({'id':te.id,'prediction':T@wf}).to_csv(out,index=False); print('wrote',out)
# honest check: tune weights on one half of holdout, score on the other half
h=np.random.RandomState(0).rand(len(va))<0.5; res=[]
for A,B in [(h,~h),(~h,h)]:
    vA,vB=va[A],va[B]
    g=lambda w: -score(vA,V[A]@(np.abs(w)/np.abs(w).sum()))['final']
    r2=minimize(g,eq,method='Nelder-Mead',options={'maxiter':200,'xatol':1e-3,'fatol':1e-6})
    w2=np.abs(r2.x)/np.abs(r2.x).sum(); w2=0.8*w2+0.2*eq
    res.append((score(vB,V[B]@w2)['final'], score(vB,V[B]@eq)['final']))
print('half-split CV: tuned %.5f equal %.5f'%tuple(np.mean(res,0)))
