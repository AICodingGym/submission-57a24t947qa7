# Stacker: CNN OOF logits + image stats + inc-angle group features (computed out-of-fold) -> LightGBM / logistic
import sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
mt=pd.read_csv('work/meta_tr.csv'); me=pd.read_csv('work/meta_te.csv'); folds=np.load('work/folds.npy')
if __import__('os').environ.get('STSEED'):
    from sklearn.model_selection import StratifiedKFold
    _y=pd.read_csv('work/meta_tr.csv').y.values; folds=np.zeros(len(_y),int)
    for k,(_,v) in enumerate(StratifiedKFold(5,shuffle=True,random_state=int(__import__('os').environ['STSEED'])).split(_y,_y)): folds[v]=k
X=np.load('work/Xtr.npy'); XT=np.load('work/Xte.npy'); y=mt.y.values
names=sys.argv[1].split(',') if len(sys.argv)>1 and sys.argv[1] else []
def imgfeat(X):
    f={}
    for c in range(2):
        b=X[:,c].reshape(len(X),-1); f[f'b{c}_max']=b.max(1); f[f'b{c}_mean']=b.mean(axis=1); f[f'b{c}_std']=b.std(1)
        f[f'b{c}_p90']=np.percentile(b,90,1); f[f'b{c}_p50']=np.median(b,1)
        thr=b.max(1,keepdims=True)-6; f[f'b{c}_area']=(b>thr).sum(1)
        cen=X[:,c,27:48,27:48].reshape(len(X),-1); f[f'b{c}_cmean']=cen.mean(axis=1)
    f['dmax']=f['b0_max']-f['b1_max']; return pd.DataFrame(f)
Ftr=imgfeat(X); Fte=imgfeat(XT)
Ftr['inc']=mt.inc.values; Fte['inc']=me.inc.values
allinc=pd.concat([mt.inc,me.inc])
vc=allinc.value_counts()
Ftr['inc_cnt_all']=mt.inc.map(vc).fillna(0).values; Fte['inc_cnt_all']=me.inc.map(vc).fillna(0).values
Ftr['inc_na']=mt.inc.isna().astype(int); Fte['inc_na']=me.inc.isna().astype(int)
def grp(src_inc,src_y,tgt_inc,self_mask=None):
    d=pd.DataFrame({'inc':src_inc,'y':src_y}).dropna(subset=['inc']); g=d.groupby('inc').y.agg(['sum','count'])
    s=pd.Series(tgt_inc).map(g['sum']).fillna(0).values; c=pd.Series(tgt_inc).map(g['count']).fillna(0).values
    return s,c
for nm in names:
    Ftr['cnn_'+nm]=np.load(f'work/preds/{nm}_oof.npy'); Fte['cnn_'+nm]=np.load(f'work/preds/{nm}_te.npy')
if names: Ftr['cnn_mean']=Ftr[['cnn_'+n for n in names]].mean(axis=1); Fte['cnn_mean']=Fte[['cnn_'+n for n in names]].mean(axis=1)
if names and int(__import__('os').environ.get('SPLITMEAN','0')):
    sn=[n for n in names if n.startswith('simple')]; pn=[n for n in names if not n.startswith('simple')]
    for F in (Ftr,Fte):
        F['cnn_smean']=F[['cnn_'+n for n in sn]].mean(axis=1) if sn else F['cnn_mean']; F['cnn_pmean']=F[['cnn_'+n for n in pn]].mean(axis=1) if pn else F['cnn_mean']
if int(__import__('os').environ.get('MEANONLY','0')):
    Ftr=Ftr.drop(columns=['cnn_'+n for n in names]); Fte=Fte.drop(columns=['cnn_'+n for n in names])
tek=np.zeros((5,len(me))); oof=np.zeros(len(y)); te=np.zeros(len(me)); oofl=np.zeros(len(y)); tel=np.zeros(len(me))
NEAR=int(__import__('os').environ.get('NEAR','0'))
NNG=int(__import__('os').environ.get('NNG','0'))
WINS=[float(w) for w in __import__('os').environ.get('WINS','0.05').split(',')]
def near(src_inc,src_y,tgt_inc,loo):
    si=np.asarray(src_inc); ok=~np.isnan(si); si=si[ok]; sy=np.asarray(src_y)[ok]
    out={}
    for w in WINS: out[f'n_pos_{w}']=np.zeros(len(tgt_inc)); out[f'n_cnt_{w}']=np.zeros(len(tgt_inc))
    out['n_dist']=np.full(len(tgt_inc),np.nan)
    if NNG:
        for side in ('lo','hi'):
            for q in 'drc': out[f'nn_{side}_{q}']=np.full(len(tgt_inc),np.nan)
    for j,t in enumerate(tgt_inc):
        if np.isnan(t): continue
        dd=np.abs(si-t); nz=dd>1e-6
        for w in WINS:
            m=nz&(dd<w); out[f'n_pos_{w}'][j]=sy[m].sum(); out[f'n_cnt_{w}'][j]=m.sum()
        if nz.any(): out['n_dist'][j]=dd[nz].min()
        if NNG:
            # nearest distinct-angle groups on each side (below / above)
            for side,msk in (('lo',si<t-1e-6),('hi',si>t+1e-6)):
                if msk.any():
                    a=si[msk][np.argmin(dd[msk])]; g=np.abs(si-a)<1e-6
                    out[f'nn_{side}_d'][j]=abs(a-t); out[f'nn_{side}_r'][j]=sy[g].mean(); out[f'nn_{side}_c'][j]=g.sum()
    return out
KNN=int(__import__('os').environ.get('KNN','0'))
def knnf(Fsrc,ysrc,Ftgt,loo):
    from sklearn.neighbors import NearestNeighbors
    cols=['inc','b0_p50','b1_p50']; sc=np.array([0.05,1.0,1.0])
    A=(Fsrc[cols].fillna(-100).values)/sc; B=(Ftgt[cols].fillna(-100).values)/sc
    nn_=NearestNeighbors(n_neighbors=KNN+loo).fit(A); d,i=nn_.kneighbors(B)
    if loo: d,i=d[:,1:],i[:,1:]
    yy=np.asarray(ysrc)[i]; w=1/(d+0.1)
    return (yy*w).sum(1)/w.sum(1), d[:,0]
def addgrp(F,s,c,tinc=None,src=None,knn=None):
    F=F.copy()
    if KNN and knn is not None: F['knn_y'],F['knn_d']=knnf(knn[0],knn[1],F,knn[2])
    F['g_cnt']=c; F['g_pos']=s; F['g_neg']=c-s; F['g_rate']=(s+0.5)/(c+1)
    if NEAR and tinc is not None:
        for k_,v_ in near(src[0],src[1],tinc,False).items(): F[k_]=v_
        for w in WINS: F[f'n_rate_{w}']=(F[f'n_pos_{w}']+0.5)/(F[f'n_cnt_{w}']+1)
    return F
import json as _j, os as _o
prm=dict(objective='binary',learning_rate=0.02,num_leaves=8,min_data_in_leaf=15,feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,seed=0)
prm.update(_j.loads(_o.environ.get('PRM','{}')))
LRC=float(_o.environ.get('LRC','1.0'))
NR=int(sys.argv[2]) if len(sys.argv)>2 else 400
lrcols=(__import__('os').environ.get('LRCOLS','cnn_mean').split(',') if names else ['b0_max','b1_max'])
for k in range(5):
    tri=np.where(folds!=k)[0]; vai=np.where(folds==k)[0]
    # val group feats from training fold only
    s,c=grp(mt.inc.values[tri],y[tri],mt.inc.values[vai]); Fv=addgrp(Ftr.iloc[vai],s,c,mt.inc.values[vai],(mt.inc.values[tri],y[tri]),(Ftr.iloc[tri],y[tri],0))
    # train-fold group feats: leave-one-out within training fold (excludes self) -> mimics "other labeled rows with same angle"
    s,c=grp(mt.inc.values[tri],y[tri],mt.inc.values[tri]); notna=~np.isnan(mt.inc.values[tri])
    s=s-y[tri]*notna; c=c-notna; Ft=addgrp(Ftr.iloc[tri],s,c,mt.inc.values[tri],(mt.inc.values[tri],y[tri]),(Ftr.iloc[tri],y[tri],1))
    s,c=grp(mt.inc.values[tri],y[tri],me.inc.values); Fe=addgrp(Fte,s,c,me.inc.values,(mt.inc.values[tri],y[tri]),(Ftr.iloc[tri],y[tri],0))
    for sd in range(3):
        m=lgb.train({**prm,'seed':sd},lgb.Dataset(Ft,y[tri]),NR)
        oof[vai]+=m.predict(Fv)/3; pe=m.predict(Fe); te+=pe/15; tek[k]+=pe/6
    # simple logistic on cnn + group evidence
    def L(F):
        cols=[F[lrcols].values, F.g_pos, F.g_neg, np.log1p(F.g_pos), np.log1p(F.g_neg), F.inc_na]
        if NEAR:
            for w in WINS: cols+= [np.log1p(F[f'n_pos_{w}']), np.log1p(F[f'n_cnt_{w}']-F[f'n_pos_{w}'])]
        if int(__import__('os').environ.get('INTER','0')):
            ev=np.log1p(F.g_cnt.values+(F[f'n_cnt_{WINS[-1]}'].values if NEAR else 0))
            cols+=[F['cnn_mean'].values*ev, ev]
        return np.c_[tuple(cols)]
    lr=LogisticRegression(C=LRC,max_iter=2000).fit(L(Ft),y[tri]); oofl[vai]=lr.predict_proba(L(Fv))[:,1]; tel+=lr.predict_proba(L(Fe))[:,1]/5
    tek[k]+=lr.predict_proba(L(Fe))[:,1]/2
print('lgb OOF',round(log_loss(y,oof),4),' lr OOF',round(log_loss(y,oofl),4),' blend',round(log_loss(y,(oof+oofl)/2),4))
for nm in names: 'cnn_'+nm in Ftr and print(' cnn',nm,round(log_loss(y,1/(1+np.exp(-Ftr['cnn_'+nm]))),4))
if names: print(' cnn_mean',round(log_loss(y,1/(1+np.exp(-Ftr['cnn_mean']))),4))
tag=sys.argv[3] if len(sys.argv)>3 else 'stack'
np.save(f'work/preds/{tag}_tek.npy',tek); np.save(f'work/preds/{tag}_lgb_oof.npy',oof); np.save(f'work/preds/{tag}_lgb_te.npy',te)
np.save(f'work/preds/{tag}_lr_oof.npy',oofl); np.save(f'work/preds/{tag}_lr_te.npy',tel)

# full refit: train on all rows (LOO group feats), test group feats from all train
s,c=grp(mt.inc.values,y,mt.inc.values); notna=~np.isnan(mt.inc.values); Fa=addgrp(Ftr,s-y*notna,c-notna,mt.inc.values,(mt.inc.values,y),(Ftr,y,1))
s,c=grp(mt.inc.values,y,me.inc.values); Fe=addgrp(Fte,s,c,me.inc.values,(mt.inc.values,y),(Ftr,y,0))
tf=np.mean([lgb.train({**prm,'seed':sd},lgb.Dataset(Fa,y),NR).predict(Fe) for sd in range(3)],0)
tfl=LogisticRegression(C=LRC,max_iter=2000).fit(L(Fa),y).predict_proba(L(Fe))[:,1]
np.save(f'work/preds/{tag}_full_te.npy',(tf+tfl)/2); np.save(f'work/preds/{tag}_blend_oof.npy',(oof+oofl)/2)
print('corr full vs foldavg', np.corrcoef((tf+tfl)/2,(te+tel)/2)[0,1])
