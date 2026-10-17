import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
IDS=['male','female','homosexual_gay_or_lesbian','christian','jewish','muslim','black','white','psychiatric_or_mental_illness']
def score(df, pred, verbose=False):
    y=(df.target.values>=0.5); p=np.asarray(pred)
    out={'overall':roc_auc_score(y,p)}
    sub,bpsn,bnsp=[],[],[]
    for c in IDS:
        g=(df[c].fillna(0).values>=0.5)
        sub.append(roc_auc_score(y[g],p[g]))
        m=(g&~y)|(~g&y); bpsn.append(roc_auc_score(y[m],p[m]))
        m=(g&y)|(~g&~y); bnsp.append(roc_auc_score(y[m],p[m]))
    pm=lambda a:np.power(np.mean(np.power(np.array(a),-5)),-1/5)
    out.update(sub=pm(sub),bpsn=pm(bpsn),bnsp=pm(bnsp))
    out['final']=0.25*(out['overall']+out['sub']+out['bpsn']+out['bnsp'])
    if verbose:
        print(pd.DataFrame({'sub':sub,'bpsn':bpsn,'bnsp':bnsp},index=IDS).round(4))
    return out
