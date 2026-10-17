import pandas as pd, numpy as np, time, sys
sys.path.insert(0,'src'); from metric import score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
t=time.time()
tr=pd.read_parquet('work/train.parquet'); te=pd.read_parquet('work/test.parquet')
va=tr[tr.fold==0]; trn=tr[tr.fold!=0].sample(600000,random_state=0)
vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=300000,sublinear_tf=True,dtype=np.float32)
X=vec.fit_transform(trn.comment_text); Xv=vec.transform(va.comment_text); Xt=vec.transform(te.comment_text)
clf=LogisticRegression(C=4,max_iter=300,solver='lbfgs')
clf.fit(X,trn.target.values>=0.5)
pv=clf.predict_proba(Xv)[:,1]; pt=clf.predict_proba(Xt)[:,1]
print(score(va,pv,True), time.time()-t)
np.save('work/preds/tfidf_val.npy',pv); np.save('work/preds/tfidf_test.npy',pt)
pd.DataFrame({'id':te.id,'prediction':pt}).to_csv('submission.csv',index=False)
