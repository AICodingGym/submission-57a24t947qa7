import pandas as pd, numpy as np
IDS=['male','female','homosexual_gay_or_lesbian','christian','jewish','muslim','black','white','psychiatric_or_mental_illness']
AUX=['severe_toxicity','obscene','identity_attack','insult','threat','sexual_explicit']
tr=pd.read_csv('data/train.csv')
te=pd.read_csv('data/test.csv')
print(tr.shape, te.shape, te.columns.tolist())
keep=['id','target','comment_text']+AUX+IDS+['identity_annotator_count','toxicity_annotator_count']
tr=tr[keep]
rng=np.random.RandomState(42)
tr['fold']=rng.randint(0,20,len(tr))  # fold 0 (~5%) = holdout
tr.to_parquet('work/train.parquet'); te.to_parquet('work/test.parquet')
print((tr.fold==0).sum(), tr.target.ge(.5).mean(), tr[IDS].notna().all(1).mean())
