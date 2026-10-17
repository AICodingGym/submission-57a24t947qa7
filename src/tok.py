# pre-tokenize train+test for a tokenizer family; stores ragged ids (head truncation to MAXLEN)
import pandas as pd, numpy as np, sys, time, glob
from transformers import AutoTokenizer
name, tag = sys.argv[1], sys.argv[2]; MAXLEN=int(sys.argv[3]) if len(sys.argv)>3 else 256
path=glob.glob(f'work/hf/models--{name.replace("/","--")}/snapshots/*')[0]
tok=AutoTokenizer.from_pretrained(path)
t=time.time()
for split in ['train','test']:
    df=pd.read_parquet(f'work/{split}.parquet',columns=['comment_text'])
    txt=df.comment_text.fillna('').astype(str).tolist()
    ids=[]
    for i in range(0,len(txt),50000):
        enc=tok(txt[i:i+50000],truncation=True,max_length=MAXLEN,add_special_tokens=True)['input_ids']
        ids.extend(enc)
    lens=np.array([len(x) for x in ids],dtype=np.int32)
    flat=np.concatenate([np.array(x,dtype=np.int32) for x in ids])
    np.save(f'work/tok_{tag}_{split}_flat.npy',flat); np.save(f'work/tok_{tag}_{split}_lens.npy',lens)
    print(split,len(lens),lens.mean(),np.percentile(lens,[50,90,99]),time.time()-t,flush=True)
