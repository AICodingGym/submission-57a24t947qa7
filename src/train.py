import pandas as pd, numpy as np, torch, time, sys, glob, json, argparse, math, os
sys.path.insert(0,'src'); from metric import score, IDS
from transformers import AutoModel, get_linear_schedule_with_warmup
p=argparse.ArgumentParser()
p.add_argument('--model'); p.add_argument('--tok'); p.add_argument('--name')
p.add_argument('--lr',type=float,default=2e-5); p.add_argument('--bs',type=int,default=64)
p.add_argument('--nsamp',type=int,default=0)   # number of training samples to use (0=all)
p.add_argument('--maxlen',type=int,default=256); p.add_argument('--seed',type=int,default=0)
p.add_argument('--evals',type=int,default=4); p.add_argument('--deadline',type=str,default='')
a=p.parse_args()
torch.manual_seed(a.seed); rng=np.random.RandomState(a.seed)
AUX=['severe_toxicity','obscene','identity_attack','insult','threat','sexual_explicit']
dev='cuda'
tr=pd.read_parquet('work/train.parquet',columns=['target','fold']+AUX+IDS)
def load(split):
    flat=np.load(f'work/tok_{a.tok}_{split}_flat.npy'); lens=np.load(f'work/tok_{a.tok}_{split}_lens.npy')
    off=np.concatenate([[0],np.cumsum(lens)]); return flat,lens,off
TF,TL,TO=load('train'); EF,EL,EO=load('test')
path=glob.glob(f'work/hf/models--{a.model.replace("/","--")}/snapshots/*')[0]
cfg=json.load(open(path+'/config.json')); pad_id=cfg.get('pad_token_id',0) or 0
# labels / weights
y=tr.target.values.astype(np.float32)
idn=(tr[IDS].fillna(0).values>=0.5)
anyid=idn.any(1); pos=y>=0.5
w=1.0+anyid+(pos&~anyid)+((~pos)&anyid)*2.0   # upweight identity mentions, esp. non-toxic ones (BPSN)
w=(w/w.mean()).astype(np.float32)
Y=np.concatenate([y[:,None],tr[AUX].fillna(0).values.astype(np.float32)],1)
tri=np.where(tr.fold.values!=0)[0]; vai=np.where(tr.fold.values==0)[0]
rng.shuffle(tri)
if a.nsamp: tri=tri[:a.nsamp]
# length bucketing: chunks of bs*50, sort inside chunk, then shuffle batches
def batches(idx,lens,bs,shuffle):
    out=[]
    for c in range(0,len(idx),bs*50):
        ch=idx[c:c+bs*50]; ch=ch[np.argsort(lens[ch],kind='stable')]
        out+= [ch[i:i+bs] for i in range(0,len(ch),bs)]
    if shuffle: rng.shuffle(out)
    return out
def collate(b,F,L,O):
    m=min(int(L[b].max()),a.maxlen); x=np.full((len(b),m),pad_id,dtype=np.int64)
    for j,i in enumerate(b):
        s=F[O[i]:O[i]+min(L[i],m)]; x[j,:len(s)]=s
    x=torch.from_numpy(x); return x, (x!=pad_id).long()
class M(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.enc=AutoModel.from_pretrained(path); h=s.enc.config.hidden_size
        s.drop=torch.nn.Dropout(0.1); s.head=torch.nn.Linear(h,1+1+len(AUX))
    def forward(s,x,m):
        o=s.enc(input_ids=x,attention_mask=m).last_hidden_state
        cls=o[:,0]; mean=(o*m.unsqueeze(-1)).sum(1)/m.sum(1,keepdim=True)
        return s.head(s.drop((cls+mean)/2)).float()
model=M().to(dev)
# for models whose pad id may legitimately appear (none of ours), mask derived from pad is fine
nd=['bias','LayerNorm.weight','layernorm','norm']
params=[{'params':[p for n,p in model.named_parameters() if not any(k in n for k in nd)],'weight_decay':0.01},
        {'params':[p for n,p in model.named_parameters() if any(k in n for k in nd)],'weight_decay':0.0}]
opt=torch.optim.AdamW(params,lr=a.lr,fused=True)
tb=batches(tri,TL,a.bs,True); nsteps=len(tb)
sch=get_linear_schedule_with_warmup(opt,int(0.05*nsteps),nsteps)
bce=torch.nn.BCEWithLogitsLoss(reduction='none')
@torch.no_grad()
def predict(idx,F,L,O):
    model.eval(); out=np.zeros(len(idx),dtype=np.float32)
    order=np.argsort(L[idx]); 
    for i in range(0,len(idx),512):
        b=order[i:i+512]; x,m=collate(idx[b],F,L,O)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out[b]=torch.sigmoid(model(x.to(dev),m.to(dev))[:,0]).float().cpu().numpy()
    model.train(); return out
vdf=pd.read_parquet('work/train.parquet',columns=['target']+IDS).iloc[vai]
t0=time.time(); model.train(); evpts=set(int(nsteps*k/a.evals)-1 for k in range(1,a.evals+1)); best=-1; log=[]
print(f'{a.name}: {len(tri)} samples, {nsteps} steps',flush=True)
import datetime
dl=datetime.datetime.fromisoformat(a.deadline).timestamp() if a.deadline else 1e18
for st,b in enumerate(tb):
    if time.time()>dl:
        print('DEADLINE hit at step',st,flush=True); pv=predict(vai,TF,TL,TO); break
    x,m=collate(b,TF,TL,TO); x,m=x.to(dev,non_blocking=True),m.to(dev,non_blocking=True)
    yt=torch.from_numpy(Y[b]).to(dev); wt=torch.from_numpy(w[b]).to(dev)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        lo=model(x,m)
    l_main=(bce(lo[:,0],yt[:,0])*wt).mean()
    l_aux=bce(lo[:,1:],yt).mean()
    loss=l_main+0.5*l_aux
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    if st%500==0: print(f'step {st}/{nsteps} loss {loss.item():.4f} {time.time()-t0:.0f}s',flush=True)
    if st in evpts:
        pv=predict(vai,TF,TL,TO); s=score(vdf,pv)
        print(f'EVAL step {st} {json.dumps({k:round(v,5) for k,v in s.items()})} {time.time()-t0:.0f}s',flush=True)
        log.append((st,s['final']))
        np.save(f'work/preds/{a.name}_val_s{st}.npy',pv)
pt=predict(np.arange(len(EL)),EF,EL,EO)
np.save(f'work/preds/{a.name}_val.npy',pv); np.save(f'work/preds/{a.name}_test.npy',pt)
s=score(vdf,pv,True); print('FINAL',a.name,json.dumps(s),f'{(time.time()-t0)/60:.1f}min',flush=True)
