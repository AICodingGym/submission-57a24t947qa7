import argparse, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
p=argparse.ArgumentParser()
p.add_argument('--arch',default='simple'); p.add_argument('--seed',type=int,default=0)
p.add_argument('--epochs',type=int,default=60); p.add_argument('--bs',type=int,default=32)
p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--size',type=int,default=75)
p.add_argument('--use_inc',type=int,default=0); p.add_argument('--gpu',type=int,default=0); p.add_argument('--pseudo',default=''); p.add_argument('--name',required=True)
a=p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev=f'cuda:{a.gpu}'; torch.cuda.set_device(dev)
X=np.load('work/Xtr.npy'); XT=np.load('work/Xte.npy')
mt=pd.read_csv('work/meta_tr.csv'); me=pd.read_csv('work/meta_te.csv'); folds=np.load('work/folds.npy')
y=mt.y.values.astype(np.float32)
def norm(X):
    X=np.concatenate([X,(X[:,:1]+X[:,1:2])/2],1)
    mu=np.array([-20.6,-26.3,-23.5],np.float32)[None,:,None,None]; sd=np.array([5.2,3.4,3.9],np.float32)[None,:,None,None]
    return (X-mu)/sd
X=norm(X); XT=norm(XT)
incm=39.27; inc=lambda m: ((m.inc.fillna(incm).values-incm)/4).astype(np.float32)
I=inc(mt); IT=inc(me)

class Simple(nn.Module):
    def __init__(s, use_inc):
        super().__init__()
        def blk(i,o,d=0.2): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout(d))
        s.f=nn.Sequential(blk(3,32),blk(32,64),blk(64,128),blk(128,256))
        s.use_inc=use_inc; s.h=nn.Sequential(nn.Linear(512+use_inc,128),nn.ReLU(),nn.Dropout(0.4),nn.Linear(128,1))
    def forward(s,x,i):
        z=s.f(x); z=torch.cat([F.adaptive_avg_pool2d(z,1).flatten(1),F.adaptive_max_pool2d(z,1).flatten(1)],1)
        if s.use_inc: z=torch.cat([z,i[:,None]],1)
        return s.h(z).squeeze(1)
class Timm(nn.Module):
    def __init__(s,name,use_inc):
        super().__init__(); import timm
        s.b=timm.create_model(name,pretrained=True,in_chans=3,num_classes=0,drop_rate=0.3)
        s.use_inc=use_inc; s.h=nn.Linear(s.b.num_features+use_inc,1)
    def forward(s,x,i):
        if a.size!=75: x=F.interpolate(x,size=a.size,mode='bilinear',align_corners=False)
        z=s.b(x)
        if s.use_inc: z=torch.cat([z,i[:,None]],1)
        return s.h(z).squeeze(1)
def make(): return (Simple(a.use_inc) if a.arch=='simple' else Timm(a.arch,a.use_inc)).to(dev)

def aug(x):
    # x: B,C,H,W tensor on gpu; random dihedral + small shift
    if np.random.rand()<0.5: x=x.flip(3)
    if np.random.rand()<0.5: x=x.flip(2)
    k=np.random.randint(4); x=torch.rot90(x,k,(2,3))
    dx,dy=np.random.randint(-6,7,2); x=torch.roll(x,(int(dx),int(dy)),(2,3))
    return x
def tta(model,x,i):
    model.eval(); outs=[]
    with torch.no_grad():
        for fl in [False,True]:
            for k in range(4):
                xx=x.flip(3) if fl else x; xx=torch.rot90(xx,k,(2,3))
                outs.append(model(xx,i))
    return torch.stack(outs).mean(0)  # mean logits
def predict(model,X,I):
    out=[]
    for s in range(0,len(X),128):
        out.append(tta(model,torch.tensor(X[s:s+128],device=dev),torch.tensor(I[s:s+128],device=dev)).cpu().numpy())
    return np.concatenate(out)

t0=time.time(); oof=np.zeros(len(X)); te=np.zeros((5,len(XT)))
for k in range(5):
    tri=np.where(folds!=k)[0]; vai=np.where(folds==k)[0]
    Xall,Iall,yall=X,I,y
    if a.pseudo:
        pl=np.load(a.pseudo)[k].astype(np.float32); Xall=np.concatenate([X,XT]); Iall=np.concatenate([I,IT]); yall=np.concatenate([y,pl])
        tpart=np.random.RandomState(123).permutation(len(XT))%5
        tri=np.concatenate([tri,len(X)+np.where(tpart!=k)[0]])
    model=make(); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    steps=a.epochs*int(np.ceil(len(tri)/a.bs)); sch=torch.optim.lr_scheduler.OneCycleLR(opt,a.lr,total_steps=steps,pct_start=0.1)
    Xg=torch.tensor(Xall,device=dev); Ig=torch.tensor(Iall,device=dev); yg=torch.tensor(yall,device=dev)
    for ep in range(a.epochs):
        model.train(); perm=np.random.permutation(tri)
        for s in range(0,len(perm),a.bs):
            b=perm[s:s+a.bs]
            if len(b)<2: continue
            xb=aug(Xg[b]); loss=F.binary_cross_entropy_with_logits(model(xb,Ig[b]),yg[b])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    oof[vai]=predict(model,X[vai],I[vai]); te[k]=predict(model,XT,IT)
    if a.pseudo: te[k][tpart!=k]=0
    pv=1/(1+np.exp(-oof[vai])); ll=-np.mean(y[vai]*np.log(np.clip(pv,1e-7,1))+(1-y[vai])*np.log(np.clip(1-pv,1e-7,1)))
    print(f'fold {k} logloss {ll:.4f} t={time.time()-t0:.0f}s',flush=True)
from sklearn.metrics import log_loss
P=1/(1+np.exp(-oof)); print(f'{a.name} OOF logloss {log_loss(y,P):.4f} time {time.time()-t0:.0f}s',flush=True)
np.save(f'work/preds/{a.name}_oof.npy',oof); np.save(f'work/preds/{a.name}_te.npy',te.sum(0) if a.pseudo else te.mean(0))
