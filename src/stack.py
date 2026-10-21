"""Second-level stacker over OOF log-probs of several prediction sets.
logits = sum_m a_m * logp_m + mean_m(logp_m) @ W + b, W (120x120) with L2. Nested 5-fold CV on OOF rows.
usage: python stack.py [--wd X] [--out sub.csv] name1 name2 ..."""
import os, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('names', nargs='+')
ap.add_argument('--wd', type=float, default=1e-2)
ap.add_argument('--epochs', type=int, default=300)
ap.add_argument('--out', default=None)
ap.add_argument('--save', default=None, help='save OOF/test under work/preds/<save>')
args = ap.parse_args()
dev = 'cuda'
lab = pd.read_csv(os.path.join(ROOT, 'work', 'cache', 'train_ids.csv'))
sub = pd.read_csv(os.path.join(ROOT, 'data', 'sample_submission.csv'))
classes = list(sub.columns[1:])
y = lab.breed.map({c: i for i, c in enumerate(classes)}).values
folds = np.load(os.path.join(ROOT, 'work', 'folds.npy'))
P = os.path.join(ROOT, 'work', 'preds')
L = torch.tensor(np.stack([np.log(np.clip(np.load(f'{P}/{n}_oof.npy'), 1e-8, 1)) for n in args.names]), device=dev)
LT = torch.tensor(np.stack([np.log(np.clip(np.load(f'{P}/{n}_test.npy'), 1e-8, 1)) for n in args.names]), device=dev)
yt = torch.tensor(y, device=dev)
M = L.shape[0]


def fwd(Lx, a, W, b):
    return torch.einsum('m,mnc->nc', a, Lx) + Lx.mean(0) @ W + b


def fit(Lx, yy):
    a = torch.full((M,), 1.0 / M, device=dev, requires_grad=True)
    W = torch.zeros(120, 120, device=dev, requires_grad=True)
    b = torch.zeros(120, device=dev, requires_grad=True)
    opt = torch.optim.Adam([a, W, b], lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for _ in range(args.epochs):
        opt.zero_grad()
        loss = F.cross_entropy(fwd(Lx, a, W, b), yy) + args.wd * (W ** 2).sum() + args.wd * (b ** 2).sum()
        loss.backward(); opt.step(); sch.step()
    return a.detach(), W.detach(), b.detach()


oof = torch.zeros(len(y), 120, device=dev)
for k in range(5):
    tr, va = torch.tensor(folds != k, device=dev), torch.tensor(folds == k, device=dev)
    pr = fit(L[:, tr], yt[tr])
    oof[va] = torch.softmax(fwd(L[:, va], *pr), -1)
ll = F.nll_loss(torch.log(oof.clamp_min(1e-15)), yt).item()
print(f'STACK wd {args.wd} cv logloss {ll:.5f} acc {(oof.argmax(1) == yt).float().mean().item():.4f}')
pr = fit(L, yt)
print('a', pr[0].cpu().numpy().round(3))
te = torch.softmax(fwd(LT, *pr), -1).cpu().numpy()
if args.save:
    np.save(f'{P}/{args.save}_oof.npy', oof.cpu().numpy()); np.save(f'{P}/{args.save}_test.npy', te)
if args.out:
    out = pd.DataFrame(te, columns=classes); out.insert(0, 'id', sub.id.values); out.to_csv(args.out, index=False)
    print('wrote', args.out)
