"""BiLSTM sequence model for ventilator pressure. One fold per process; all data lives on the GPU."""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--fold', type=int, default=0)
p.add_argument('--nfolds', type=int, default=5)
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--bs', type=int, default=512)
p.add_argument('--lr', type=float, default=2e-3)
p.add_argument('--wd', type=float, default=1e-5)
p.add_argument('--hidden', type=str, default='512,384,256,128')
p.add_argument('--head', type=int, default=128)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--exp_w', type=float, default=0.0, help='loss weight for expiratory (u_out=1) steps')
p.add_argument('--eval_every', type=int, default=5)
p.add_argument('--warmup', type=int, default=3)
p.add_argument('--model', type=str, default='lstm')
p.add_argument('--conv', type=int, default=0, help='prepend conv block with this many channels')
p.add_argument('--max_minutes', type=float, default=1e9)
p.add_argument('--loss', type=str, default='l1')
p.add_argument('--gpu', type=int, default=0, help='index within CUDA_VISIBLE_DEVICES')
p.add_argument('--cls', type=int, default=0, help='classification over the pressure grid')
p.add_argument('--cls_l1', type=float, default=1.0)
p.add_argument('--drop', type=float, default=0.0, help='dropout between LSTM layers')
p.add_argument('--amp', type=int, default=0, help='fp16 autocast for the LSTM body (head stays fp32)')
p.add_argument('--rnn', type=str, default='lstm', help='lstm or gru')
p.add_argument('--full', type=int, default=0, help='train on all data (no validation)')
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = f'cuda:{args.gpu}'
torch.cuda.set_device(dev)
D = 'work/data'
OUT = f'work/preds/{args.name}'
os.makedirs(OUT, exist_ok=True)
os.makedirs('work/ckpt', exist_ok=True)


def feats(a):
    # a: (N, 80, 5) R, C, time_step, u_in, u_out
    R, C, t, u, uo = [a[..., i] for i in range(5)]
    N = a.shape[0]
    dt = np.diff(t, axis=1, prepend=t[:, :1])
    f = {}
    f['R'] = R; f['C'] = C
    for r in (5, 20, 50): f[f'R{r}'] = (R == r).astype(np.float32)
    for c in (10, 20, 50): f[f'C{c}'] = (C == c).astype(np.float32)
    for r in (5, 20, 50):
        for c in (10, 20, 50): f[f'RC{r}_{c}'] = ((R == r) & (C == c)).astype(np.float32)
    f['t'] = t; f['dt'] = dt
    f['u_in'] = u; f['u_out'] = uo
    f['cum'] = np.cumsum(u, 1)
    f['area'] = np.cumsum(u * dt, 1)
    f['area_insp'] = np.cumsum(u * dt * (1 - uo), 1)
    f['cnt'] = np.cumsum(np.ones_like(u), 1)
    f['emean'] = f['cum'] / f['cnt']
    for k in (1, 2, 3, 4):
        lag = np.concatenate([np.zeros((N, k), np.float32), u[:, :-k]], 1)
        lead = np.concatenate([u[:, k:], np.zeros((N, k), np.float32)], 1)
        f[f'lag{k}'] = lag; f[f'lead{k}'] = lead
        f[f'dlag{k}'] = u - lag; f[f'dlead{k}'] = u - lead
        f[f'uolag{k}'] = np.concatenate([np.zeros((N, k), np.float32), uo[:, :-k]], 1)
        f[f'uolead{k}'] = np.concatenate([uo[:, k:], np.ones((N, k), np.float32)], 1)
    f['bmax'] = np.repeat(u.max(1, keepdims=True), 80, 1)
    f['bmean'] = np.repeat(u.mean(1, keepdims=True), 80, 1)
    f['u_minus_mean'] = u - f['bmean']
    f['u_minus_max'] = u - f['bmax']
    insp_len = (1 - uo).sum(1, keepdims=True)
    f['insp_len'] = np.repeat(insp_len, 80, 1)
    f['ewm'] = ewm(u, 0.3)
    f['ewm2'] = ewm(u, 0.1)
    f['area_RC'] = f['area'] / C
    f['uR'] = u * R / 50
    X = np.stack([f[k].astype(np.float32) for k in f], -1)
    return X, list(f)


def ewm(u, alpha):
    out = np.zeros_like(u); s = u[:, 0].copy()
    for i in range(u.shape[1]):
        s = alpha * u[:, i] + (1 - alpha) * s if i else s
        out[:, i] = s
    return out


tr = np.load(f'{D}/train_arr.npy'); te = np.load(f'{D}/test_arr.npy')
Xtr, names = feats(tr[..., :5]); Xte, _ = feats(te)
ytr = tr[..., 5]; mtr = 1 - tr[..., 4]; mte = 1 - te[..., 4]
allX = np.concatenate([Xtr, Xte]).reshape(-1, Xtr.shape[-1])
mu, sd = allX.mean(0), allX.std(0) + 1e-6
Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
del allX

N = len(Xtr)
rng = np.random.RandomState(1234)  # fixed fold split across all experiments
perm = rng.permutation(N)
folds = np.zeros(N, int)
folds[perm] = np.arange(N) % args.nfolds
if args.full:
    tri = np.arange(N); vai = np.arange(N)[folds == args.fold]
else:
    tri = np.where(folds != args.fold)[0]; vai = np.where(folds == args.fold)[0]

def nearest_idx(x, g):
    i = np.clip(np.searchsorted(g, x), 1, len(g) - 1)
    return np.where(x - g[i - 1] < g[i] - x, i - 1, i)


T = lambda x: torch.tensor(x, device=dev)
Xtr_t, ytr_t, mtr_t = T(Xtr), T(ytr), T(mtr.astype(np.float32))
Xte_t = T(Xte)
grid = np.load(f'{D}/pressure_grid.npy').astype(np.float32); K = len(grid)
grid_t = T(grid)
ytr_idx_t = torch.tensor(nearest_idx(ytr, grid) if args.cls else np.zeros(1), device=dev).long()
F = Xtr.shape[-1]
print('features', F, names, flush=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        hs = [int(h) for h in args.hidden.split(',')]
        inp = F
        self.conv = None
        if args.conv:
            self.conv = nn.Sequential(nn.Conv1d(F, args.conv, 3, padding=1), nn.GELU(),
                                      nn.Conv1d(args.conv, args.conv, 3, padding=1), nn.GELU())
            inp = F + args.conv
        self.lstms = nn.ModuleList()
        for h in hs:
            self.lstms.append((nn.GRU if args.rnn == 'gru' else nn.LSTM)(inp, h, batch_first=True, bidirectional=True))
            inp = 2 * h
        self.head = nn.Sequential(nn.Linear(inp, args.head), nn.SELU(), nn.Linear(args.head, K if args.cls else 1))

    def forward(self, x):
        if self.conv is not None:
            x = torch.cat([x, self.conv(x.transpose(1, 2)).transpose(1, 2)], -1)
        for j, l in enumerate(self.lstms):
            if j and args.drop:
                x = nn.functional.dropout(x, args.drop, self.training)
            with torch.autocast('cuda', dtype=torch.float16, enabled=bool(args.amp)):
                x, _ = l(x)
        o = self.head(x.float())
        return o if args.cls else o.squeeze(-1)


model = Net().to(dev)
print('params', sum(p.numel() for p in model.parameters()), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
scaler = torch.amp.GradScaler(enabled=bool(args.amp))
steps_per_epoch = math.ceil(len(tri) / args.bs)
total = steps_per_epoch * args.epochs
warm = steps_per_epoch * args.warmup
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))


def predict(X, mode='exp'):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            o = model(X[i:i + 2048])
            if args.cls:
                pr_ = o.float().softmax(-1)
                if mode == 'exp':
                    o = (pr_ * grid_t).sum(-1)
                else:  # median of predicted distribution
                    o = grid_t[(pr_.cumsum(-1) < 0.5).sum(-1).clamp(max=K - 1)]
            out.append(o)
    model.train()
    return torch.cat(out).float().cpu().numpy()


best = 1e9; t0 = time.time(); hist = []
vai_t = torch.tensor(vai, device=dev)
for ep in range(args.epochs):
    idx = torch.tensor(np.random.permutation(tri), device=dev)
    tl = 0.
    for i in range(0, len(idx), args.bs):
        b = idx[i:i + args.bs]
        x, y, m = Xtr_t[b], ytr_t[b], mtr_t[b]
        pr = model(x)
        w = m + args.exp_w * (1 - m)
        if args.cls:
            ce = nn.functional.cross_entropy(pr.reshape(-1, K), ytr_idx_t[b].reshape(-1), reduction='none').view_as(y)
            ev = (pr.softmax(-1) * grid_t).sum(-1)
            l = ((ce + args.cls_l1 * (ev - y).abs()) * w).sum() / w.sum()
        elif args.loss == 'l1':
            l = ((pr - y).abs() * w).sum() / w.sum()
        else:
            l = (nn.functional.smooth_l1_loss(pr, y, reduction='none', beta=0.1) * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        scaler.scale(l).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update(); sched.step()
        tl += l.item() * len(b)
    tl /= len(idx)
    elapsed = (time.time() - t0) / 60
    last = ep == args.epochs - 1 or elapsed > args.max_minutes
    if (ep + 1) % args.eval_every == 0 or last:
        pv = predict(Xtr_t[vai_t])
        mae = float((np.abs(pv - ytr[vai]) * mtr[vai]).sum() / mtr[vai].sum())
        if args.cls:
            pm = predict(Xtr_t[vai_t], 'med')
            mae_med = float((np.abs(pm - ytr[vai]) * mtr[vai]).sum() / mtr[vai].sum())
            print(f'   median-readout val {mae_med:.4f}', flush=True)
        hist.append((ep + 1, tl, mae))
        print(f'ep {ep+1} train {tl:.4f} val {mae:.4f} lr {sched.get_last_lr()[0]:.2e} {elapsed:.1f}m', flush=True)
        # with --full the "val" fold is in training, so always keep the latest model
        if mae < best or args.full:
            best = mae
            np.save(f'{OUT}/oof_f{args.fold}.npy', pv)
            np.save(f'{OUT}/oofidx_f{args.fold}.npy', vai)
            np.save(f'{OUT}/test_f{args.fold}.npy', predict(Xte_t))
            torch.save(model.state_dict(), f'work/ckpt/{args.name}_f{args.fold}.pt')
            json.dump({'best': best, 'epoch': ep + 1, 'args': vars(args), 'hist': hist},
                      open(f'{OUT}/info_f{args.fold}.json', 'w'))
    if last:
        break
print('done best', best, 'minutes', (time.time() - t0) / 60, flush=True)
