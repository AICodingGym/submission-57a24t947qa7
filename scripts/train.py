"""Dense molecular pair-transformer for scalar coupling prediction.

Every molecule is a padded (N<=29) set of atoms; we keep node states h_i and pair states e_ij,
update them with pair-biased attention, outer-sum pair updates and triangle multiplicative updates,
and read out the coupling constant (plus fc/sd/pso/dso contributions as auxiliary targets) from e_ij.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--layers', type=int, default=8)
p.add_argument('--dn', type=int, default=256)
p.add_argument('--de', type=int, default=128)
p.add_argument('--heads', type=int, default=8)
p.add_argument('--tri', type=int, default=64)
p.add_argument('--bs', type=int, default=64)
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--wd', type=float, default=0.01)
p.add_argument('--epochs', type=int, default=60)
p.add_argument('--hours', type=float, default=0, help='if >0, set epochs to fit this time budget after epoch 1')
p.add_argument('--warmup', type=int, default=1000)
p.add_argument('--aux', type=float, default=0.3)
p.add_argument('--atom_aux', type=float, default=0.1)
p.add_argument('--ema', type=float, default=0.999)
p.add_argument('--eval_every', type=int, default=2)
p.add_argument('--val_frac', type=float, default=0.05)
p.add_argument('--full', type=int, default=0, help='1 = also train on validation molecules')
p.add_argument('--max_steps', type=int, default=0)
p.add_argument('--drop', type=float, default=0.0)
p.add_argument('--compile', type=int, default=0)
p.add_argument('--init', default='', help='start from this checkpoint (warm restart)')
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = 'cuda'
out_dir = f'work/runs/{args.name}'
os.makedirs(out_dir, exist_ok=True)
logf = open(f'{out_dir}/log.txt', 'a')


def log(*a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + '\n')
    logf.flush()


log(json.dumps(vars(args)))
D = torch.load('work/prep.pt')
TYPES = D['types']
NT = len(TYPES)
M = len(D['mols'])
NMAX = 29
Z = D['Z'].to(dev)
P = D['P'].to(dev)
nat = D['nat'].to(dev)
pm, a0, a1, pt = D['pm'], D['a0'], D['a1'], D['ptype']
Y = D['y']
R = len(pm)
is_test = D['is_test'].bool()

# dense per-molecule target layout: row index of each listed (a0, a1) pair
ROW = torch.full((M, NMAX, NMAX), -1, dtype=torch.int32)
ROW[pm, a0, a1] = torch.arange(R, dtype=torch.int32)
ROW = ROW.to(dev)
TYP = torch.full((M, NMAX, NMAX), -1, dtype=torch.int8)
TYP[pm, a0, a1] = pt.to(torch.int8)
TYP = TYP.to(dev)
Ydev = Y.to(dev)  # (R, 5); nan for test
mull = D['mull'].to(dev)
shield = D['shield'].to(dev)

# molecule split
mol_test = torch.zeros(M, dtype=torch.bool)
mol_test[pm[is_test]] = True
train_mols = torch.nonzero(~mol_test).squeeze(1).numpy()
rng = np.random.RandomState(1234)  # fixed split independent of seed
perm = rng.permutation(train_mols)
nval = int(len(perm) * args.val_frac)
val_mols = np.sort(perm[:nval])
fit_mols = np.sort(perm[nval:]) if not args.full else np.sort(train_mols)
test_mols = torch.nonzero(mol_test).squeeze(1).numpy()
val_rows = torch.from_numpy(np.isin(pm.numpy(), val_mols))
log('mols fit/val/test', len(fit_mols), len(val_mols), len(test_mols))

# per-type normalisation stats (from fit rows)
fit_rows = torch.from_numpy(np.isin(pm.numpy(), fit_mols))
mu = torch.zeros(NT, 5)
sd = torch.ones(NT, 5)
for t in range(NT):
    yy = Y[fit_rows & (pt == t)]
    mu[t] = yy.mean(0)
    sd[t] = yy.std(0).clamp_min(1e-3)
mu, sd = mu.to(dev), sd.to(dev)
mull_mu, mull_sd = 0.0, float(np.nanstd(D['mull'].numpy()))
sh = D['shield'].numpy().reshape(-1, 4)
sh_mu = torch.tensor(np.nanmean(sh, 0), device=dev)
sh_sd = torch.tensor(np.nanstd(sh, 0), device=dev)

RAD = torch.tensor([0.0, 0.38, 0.77, 0.75, 0.73, 0.71], device=dev)


def make_batch(mids):
    mids = torch.as_tensor(mids, device=dev)
    n = int(nat[mids].max())
    if args.compile:  # few distinct shapes -> few recompiles
        n = min(NMAX, (n + 3) // 4 * 4)
    z = Z[mids, :n]
    pos = P[mids, :n]
    mask = z > 0
    B = len(mids)
    dvec = pos[:, None, :, :] - pos[:, :, None, :]  # (B,i,j,3) = pos_j - pos_i
    d = dvec.norm(dim=-1)
    pm2 = mask[:, :, None] & mask[:, None, :]
    eye = torch.eye(n, device=dev, dtype=torch.bool)[None]
    r = RAD[z]
    bond = (d < 1.15 * (r[:, :, None] + r[:, None, :])) & pm2 & ~eye
    A = bond.float()
    # topological distance via BFS on adjacency powers
    topo = torch.full((B, n, n), 8, device=dev, dtype=torch.long)
    topo[eye.expand(B, n, n)] = 0
    reach = eye.expand(B, n, n).float()
    cur = reach
    for k in range(1, 8):
        cur = (torch.bmm(cur, A) > 0).float()
        new = (cur > 0) & (topo == 8)
        topo[new] = k
    topo[~pm2] = 8
    # path geometry
    u = dvec / d.clamp_min(1e-6)[..., None]  # unit vector i->j
    # 2-bond paths: middle atom k with A[i,k] A[k,j]
    P2 = A[:, :, :, None] * A[:, None, :, :]  # P2[b,i,k,j] = A[b,i,k] * A[b,k,j]
    kmid = P2.argmax(2)  # (B,i,j)
    bi = torch.arange(B, device=dev)[:, None, None]
    ii = torch.arange(n, device=dev)[None, :, None]
    jj = torch.arange(n, device=dev)[None, None, :]
    # angle i-k-j: vectors k->i and k->j
    uki = u[bi, kmid, ii]
    ukj = u[bi, kmid, jj]
    cos2 = (uki * ukj).sum(-1)
    has2 = (topo == 2).float()
    # 3-bond paths i-k-l-j: P3[i,k,l,j] = A[i,k] A[k,l] A[l,j]
    P3 = A[:, :, :, None, None] * A[:, None, :, :, None] * A[:, None, None, :, :]
    flat = P3.reshape(B, n, n * n, n).argmax(2)
    k3, l3 = flat // n, flat % n
    b_ik = pos[bi, k3] - pos[bi, ii.expand_as(k3)]
    b_kl = pos[bi, l3] - pos[bi, k3]
    b_lj = pos[bi, jj.expand_as(l3)] - pos[bi, l3]
    n1 = torch.cross(b_ik, b_kl, dim=-1)
    n2 = torch.cross(b_kl, b_lj, dim=-1)
    cosphi = (n1 * n2).sum(-1) / (n1.norm(dim=-1) * n2.norm(dim=-1)).clamp_min(1e-6)
    cos_a1 = (-(b_ik) * b_kl).sum(-1) / (b_ik.norm(dim=-1) * b_kl.norm(dim=-1)).clamp_min(1e-6)
    cos_a2 = (-(b_kl) * b_lj).sum(-1) / (b_kl.norm(dim=-1) * b_lj.norm(dim=-1)).clamp_min(1e-6)
    has3 = (topo == 3).float()
    geo = torch.stack([cos2 * has2, has2, cosphi * has3, (2 * cosphi ** 2 - 1) * has3,
                       cos_a1 * has3, cos_a2 * has3, has3, P3.sum((2, 3)).clamp_max(3) / 3 * has3], -1)
    # node features: degree and number of H neighbours
    deg = A.sum(-1).long().clamp_max(5)
    nh = (A * (z == 1).float()[:, None, :]).sum(-1).long().clamp_max(4)
    rows = ROW[mids, :n, :n]
    typ = TYP[mids, :n, :n].long()
    return dict(z=z, mask=mask, d=d, topo=topo, geo=geo, deg=deg, nh=nh, rows=rows, typ=typ, mids=mids, n=n)


if args.name == 'check':  # sanity: topological distance vs coupling type "nJ"
    nb = torch.tensor([int(t[0]) for t in TYPES], device=dev)
    agree, tot = 0, 0
    for s in range(0, M, 256):
        b = make_batch(np.arange(s, min(M, s + 256)))
        sel = b['rows'] >= 0
        agree += (b['topo'][sel] == nb[b['typ'][sel]]).sum().item()
        tot += sel.sum().item()
    print('topo==n agreement', agree / tot, tot)
    raise SystemExit


class RBF(nn.Module):
    def __init__(self, k=64, lo=0.5, hi=8.0):
        super().__init__()
        self.register_buffer('c', torch.linspace(lo, hi, k))
        self.g = 1.0 / (2 * ((hi - lo) / k) ** 2)

    def forward(self, d):
        return torch.exp(-self.g * (d[..., None] - self.c) ** 2)


class Layer(nn.Module):
    def __init__(self, dn, de, H, tri, drop):
        super().__init__()
        self.H = H
        self.ln_h = nn.LayerNorm(dn)
        self.ln_e = nn.LayerNorm(de)
        self.qkv = nn.Linear(dn, 3 * dn)
        self.eb = nn.Linear(de, H)
        self.ev = nn.Linear(de, dn)
        self.o = nn.Linear(dn, dn)
        self.ffn_h = nn.Sequential(nn.LayerNorm(dn), nn.Linear(dn, 4 * dn), nn.GELU(), nn.Linear(4 * dn, dn))
        # pair update from nodes
        self.ln_h2 = nn.LayerNorm(dn)
        self.hi = nn.Linear(dn, de)
        self.hj = nn.Linear(dn, de)
        self.pe = nn.Sequential(nn.LayerNorm(de), nn.Linear(de, de))
        self.pout = nn.Sequential(nn.GELU(), nn.Linear(de, de))
        # triangle multiplicative update (outgoing + incoming)
        self.t_ln = nn.LayerNorm(de)
        self.t_ab = nn.Linear(de, 4 * tri)
        self.t_g = nn.Linear(de, 4 * tri)
        self.t_ln2 = nn.LayerNorm(2 * tri)
        self.t_o = nn.Linear(2 * tri, de)
        self.t_go = nn.Linear(de, de)
        self.ffn_e = nn.Sequential(nn.LayerNorm(de), nn.Linear(de, 4 * de), nn.GELU(), nn.Linear(4 * de, de))
        self.drop = nn.Dropout(drop)

    def forward(self, h, e, mask, pmask):
        B, n, dn = h.shape
        H = self.H
        x = self.ln_h(h)
        ee = self.ln_e(e)
        q, k, v = self.qkv(x).view(B, n, 3, H, dn // H).unbind(2)
        att = torch.einsum('bihd,bjhd->bhij', q, k) / math.sqrt(dn // H) + self.eb(ee).permute(0, 3, 1, 2)
        att = att.masked_fill(~mask[:, None, None, :], -1e4).float().softmax(-1).to(v.dtype)
        o = torch.einsum('bhij,bjhd->bihd', att, v).reshape(B, n, dn)
        o = o + torch.einsum('bhij,bijhd->bihd', att, self.ev(ee).view(B, n, n, H, dn // H)).reshape(B, n, dn)
        h = h + self.drop(self.o(o))
        h = h + self.drop(self.ffn_h(h))
        x = self.ln_h2(h)
        e = e + self.drop(self.pout(self.pe(e) + self.hi(x)[:, :, None] + self.hj(x)[:, None, :]))
        # triangle
        t = self.t_ln(e)
        ab = torch.sigmoid(self.t_g(t)) * self.t_ab(t) * pmask[..., None]
        a_o, b_o, a_i, b_i = ab.chunk(4, -1)
        out_o = torch.einsum('bikc,bjkc->bijc', a_o, b_o)
        out_i = torch.einsum('bkic,bkjc->bijc', a_i, b_i)
        tt = self.t_o(self.t_ln2(torch.cat([out_o, out_i], -1) / n))
        e = e + self.drop(torch.sigmoid(self.t_go(t)) * tt)
        e = e + self.drop(self.ffn_e(e))
        e = e * pmask[..., None]
        return h, e


class Net(nn.Module):
    def __init__(self, a):
        super().__init__()
        dn, de = a.dn, a.de
        self.z_emb = nn.Embedding(6, dn)
        self.deg_emb = nn.Embedding(6, dn)
        self.nh_emb = nn.Embedding(5, dn)
        self.rbf = RBF(96, 0.5, 9.0)
        self.topo_emb = nn.Embedding(9, de)
        self.zz_emb = nn.Embedding(36, de)
        self.e_in = nn.Sequential(nn.Linear(96 + 8 + 1, de), nn.GELU(), nn.Linear(de, de))
        self.layers = nn.ModuleList([Layer(dn, de, a.heads, a.tri, a.drop) for _ in range(a.layers)])
        self.ln_e = nn.LayerNorm(de)
        self.ln_h = nn.LayerNorm(dn)
        self.head = nn.Sequential(nn.Linear(2 * de + 2 * dn, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(),
                                  nn.Linear(256, NT * 5))
        self.atom_head = nn.Sequential(nn.Linear(dn, 256), nn.GELU(), nn.Linear(256, 5))

    def forward(self, b):
        z, mask = b['z'], b['mask']
        pmask = (mask[:, :, None] & mask[:, None, :]).float()
        h = self.z_emb(z) + self.deg_emb(b['deg']) + self.nh_emb(b['nh'])
        d = b['d']
        ef = torch.cat([self.rbf(d), b['geo'], (1.0 / d.clamp_min(0.5))[..., None]], -1)
        e = self.e_in(ef) + self.topo_emb(b['topo']) + self.zz_emb(z[:, :, None] * 6 + z[:, None, :])
        e = e * pmask[..., None]
        for L in self.layers:
            h, e = L(h, e, mask, pmask)
        return h, e

    def readout(self, h, e, bi, i, j, typ):
        h = self.ln_h(h)
        e = self.ln_e(e)
        x = torch.cat([e[bi, i, j], e[bi, j, i], h[bi, i], h[bi, j]], -1)
        o = self.head(x).view(-1, NT, 5)
        return o[torch.arange(len(typ), device=o.device), typ]  # (K,5) normalised


model = Net(args).to(dev)
if args.init:
    model.load_state_dict(torch.load(args.init)['model'])
ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema))
if args.compile:
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    model.forward = torch.compile(model.forward)
log('params', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))


def batches(mols, bs, shuffle):
    mols = np.asarray(mols)
    sizes = D['nat'].numpy()[mols]
    if shuffle:
        mols = mols[np.random.permutation(len(mols))]
        sizes = D['nat'].numpy()[mols]
        # sort within chunks of 50 batches to reduce padding while keeping randomness
        out = []
        C = bs * 50
        for s in range(0, len(mols), C):
            m = mols[s:s + C]
            m = m[np.argsort(sizes[s:s + C], kind='stable')]
            out += [m[k:k + bs] for k in range(0, len(m), bs)]
        np.random.shuffle(out)
        return out
    o = np.argsort(sizes, kind='stable')
    m = mols[o]
    return [m[k:k + bs] for k in range(0, len(m), bs)]


type_w = torch.ones(NT, device=dev)  # running MAE per type (normalised units) -> loss weight 1/mae (log-MAE metric)


def step_loss(mdl, b):
    h, e = mdl(b)
    sel = b['rows'] >= 0
    bi, i, j = torch.nonzero(sel, as_tuple=True)
    typ = b['typ'][bi, i, j]
    rows = b['rows'][bi, i, j].long()
    pred = mdl.readout(h, e, bi, i, j, typ).float()
    y = (Ydev[rows] - mu[typ]) / sd[typ]
    err = (pred - y).abs()  # (K,5)
    # per-type mean of main error, weighted by 1/running-mae
    cnt = torch.zeros(NT, device=dev).index_add_(0, typ, torch.ones_like(typ, dtype=torch.float))
    s_main = torch.zeros(NT, device=dev).index_add_(0, typ, err[:, 0])
    s_aux = torch.zeros(NT, device=dev).index_add_(0, typ, err[:, 1:].mean(1))
    present = cnt > 0
    mae_t = s_main / cnt.clamp_min(1)
    aux_t = s_aux / cnt.clamp_min(1)
    loss = ((mae_t + args.aux * aux_t) / type_w)[present].mean()
    # atom aux: mulliken + shielding
    hn = mdl.atom_head(h).float()
    am = mull[b['mids'], :b['n']]
    ash = shield[b['mids'], :b['n']]
    m_ok = b['mask'] & ~torch.isnan(am)
    if m_ok.any():
        la = (hn[..., 0] - am / mull_sd).abs()[m_ok].mean() + (hn[..., 1:] - (ash - sh_mu) / sh_sd).abs()[m_ok].mean()
        loss = loss + args.atom_aux * la
    return loss, mae_t.detach(), present


@torch.no_grad()
def predict(mdl, mols):
    mdl.eval()
    out = torch.full((R,), float('nan'), device=dev)
    for mb in batches(mols, 128, False):
        b = make_batch(mb)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h, e = mdl(b)
            sel = b['rows'] >= 0
            bi, i, j = torch.nonzero(sel, as_tuple=True)
            typ = b['typ'][bi, i, j]
            rows = b['rows'][bi, i, j].long()
            pr = mdl.readout(h, e, bi, i, j, typ).float()
        out[rows] = pr[:, 0] * sd[typ, 0] + mu[typ, 0]
    mdl.train()
    return out.cpu()


def score(pred):
    s = []
    per = {}
    for t in range(NT):
        m = val_rows & (pt == t)
        mae = (pred[m] - Y[m, 0]).abs().mean().item()
        per[TYPES[t]] = round(math.log(max(mae, 1e-9)), 4)
        s.append(math.log(max(mae, 1e-9)))
    return float(np.mean(s)), per


steps_per_epoch = math.ceil(len(fit_mols) / args.bs)
total_steps = args.epochs * steps_per_epoch
sched_total = [total_steps]


def lr_at(s):
    if s < args.warmup:
        return args.lr * s / args.warmup
    prog = min(1.0, (s - args.warmup) / max(1, sched_total[0] - args.warmup))
    return args.lr * 0.5 * (1 + math.cos(math.pi * prog))


step = 0
t0 = time.time()
best = 1e9
last_save = time.time()
ep = 0
while ep < args.epochs:
    model.train()
    run_loss = 0.0
    nb = 0
    for mb in batches(fit_mols, args.bs, True):
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        b = make_batch(mb)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, mae_t, present = step_loss(model, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update_parameters(model)
        type_w[present] = 0.99 * type_w[present] + 0.01 * mae_t[present]
        run_loss += loss.item()
        nb += 1
        step += 1
        if args.max_steps and step >= args.max_steps:
            break
    ep += 1
    el = time.time() - t0
    if ep == 1 and args.hours > 0:
        ep_time = el
        args.epochs = max(2, int(args.hours * 3600 / (ep_time * 1.04)))
        sched_total[0] = args.epochs * steps_per_epoch
        log('epoch time', round(ep_time, 1), 's -> epochs', args.epochs)
    log(f'ep {ep}/{args.epochs} step {step} loss {run_loss / max(nb, 1):.4f} lr {lr_at(step):.2e} '
        f'tw {[round(x, 3) for x in type_w.tolist()]} el {el / 60:.1f}m')
    if args.max_steps and step >= args.max_steps:
        break
    if ep % args.eval_every == 0 or ep == args.epochs or time.time() - last_save > 1200:
        vp = predict(ema.module, val_mols)
        sc, per = score(vp)
        log(f'  val(ema) {sc:.4f} {per}')
        tp = predict(ema.module, test_mols)
        allp = torch.where(is_test, tp, vp)
        np.save(f'{out_dir}/pred.npy', allp.numpy())
        torch.save({'model': ema.module.state_dict(), 'args': vars(args), 'ep': ep, 'val': sc}, f'{out_dir}/ckpt.pt')
        json.dump({'val': sc, 'per_type': per, 'ep': ep, 'epochs': args.epochs, 'minutes': el / 60},
                  open(f'{out_dir}/status.json', 'w'))
        last_save = time.time()
log('done')
