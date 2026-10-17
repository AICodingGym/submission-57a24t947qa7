"""Speech commands: 30-word classifier on GPU log-mels, mapped to 12 competition labels.
All audio lives on the GPU; augmentation (shift, gain, background noise, SpecAugment, mixup) is done there."""
import argparse, json, math, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold

p = argparse.ArgumentParser()
p.add_argument('--arch', default='resnet')       # resnet | timm:<name>
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--bs', type=int, default=256)
p.add_argument('--lr', type=float, default=3e-3)
p.add_argument('--wd', type=float, default=1e-2)
p.add_argument('--width', type=int, default=64)
p.add_argument('--n_mels', type=int, default=64)
p.add_argument('--folds', default='0')            # comma list of folds to run, or 'all' to train on everything
p.add_argument('--nfolds', type=int, default=5)
p.add_argument('--mixup', type=float, default=0.0)
p.add_argument('--noise_p', type=float, default=0.6)
p.add_argument('--ls', type=float, default=0.05)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--name', required=True)
args = p.parse_args()

CMDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
dev = 'cuda'
C = 'work/cache'
OUT = f'work/runs/{args.name}'
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(args.seed); np.random.seed(args.seed)

meta = pd.read_csv(f'{C}/train_meta.csv')
WORDS = sorted(meta.word.unique())
y_all = meta.word.map({w: i for i, w in enumerate(WORDS)}).values
X = torch.from_numpy(np.load(f'{C}/train_x.npy')).to(dev)            # int16 [N,16000]
XT = torch.from_numpy(np.load(f'{C}/test_x.npy')).to(dev)
BG = torch.from_numpy(np.load(f'{C}/bg.npy')).to(dev)
Y = torch.from_numpy(y_all).to(dev)
# map 30 words -> 11 test labels (10 commands + unknown); silence handled by rule
lab12 = CMDS + ['unknown', 'silence']
w2l = torch.tensor([lab12.index(w) if w in CMDS else 10 for w in WORDS], device=dev)

# ---------------- features ----------------
SR, NFFT, HOP = 16000, 480, 160
def mel_fb(n_mels, n_fft=NFFT, sr=SR, fmin=20, fmax=8000):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)), np.float32)
    for i in range(n_mels):
        l, c, r = pts[i:i + 3]
        fb[i] = np.clip(np.minimum((freqs - l) / (c - l), (r - freqs) / (r - c)), 0, None)
    return torch.from_numpy(fb).to(dev)
FB = mel_fb(args.n_mels)
WIN = torch.hann_window(NFFT, device=dev)
def logmel(w):  # w float [B,16000] -> [B,1,n_mels,T]
    s = torch.stft(w, NFFT, HOP, window=WIN, return_complex=True).abs() ** 2
    m = torch.matmul(FB, s)
    return torch.log(m + 1e-6).unsqueeze(1)

def augment(w):
    B = w.shape[0]
    # time shift up to +-100ms
    sh = torch.randint(-1600, 1601, (B,), device=dev)
    idx = (torch.arange(16000, device=dev)[None] - sh[:, None])
    valid = (idx >= 0) & (idx < 16000)
    w = torch.gather(w, 1, idx.clamp(0, 15999)) * valid
    # gain
    w = w * torch.exp(torch.empty(B, 1, device=dev).uniform_(math.log(0.5), math.log(1.5)))
    # background noise
    st = torch.randint(0, BG.shape[0] - 16000, (B,), device=dev)
    noise = BG[st[:, None] + torch.arange(16000, device=dev)[None]]
    amp = torch.empty(B, 1, device=dev).uniform_(0, 0.3) * (torch.rand(B, 1, device=dev) < args.noise_p)
    w = w + noise * amp
    # speed/pitch perturbation via resampling (0.85-1.15)
    if True:
        rate = torch.empty(B, device=dev).uniform_(0.85, 1.15)
        pos = (torch.arange(16000, device=dev)[None].float() - 8000) * rate[:, None] + 8000
        lo = pos.floor().long(); fr = pos - lo
        ok = (lo >= 0) & (lo < 15999)
        lo = lo.clamp(0, 15998)
        w = (torch.gather(w, 1, lo) * (1 - fr) + torch.gather(w, 1, lo + 1) * fr) * ok
    return w

def specaug(m):
    B, _, Fq, T = m.shape
    for _ in range(2):
        f = torch.randint(0, Fq // 8 + 1, (B, 1, 1, 1), device=dev); f0 = (torch.rand(B, 1, 1, 1, device=dev) * (Fq - f)).long()
        fi = torch.arange(Fq, device=dev)[None, None, :, None]
        m = m.masked_fill((fi >= f0) & (fi < f0 + f), 0.)
        t = torch.randint(0, T // 8 + 1, (B, 1, 1, 1), device=dev); t0 = (torch.rand(B, 1, 1, 1, device=dev) * (T - t)).long()
        ti = torch.arange(T, device=dev)[None, None, None, :]
        m = m.masked_fill((ti >= t0) & (ti < t0 + t), 0.)
    return m

# ---------------- model ----------------
class Block(nn.Module):
    def __init__(s, cin, cout, stride):
        super().__init__()
        s.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); s.b1 = nn.BatchNorm2d(cout)
        s.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(cout)
        s.sc = None if (cin == cout and stride == 1) else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
    def forward(s, x):
        o = F.relu(s.b1(s.c1(x))); o = s.b2(s.c2(o))
        return F.relu(o + (x if s.sc is None else s.sc(x)))

class ResNet(nn.Module):
    def __init__(s, n_cls, w):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(1, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        chans = [w, w, 2 * w, 2 * w, 4 * w, 4 * w, 8 * w, 8 * w]
        strides = [1, 1, 2, 1, 2, 1, 2, 1]
        layers, cin = [], w
        for c, st in zip(chans, strides): layers.append(Block(cin, c, st)); cin = c
        s.body = nn.Sequential(*layers)
        s.drop = nn.Dropout(0.2); s.fc = nn.Linear(cin, n_cls)
    def forward(s, x):
        x = s.body(s.stem(x))
        x = x.mean((2, 3))
        return s.fc(s.drop(x))

class TimmNet(nn.Module):
    def __init__(s, name, n_cls):
        super().__init__()
        import timm
        s.m = timm.create_model(name, pretrained=True, num_classes=n_cls, in_chans=1, drop_rate=0.2)
    def forward(s, x):
        x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=False)
        return s.m(x)

def make_model():
    if args.arch == 'resnet': return ResNet(len(WORDS), args.width)
    return TimmNet(args.arch.split(':', 1)[1], len(WORDS))

# normalization stats from a sample
with torch.no_grad():
    smp = logmel(X[::50].float() / 32768.)
    MU, SD = smp.mean().item(), smp.std().item()
def feats(w, train):
    if train: w = augment(w)
    m = (logmel(w) - MU) / SD
    if train: m = specaug(m)
    return m

@torch.no_grad()
def predict(model, Xs, bs=1024):
    model.eval(); out = []
    for i in range(0, len(Xs), bs):
        w = Xs[i:i + bs].float() / 32768.
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lo = model(feats(w, False)).float()
        out.append(lo.softmax(1))
    return torch.cat(out)

def to12(p30):  # [N,30] -> [N,11]
    p = torch.zeros(p30.shape[0], 11, device=dev)
    p.index_add_(1, w2l, p30)
    return p

def run_fold(tr_idx, va_idx, tag):
    model = make_model().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per = len(tr_idx) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per, pct_start=0.15)
    tr_idx_t = torch.from_numpy(tr_idx).to(dev)
    t0 = time.time(); best = None
    for ep in range(args.epochs):
        model.train(); perm = tr_idx_t[torch.randperm(len(tr_idx_t), device=dev)]; tl = 0
        for b in range(steps_per):
            bi = perm[b * args.bs:(b + 1) * args.bs]
            w = X[bi].float() / 32768.; yb = Y[bi]
            x = feats(w, True)
            yt = F.one_hot(yb, len(WORDS)).float() * (1 - args.ls) + args.ls / len(WORDS)
            if args.mixup > 0 and np.random.rand() < 0.5:
                lam = float(np.random.beta(args.mixup, args.mixup)); pi = torch.randperm(len(bi), device=dev)
                x = lam * x + (1 - lam) * x[pi]; yt = lam * yt + (1 - lam) * yt[pi]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lo = model(x.contiguous(memory_format=torch.channels_last))
            loss = torch.sum(-yt * F.log_softmax(lo.float(), 1), 1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        msg = f'{tag} ep{ep} loss {tl / steps_per:.4f} t {time.time() - t0:.0f}s'
        if va_idx is not None and (ep % 5 == 4 or ep == args.epochs - 1):
            pv = predict(model, X[va_idx]); yv = Y[va_idx]
            a30 = (pv.argmax(1) == yv).float().mean().item()
            a12 = (to12(pv).argmax(1) == w2l[yv]).float().mean().item()
            msg += f' val30 {a30:.4f} val12 {a12:.4f}'
        print(msg, flush=True)
    return model

res = {}
if args.folds == 'all':
    model = run_fold(np.arange(len(meta)), None, 'all')
    pt = predict(model, XT).cpu().numpy(); np.save(f'{OUT}/test_all.npy', pt)
    torch.save(model.state_dict(), f'{OUT}/model_all.pt')
else:
    gkf = GroupKFold(n_splits=args.nfolds, shuffle=True, random_state=42)
    splits = list(gkf.split(meta, y_all, meta.speaker))
    oof = np.full((len(meta), len(WORDS)), np.nan, np.float32)
    for f in [int(v) for v in args.folds.split(',')]:
        tr_idx, va_idx = splits[f]
        model = run_fold(tr_idx, va_idx, f'fold{f}')
        pv = predict(model, X[va_idx]); oof[va_idx] = pv.cpu().numpy()
        pt = predict(model, XT).cpu().numpy(); np.save(f'{OUT}/test_f{f}.npy', pt)
        yv = Y[va_idx]
        res[f] = {'val30': (pv.argmax(1) == yv).float().mean().item(), 'val12': (to12(pv).argmax(1) == w2l[yv]).float().mean().item()}
        print(f'FOLD {f}', res[f], flush=True)
    np.save(f'{OUT}/oof.npy', oof)
    json.dump({'args': vars(args), 'folds': res}, open(f'{OUT}/res.json', 'w'), indent=1)
    print('MEAN val12', np.mean([r['val12'] for r in res.values()]))
