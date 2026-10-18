# Ensemble inference on test cache -> submission.csv
import argparse, os, json, numpy as np, pandas as pd, torch, torch.nn.functional as F, cv2
import segmentation_models_pytorch as smp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = f'{ROOT}/work/cache'
CLASSES = ['large_bowel', 'small_bowel', 'stomach']
p = argparse.ArgumentParser()
p.add_argument('--runs', required=True)              # comma separated run names
p.add_argument('--post', default='')                 # json with thr / min_area
p.add_argument('--out', default=f'{ROOT}/submission.csv')
p.add_argument('--dev', type=int, default=0)
a = p.parse_args()
dev = torch.device(f'cuda:{a.dev}')

meta = pd.read_csv(f'{C}/test_meta.csv')
N = len(meta)
start = meta.groupby('day').cumcount().values; vstart = np.arange(N) - start
vlen = meta.groupby('day').slice.transform('size').values
X = torch.from_numpy(np.load(f'{C}/test_img.npy')).to(dev)
post = json.load(open(a.post)) if a.post else dict(thr=[0.5] * 3, min_area=[0] * 3)
post.setdefault('cc_frac', [0, 0, 0])

prob = torch.zeros(N, 3, X.shape[-1], X.shape[-1], dtype=torch.float16, device=dev)
runs = a.runs.split(',')
for r in runs:
    cfg = json.load(open(f'{ROOT}/work/runs/{r}/log.json'))
    offs = [int(o) for o in cfg['offsets'].split(',')]
    nb = torch.from_numpy(np.stack([np.clip(np.arange(N) + o, vstart, vstart + vlen - 1) for o in offs], 1)).to(dev)
    m = getattr(smp, cfg['arch'])(cfg['enc'], encoder_weights=None, in_channels=len(offs), classes=3)
    m.load_state_dict(torch.load(f'{ROOT}/work/runs/{r}/model.pt', map_location='cpu'))
    m = m.to(dev).eval().to(memory_format=torch.channels_last)
    with torch.no_grad():
        for i in range(0, N, 64):
            x = X[nb[i:i + 64]].float().contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prob[i:i + 64] += (m(x).float().sigmoid() / len(runs)).half()
    print('done', r, flush=True)


def rle_encode(mk):
    px = np.concatenate([[0], mk.flatten(), [0]])
    runs = np.where(px[1:] != px[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return ' '.join(map(str, runs))


from post import apply as post_apply
P8 = np.concatenate([(prob[i:i + 256].float() * 255).round().byte().cpu().numpy() for i in range(0, N, 256)])
np.save(f'{ROOT}/work/test_prob.npy', P8)
keep = post_apply(P8, meta.day.values, post['thr'], post['min_area'], post['cc_frac'])
thr = np.array(post['thr'])
rows = []
for i, r in meta.iterrows():
    H, W = int(r.H), int(r.W)
    pr = F.interpolate(prob[i].float()[None], size=(H, W), mode='bilinear', align_corners=False)[0].cpu().numpy()
    for c, cl in enumerate(CLASSES):
        if not keep[i, c].any():
            rows.append((r.id, cl, '')); continue
        k = cv2.resize(keep[i, c].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0
        mk = ((pr[c] > thr[c]) & k).astype(np.uint8)
        rows.append((r.id, cl, rle_encode(mk) if mk.any() else ''))
sub = pd.DataFrame(rows, columns=['id', 'class', 'predicted'])
ss = pd.read_csv(f'{ROOT}/data/sample_submission.csv')[['id', 'class']]
sub = ss.merge(sub, on=['id', 'class'], how='left')
sub['predicted'] = sub.predicted.fillna('')
sub.to_csv(a.out, index=False)
print('wrote', a.out, len(sub), 'nonempty', (sub.predicted != '').mean())
