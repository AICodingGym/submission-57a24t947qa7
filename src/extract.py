"""Frozen-backbone feature extraction from the uint8 image cache.
Saves, per view (crop, crop_flip, squash, squash_flip): pooled pre-logit features and 120 dog-class logits
from the model's ImageNet-1k head (when it has one).
usage: python extract.py <timm_model_name> [res]"""
import os, sys, time, numpy as np, torch, torch.nn.functional as F, timm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, 'work', 'cache')
name = sys.argv[1]
res = int(sys.argv[2]) if len(sys.argv) > 2 else None
dog_idx = np.load(os.path.join(C, 'dog_idx.npy'))

kw = {}
if res is not None and ('eva' in name or 'vit' in name or 'beit' in name):
    kw['img_size'] = res
m = timm.create_model(name, pretrained=True, **kw).cuda().eval().half()
cfg = m.pretrained_cfg
if res is None:
    res = cfg['input_size'][-1]
mean = torch.tensor(cfg['mean'], device='cuda').view(1, 3, 1, 1)
std = torch.tensor(cfg['std'], device='cuda').view(1, 3, 1, 1)
has_head = m.num_classes == 1000
MISSING = None
if m.num_classes > 1000:
    # imagenet-22k style head: map the 120 dog synsets into its label space
    from timm.data import ImageNetInfo, infer_imagenet_subset
    sub = infer_imagenet_subset(m)
    i1 = ImageNetInfo('imagenet-1k')
    pos = {s: k for k, s in enumerate(ImageNetInfo(sub).label_names())}
    syn = [i1.index_to_label_name(int(i)) for i in dog_idx]
    if all(s in pos for s in syn):
        dog_idx = np.array([pos[s] for s in syn])
        has_head = True
    else:
        # e.g. imagenet-12k lacks a few breeds: those columns get the row-min dog logit (res head bias/features compensate)
        MISSING = np.array([s not in pos for s in syn])
        dog_idx = np.array([pos.get(s, 0) for s in syn])
        has_head = True
        print('missing', MISSING.sum(), flush=True)
    print('subset', sub, 'mapped', has_head, flush=True)
SAVE_FULL = os.environ.get('SAVE_FULL') == '1' and m.num_classes > 1000  # full log-softmax over all classes (fp16)
tag = name.replace('/', '_') + f'_{res}'
out = os.path.join(ROOT, 'work', 'feats', tag)
os.makedirs(out, exist_ok=True)
print(name, 'res', res, 'head', has_head, flush=True)


@torch.no_grad()
def run(arr, flip):
    feats, logits, fulls = [], [], []
    bs = 32
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(np.ascontiguousarray(arr[i:i + bs])).cuda().permute(0, 3, 1, 2).float() / 255.
        if x.shape[-1] != res:
            x = F.interpolate(x, size=(res, res), mode='bicubic', antialias=True, align_corners=False).clamp(0, 1)
        if flip:
            x = x.flip(3)
        x = ((x - mean) / std).half()
        f = m.forward_features(x)
        p = m.forward_head(f, pre_logits=True)
        feats.append(p.float().cpu())
        if has_head:
            full = m.forward_head(f).float()
            lg = full[:, dog_idx]
            if MISSING is not None:
                mk = torch.tensor(MISSING, device=lg.device)
                lg[:, mk] = lg[:, ~mk].min(1, keepdim=True).values
            logits.append(lg.cpu())
            if SAVE_FULL:
                fulls.append(torch.log_softmax(full, -1).half().cpu())
    if SAVE_FULL:
        np.save(f'{out}/_full_tmp.npy', torch.cat(fulls).numpy())
    return torch.cat(feats).numpy().astype(np.float16), (torch.cat(logits).numpy() if has_head else None)


t0 = time.time()
for split in ['test', 'train']:
    for view in ['crop', 'squash']:
        arr = np.load(os.path.join(C, f'{split}_{view}448.npy'), mmap_mode='r')
        for flip in [0, 1]:
            f, l = run(arr, flip)
            v = view + ('_flip' if flip else '')
            np.save(f'{out}/{split}_{v}_feat.npy', f)
            if l is not None:
                np.save(f'{out}/{split}_{v}_logit.npy', l)
            if SAVE_FULL:
                os.replace(f'{out}/_full_tmp.npy', f'{out}/{split}_{v}_full.npy')
            print(split, v, f.shape, f'{time.time() - t0:.0f}s', flush=True)
print('DONE', flush=True)
