"""Extract mean-pooled patch tokens from intermediate ViT blocks (multi-layer features for the residual head).
Saved into the same feature dir as <split>_<view>_mid.npy (N, n_layers*D) fp16.
usage: python extract_mid.py <timm_model_name> [res]"""
import os, sys, time, numpy as np, torch, torch.nn.functional as F, timm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, 'work', 'cache')
name = sys.argv[1]
res = int(sys.argv[2]) if len(sys.argv) > 2 else None
kw = {'img_size': res} if res else {}
m = timm.create_model(name, pretrained=True, **kw).cuda().eval().half()
cfg = m.pretrained_cfg
res = res or cfg['input_size'][-1]
mean = torch.tensor(cfg['mean'], device='cuda').view(1, 3, 1, 1)
std = torch.tensor(cfg['std'], device='cuda').view(1, 3, 1, 1)
nb = len(m.blocks)
LAYERS = [nb - 2, nb - 5, nb - 9]
out = os.path.join(ROOT, 'work', 'feats', name + f'_{res}')
print(name, res, 'layers', LAYERS, flush=True)


@torch.no_grad()
def run(arr, flip):
    feats = []
    for i in range(0, len(arr), 32):
        x = torch.from_numpy(np.ascontiguousarray(arr[i:i + 32])).cuda().permute(0, 3, 1, 2).float() / 255.
        if x.shape[-1] != res:
            x = F.interpolate(x, size=(res, res), mode='bicubic', antialias=True, align_corners=False).clamp(0, 1)
        if flip:
            x = x.flip(3)
        x = ((x - mean) / std).half()
        inter = m.forward_intermediates(x, indices=LAYERS, intermediates_only=True, output_fmt='NLC')
        feats.append(torch.cat([t[:, m.num_prefix_tokens:].mean(1) if t.dim() == 3 else t.mean((2, 3)) for t in inter], -1).float().cpu())
    return torch.cat(feats).numpy().astype(np.float16)


t0 = time.time()
for split in ['test', 'train']:
    for view in ['crop', 'squash']:
        arr = np.load(os.path.join(C, f'{split}_{view}448.npy'), mmap_mode='r')
        for flip in [0, 1]:
            f = run(arr, flip)
            v = view + ('_flip' if flip else '')
            np.save(f'{out}/{split}_{v}_mid.npy', f)
            print(split, v, f.shape, f'{time.time() - t0:.0f}s', flush=True)
print('DONE', flush=True)
