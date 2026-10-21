"""Decode all train/test JPEGs once into uint8 memmaps (N,S,S,3) so GPU jobs don't bottleneck on the 4 CPUs.
Two views: 'crop' = center square crop of shorter side; 'squash' = full image resized to SxS (aspect not kept)."""
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
from PIL import Image

S = 448
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'work', 'cache')


def load(path):
    im = Image.open(path)
    im.draft('RGB', (S, S))
    im = im.convert('RGB')
    w, h = im.size
    m = min(w, h)
    l, t = (w - m) // 2, (h - m) // 2
    crop = im.crop((l, t, l + m, t + m)).resize((S, S), Image.BICUBIC)
    squash = im.resize((S, S), Image.BICUBIC)
    return np.asarray(crop), np.asarray(squash)


def run(split, ids):
    paths = [os.path.join(ROOT, 'data', split, i + '.jpg') for i in ids]
    n = len(paths)
    mc = np.lib.format.open_memmap(f'{OUT}/{split}_crop{S}.npy', 'w+', np.uint8, (n, S, S, 3))
    ms = np.lib.format.open_memmap(f'{OUT}/{split}_squash{S}.npy', 'w+', np.uint8, (n, S, S, 3))
    with Pool(4) as p:
        for k, (c, s) in enumerate(p.imap(load, paths, chunksize=32)):
            mc[k] = c
            ms[k] = s
    mc.flush(); ms.flush()
    print(split, n, flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(os.path.join(ROOT, 'data', 'labels.csv'))
    sub = pd.read_csv(os.path.join(ROOT, 'data', 'sample_submission.csv'))
    lab.to_csv(f'{OUT}/train_ids.csv', index=False)
    sub[['id']].to_csv(f'{OUT}/test_ids.csv', index=False)
    run('test', sub.id.tolist())
    run('train', lab.id.tolist())
