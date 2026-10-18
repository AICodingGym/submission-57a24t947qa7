# Preprocess: read all slices, per-volume normalize, resize to SxS, store as float16 memmap + packed masks.
import glob, os, sys, numpy as np, pandas as pd, cv2
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f'{ROOT}/data'; OUT = f'{ROOT}/work/cache'
S = 384
CLASSES = ['large_bowel', 'small_bowel', 'stomach']


def rle_decode(rle, shape):
    m = np.zeros(shape[0] * shape[1], np.uint8)
    if isinstance(rle, str) and rle:
        s = np.asarray(rle.split(), int)
        for st, ln in zip(s[0::2] - 1, s[1::2]):
            m[st:st + ln] = 1
    return m.reshape(shape)


def meta(split):
    fs = sorted(glob.glob(f'{D}/{split}/*/*/scans/*.png'))
    rows = []
    for f in fs:
        p = f.split('/'); day = p[-3]; b = p[-1].split('_')
        rows.append(dict(id=f'{day}_slice_{b[1]}', case=day.split('_')[0], day=day, slice=int(b[1]),
                         w=int(b[2]), h=int(b[3]), path=f))
    return pd.DataFrame(rows).sort_values(['day', 'slice']).reset_index(drop=True)


def load_vol(args):
    paths, = args
    imgs = [cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) for p in paths]
    v = np.stack(imgs)
    hi = np.percentile(v, 99.8) + 1e-3
    out = np.stack([cv2.resize(np.clip(i / hi, 0, 1.5), (S, S), interpolation=cv2.INTER_LINEAR) for i in v])
    return out.astype(np.float16), imgs[0].shape


def main(split):
    m = meta(split)
    m['vidx'] = m.groupby('day').cumcount()
    days = list(m.day.unique())
    X = np.lib.format.open_memmap(f'{OUT}/{split}_img.npy', 'w+', np.float16, (len(m), S, S))
    with Pool(4) as pool:
        res = pool.imap(load_vol, [(m[m.day == d].path.tolist(),) for d in days])
        for d, (v, shp) in zip(days, res):
            idx = np.where(m.day == d)[0]
            X[idx[0]:idx[-1] + 1] = v
            m.loc[idx, 'H'] = shp[0]; m.loc[idx, 'W'] = shp[1]
    X.flush()
    if split == 'train':
        lab = pd.read_csv(f'{D}/train.csv').pivot(index='id', columns='class', values='segmentation')
        Y = np.lib.format.open_memmap(f'{OUT}/train_mask.npy', 'w+', np.uint8, (len(m), S, S))
        for i, r in m.iterrows():
            pk = np.zeros((S, S), np.uint8)
            for c, cl in enumerate(CLASSES):
                mk = rle_decode(lab.loc[r.id, cl], (int(r.H), int(r.W)))
                pk |= cv2.resize(mk, (S, S), interpolation=cv2.INTER_NEAREST) << c
                m.loc[i, cl] = mk.sum()
            Y[i] = pk
        Y.flush()
    m.to_csv(f'{OUT}/{split}_meta.csv', index=False)
    print(split, len(m), m[['H', 'W']].value_counts())


if __name__ == '__main__':
    for s in sys.argv[1:]:
        main(s)
