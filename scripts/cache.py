import os, sys, numpy as np, pandas as pd
from scipy.io import wavfile
from multiprocessing import Pool
ROOT = sys.argv[1]; OUT = sys.argv[2]
def load(p):
    sr, x = wavfile.read(p)
    assert sr == 16000, (p, sr)
    if x.ndim > 1: x = x[:, 0]
    x = x.astype(np.int16)
    n = len(x)
    out = np.zeros(16000, np.int16)
    out[:min(n,16000)] = x[:16000]
    return out, n
if __name__ == '__main__':
    words = sorted(d for d in os.listdir(f'{ROOT}/train/audio') if not d.startswith('_'))
    rows = []
    for w in words:
        for f in sorted(os.listdir(f'{ROOT}/train/audio/{w}')):
            rows.append((w, f, f.split('_nohash_')[0]))
    tr = pd.DataFrame(rows, columns=['word','fname','speaker'])
    with Pool(4) as pool:
        res = pool.map(load, [f'{ROOT}/train/audio/{w}/{f}' for w,f,_ in rows], chunksize=256)
    tr['len'] = [r[1] for r in res]
    np.save(f'{OUT}/train_x.npy', np.stack([r[0] for r in res]))
    tr.to_csv(f'{OUT}/train_meta.csv', index=False)
    sub = pd.read_csv(f'{ROOT}/sample_submission.csv')
    with Pool(4) as pool:
        res = pool.map(load, [f'{ROOT}/test/audio/{f}' for f in sub.fname], chunksize=256)
    np.save(f'{OUT}/test_x.npy', np.stack([r[0] for r in res]))
    pd.DataFrame({'fname': sub.fname, 'len': [r[1] for r in res]}).to_csv(f'{OUT}/test_meta.csv', index=False)
    bg = []
    for f in sorted(os.listdir(f'{ROOT}/train/audio/_background_noise_')):
        if f.endswith('.wav'):
            sr, x = wavfile.read(f'{ROOT}/train/audio/_background_noise_/{f}')
            bg.append(x.astype(np.float32) / 32768.0); print(f, len(x)/sr)
    np.save(f'{OUT}/bg.npy', np.concatenate([b for b in bg]))
    np.save(f'{OUT}/bg_lens.npy', np.array([len(b) for b in bg]))
    print(tr.shape, tr.speaker.nunique(), (tr.len < 16000).mean())
