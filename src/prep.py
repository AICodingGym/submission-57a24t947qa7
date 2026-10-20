# Decode all train/test JPEGs once, resize to SxS grayscale uint8, store as .npy memmap.
import sys, os, numpy as np, pandas as pd, cv2
from multiprocessing import Pool

S = int(sys.argv[1]) if len(sys.argv) > 1 else 768
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cv2.setNumThreads(1)


def load(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)


def run(split, ids):
    out = f"{ROOT}/work/cache/{split}_{S}.npy"
    arr = np.lib.format.open_memmap(out + ".tmp", mode="w+", dtype=np.uint8, shape=(len(ids), S, S))
    paths = [f"{ROOT}/data/{split}/{i}.jpg" for i in ids]
    with Pool(16) as p:
        for k, im in enumerate(p.imap(load, paths, chunksize=16)):
            arr[k] = im
            if k % 2000 == 0:
                print(split, k, flush=True)
    arr.flush(); del arr
    os.rename(out + ".tmp", out)


if __name__ == "__main__":
    tr = pd.read_csv(f"{ROOT}/data/train.csv")
    te = pd.read_csv(f"{ROOT}/data/sample_submission.csv")
    run("test", te.StudyInstanceUID.tolist())
    run("train", tr.StudyInstanceUID.tolist())
