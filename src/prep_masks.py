# Rasterise train_annotations polylines into 4-channel (ETT, NGT, CVC, SGC) masks of size MxM,
# aligned with the SxS resized cache (which ignores aspect ratio). Stored for annotated studies only.
import os, ast, sys, numpy as np, pandas as pd, cv2
from PIL import Image

M = int(sys.argv[1]) if len(sys.argv) > 1 else 128
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(f"{ROOT}/data/train.csv")
an = pd.read_csv(f"{ROOT}/data/train_annotations.csv")
uids = tr.StudyInstanceUID[tr.StudyInstanceUID.isin(set(an.StudyInstanceUID))].tolist()
row = {u: k for k, u in enumerate(uids)}
TYPES = ["ETT", "NGT", "CVC", "Swan"]
masks = np.zeros((len(uids), 4, M, M), np.uint8)
sizes = {}
for u in uids:
    with Image.open(f"{ROOT}/data/train/{u}.jpg") as im:
        sizes[u] = im.size  # (W, H)
for u, lab, data in an[["StudyInstanceUID", "label", "data"]].itertuples(index=False):
    W, H = sizes[u]
    pts = np.array(ast.literal_eval(data), np.float32).reshape(-1, 2)
    pts[:, 0] *= M / W; pts[:, 1] *= M / H
    ch = [i for i, t in enumerate(TYPES) if lab.startswith(t)][0]
    m = np.ascontiguousarray(masks[row[u], ch])
    cv2.polylines(m, [np.round(pts * 4).astype(np.int32)], False, 1, thickness=max(1, M // 64), shift=2)
    masks[row[u], ch] = m
idx = tr.index[tr.StudyInstanceUID.isin(row)].to_numpy()
np.save(f"{ROOT}/work/cache/masks_{M}.npy", masks)
np.save(f"{ROOT}/work/cache/masks_{M}_idx.npy", idx)
print(masks.shape, masks.mean((0, 2, 3)))
