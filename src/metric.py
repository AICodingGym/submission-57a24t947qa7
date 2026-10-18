# Competition-style score proxy: 0.4 * mean 2D dice (per slice-class, empty/empty=1) + 0.6 * (1 - normalized 3D Hausdorff per day-class)
import numpy as np
from scipy.spatial import cKDTree


def dice2d(pr, gt):
    # pr, gt: bool (N,3,H,W) -> (N,3)
    inter = (pr & gt).sum((2, 3)); den = pr.sum((2, 3)) + gt.sum((2, 3))
    return np.where(den == 0, 1.0, 2 * inter / np.maximum(den, 1))


def hd3d(pr, gt, ds=2):
    # pr, gt: bool (D,H,W) for one day-class; returns normalized HD in [0,1]
    pr = pr[:, ::ds, ::ds]; gt = gt[:, ::ds, ::ds]
    if not pr.any() and not gt.any():
        return 0.0
    if not pr.any() or not gt.any():
        return 1.0
    sh = np.array(pr.shape, float)
    a = np.argwhere(pr) / sh; b = np.argwhere(gt) / sh
    # points of a inside b have distance 0; only check the symmetric difference against the other set
    pa = np.argwhere(pr & ~gt) / sh; pb = np.argwhere(gt & ~pr) / sh
    h = 0.0
    if len(pa): h = max(h, cKDTree(b).query(pa, workers=2)[0].max())
    if len(pb): h = max(h, cKDTree(a).query(pb, workers=2)[0].max())
    return min(h / np.sqrt(3), 1.0)


def score(pr, gt, days, ds=2):
    # pr, gt: bool (N,3,H,W) ordered by day/slice; days: (N,) day labels
    d = dice2d(pr, gt)
    hds = []
    for day in np.unique(days):
        ix = np.where(days == day)[0]
        for c in range(3):
            hds.append(hd3d(pr[ix, c], gt[ix, c], ds))
    hd = float(np.mean(hds))
    return dict(score=0.4 * d.mean() + 0.6 * (1 - hd), dice=float(d.mean()), hd=hd,
                dice_cls=d.mean(0).round(4).tolist())
