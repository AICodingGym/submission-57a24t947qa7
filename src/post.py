# Post-processing shared by tune.py / infer.py. Operates on uint8 prob volumes (N,3,S,S) ordered by day/slice.
import numpy as np
from scipy import ndimage


def apply(P, days, thr, min_area, cc_frac=(0, 0, 0)):
    pr = P > (np.array(thr) * 255).reshape(1, 3, 1, 1)
    if any(min_area):
        area = pr.sum((2, 3))
        pr &= ~(area < np.array(min_area))[..., None, None]
    if any(cc_frac):
        for day in np.unique(days):
            ix = np.where(days == day)[0]
            for c in range(3):
                if not cc_frac[c]:
                    continue
                v = pr[ix, c]
                lab, n = ndimage.label(v)
                if n <= 1:
                    continue
                sz = np.bincount(lab.ravel())[1:]
                drop = np.where(sz < cc_frac[c] * sz.max())[0] + 1
                if len(drop):
                    v[np.isin(lab, drop)] = False
                    pr[ix, c] = v
    return pr
