"""Extract grayscale video crops (3 temporal frames) around candidate pairs per view.
Usage: python scripts/extract_crops.py train|test
Outputs work/crops/{split}_crops.npy (uint8 [N,3,S,S]) and work/crops/{split}_index.parquet
"""
import sys, os, time
import numpy as np
import pandas as pd
import cv2
from multiprocessing import Pool

split = sys.argv[1]
D = 'data'
FPS = 59.94
S = int(os.environ.get('S', '64'))
BIG = int(os.environ.get('BIG', '0'))
OFFS = [int(v) for v in os.environ.get('OFFS', '-6,0,6').split(',')]  # frame offsets
PFX = os.environ.get('PFX', '')
DIST_MAX = 2.0
os.makedirs('work/crops', exist_ok=True)
cv2.setNumThreads(1)


def build_index():
    f = pd.read_parquet(f'work/feats/{split}_base.parquet' if split == 'train' else f'work/feats/{split}.parquet', columns=['contact_id', 'game_play', 'step', 'p1', 'p2', 'is_g', 'dist', 'contact'])
    f = f[(f.is_g == 1) | (f.dist <= DIST_MAX)]
    tr = pd.read_csv(f'{D}/{split}_player_tracking.csv', usecols=['game_play', 'step', 'datetime'])
    tr['datetime'] = pd.to_datetime(tr.datetime)
    step0 = tr[tr.step == 0].groupby('game_play').datetime.first().rename('t0').reset_index()
    vm = pd.read_csv(f'{D}/{split}_video_metadata.csv')
    vm['start_time'] = pd.to_datetime(vm.start_time)
    vm = vm.merge(step0, on='game_play')
    vm['off'] = (vm.start_time - vm.t0).dt.total_seconds()
    hm = pd.read_csv(f'{D}/{split}_baseline_helmets.csv', usecols=['game_play', 'view', 'frame', 'nfl_player_id', 'left', 'width', 'top', 'height'])
    hm = hm.merge(vm[['game_play', 'view', 'off']], on=['game_play', 'view'])
    hm['step'] = np.round((hm.frame / FPS + hm.off) * 10).astype(int)
    hm['cx'] = hm.left + hm.width / 2
    hm['cy'] = hm.top + hm.height / 2
    hb = hm.groupby(['game_play', 'view', 'nfl_player_id', 'step'])[['cx', 'cy', 'height']].mean().reset_index()
    out = []
    for view in ['Endzone', 'Sideline']:
        hv = hb[hb.view == view].drop(columns='view')
        x = f.merge(hv.rename(columns={'nfl_player_id': 'p1', 'cx': 'cx1', 'cy': 'cy1', 'height': 'h1'}), on=['game_play', 'p1', 'step'])
        x = x.merge(hv.rename(columns={'nfl_player_id': 'p2', 'cx': 'cx2', 'cy': 'cy2', 'height': 'h2'}), on=['game_play', 'p2', 'step'], how='left')
        x = x[(x.is_g == 1) | x.cx2.notna()].copy()
        g = x.is_g == 1
        h = np.where(g, x.h1, (x.h1 + x.h2.fillna(x.h1)) / 2)
        mx = np.where(g, x.cx1, (x.cx1 + x.cx2) / 2)
        gy, gs, pm = (2.5, 8, 5) if BIG else (2.0, 6, 4)
        my = np.where(g, x.cy1 + gy * x.h1, (x.cy1 + x.cy2) / 2 + 1.0 * h)
        span = np.where(g, 0, np.maximum(np.abs(x.cx1 - x.cx2), np.abs(x.cy1 - x.cy2)))
        size = np.maximum(span + pm * h, np.where(g, gs * h, 6 * h))
        x['x0'] = (mx - size / 2).astype(np.float32)
        x['y0'] = (my - size / 2).astype(np.float32)
        x['size'] = size.astype(np.float32)
        x['view'] = view
        x = x.merge(vm[vm.view == view][['game_play', 'off']], on='game_play')
        x['frame'] = np.round((x.step / 10 - x.off) * FPS).astype(int)
        out.append(x[['contact_id', 'game_play', 'view', 'step', 'frame', 'x0', 'y0', 'size', 'is_g', 'contact', 'cx1', 'cy1', 'h1', 'cx2', 'cy2', 'h2']])
    idx = pd.concat(out).sort_values(['game_play', 'view', 'frame']).reset_index(drop=True)
    return idx


def work(args):
    gp, view, sub_idx = args
    sub = IDX.loc[sub_idx]
    need = {}
    for i, fr in zip(sub_idx, sub.frame.values):
        for k, o in enumerate(OFFS):
            need.setdefault(fr + o, []).append((i, k))
    cap = cv2.VideoCapture(f'{D}/{split}/{gp}_{view}.mp4')
    fmax = max(need)
    res = {}
    fi = 0
    x0s = dict(zip(sub_idx, sub.x0.values)); y0s = dict(zip(sub_idx, sub.y0.values)); szs = dict(zip(sub_idx, sub['size'].values))
    while fi <= fmax:
        ok, img = cap.read()
        if not ok:
            break
        fi += 1  # helmet frames are 1-indexed
        if fi in need:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            Hh, Ww = gray.shape
            pad = cv2.copyMakeBorder(gray, 400, 400, 400, 400, cv2.BORDER_CONSTANT, value=0)
            for i, k in need[fi]:
                sz = int(min(szs[i], 790))
                x0 = int(x0s[i]) + 400; y0 = int(y0s[i]) + 400
                x0 = min(max(x0, 0), Ww + 800 - sz); y0 = min(max(y0, 0), Hh + 800 - sz)
                crop = pad[y0:y0 + sz, x0:x0 + sz]
                if i not in res:
                    res[i] = np.zeros((len(OFFS), S, S), np.uint8)
                res[i][k] = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA)
    cap.release()
    ids = np.array(list(res.keys()), dtype=np.int64)
    arr = np.stack([res[i] for i in ids]) if len(ids) else np.zeros((0, len(OFFS), S, S), np.uint8)
    return ids, arr


if __name__ == '__main__':
    t0 = time.time()
    IDX = build_index()
    print('index rows', len(IDX), IDX.groupby('view').size().to_dict(), 'pos', (IDX.contact == 1).sum(), flush=True)
    IDX.to_parquet(f'work/crops/{PFX}{split}_index.parquet')
    out = np.lib.format.open_memmap(f'work/crops/{PFX}{split}_crops.npy', mode='w+', dtype=np.uint8, shape=(len(IDX), len(OFFS), S, S))
    tasks = [(gp, v, g.index.values) for (gp, v), g in IDX.groupby(['game_play', 'view'])]
    done = 0
    with Pool(4) as p:
        for ids, arr in p.imap_unordered(work, tasks):
            if len(ids):
                out[ids] = arr
            done += 1
            if done % 40 == 0:
                print(done, len(tasks), f'{time.time() - t0:.0f}s', flush=True)
    out.flush()
    print('done', time.time() - t0)
