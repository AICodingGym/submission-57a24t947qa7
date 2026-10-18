"""Build tabular features for (game_play, step, p1, p2) rows from tracking + baseline helmets.
Usage: python scripts/feats.py train|test
"""
import sys
import numpy as np
import pandas as pd

split = sys.argv[1]
D = 'data'
OUT = f'work/feats/{split}.parquet'
FPS = 59.94

# ---------------- rows ----------------
if split == 'train':
    df = pd.read_csv(f'{D}/train_labels.csv', usecols=['contact_id', 'game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2', 'contact'])
else:
    df = pd.read_csv(f'{D}/sample_submission.csv')
    parts = df.contact_id.str.split('_', expand=True)
    df['game_play'] = parts[0] + '_' + parts[1]
    df['step'] = parts[2].astype(int)
    df['nfl_player_id_1'] = parts[3]
    df['nfl_player_id_2'] = parts[4]
    df['contact'] = -1
df['is_g'] = (df.nfl_player_id_2.astype(str) == 'G').astype(np.int8)
df['p1'] = df.nfl_player_id_1.astype(int)
df['p2'] = pd.to_numeric(df.nfl_player_id_2, errors='coerce').fillna(-1).astype(int)
df = df.drop(columns=['nfl_player_id_1', 'nfl_player_id_2'])

# ---------------- tracking ----------------
tr = pd.read_csv(f'{D}/{split}_player_tracking.csv')
tr['datetime'] = pd.to_datetime(tr.datetime)
tr = tr.sort_values(['game_play', 'nfl_player_id', 'step']).reset_index(drop=True)
tr['team_home'] = (tr.team == 'home').astype(np.int8)
pos_map = {p: i for i, p in enumerate(sorted(pd.read_csv(f'{D}/train_player_tracking.csv', usecols=['position']).position.unique()))}
tr['pos_code'] = tr.position.map(pos_map).fillna(-1).astype(np.int16)
for c in ['direction', 'orientation']:
    r = np.deg2rad(tr[c])
    tr[c + '_sin'] = np.sin(r)
    tr[c + '_cos'] = np.cos(r)
tr['vx'] = tr.speed * tr.direction_sin
tr['vy'] = tr.speed * tr.direction_cos
g = tr.groupby(['game_play', 'nfl_player_id'])
base = ['speed', 'acceleration', 'sa', 'distance']
for k in [1, 2, 3, 5]:
    for c in ['speed', 'acceleration', 'sa']:
        tr[f'{c}_d{k}'] = tr[c] - g[c].shift(k)
        tr[f'{c}_f{k}'] = g[c].shift(-k) - tr[c]
for w in [5, 11]:
    for c in ['acceleration', 'sa', 'speed']:
        tr[f'{c}_rm{w}'] = g[c].transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
        tr[f'{c}_rs{w}'] = g[c].transform(lambda s: s.rolling(w, center=True, min_periods=1).std())
tr['orient_change'] = np.abs(((tr.orientation - g.orientation.shift(1)) + 180) % 360 - 180)
tr['dir_change'] = np.abs(((tr.direction - g.direction.shift(1)) + 180) % 360 - 180)
tr['dir_orient_diff'] = np.abs(((tr.direction - tr.orientation) + 180) % 360 - 180)

pcols = [c for c in tr.columns if c not in ['game_play', 'game_key', 'play_id', 'nfl_player_id', 'datetime', 'step', 'team', 'position', 'jersey_number']]
trk = tr[['game_play', 'nfl_player_id', 'step'] + pcols].copy()
for c in pcols:
    if trk[c].dtype == np.float64:
        trk[c] = trk[c].astype(np.float32)

def merge_player(df, which):
    t = trk.rename(columns={c: f'{c}_{which}' for c in pcols})
    t = t.rename(columns={'nfl_player_id': f'p{which}'})
    return df.merge(t, on=['game_play', f'p{which}', 'step'], how='left')

df = merge_player(df, 1)
df = merge_player(df, 2)

# pair features
dx = df.x_position_1 - df.x_position_2
dy = df.y_position_1 - df.y_position_2
df['dist'] = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)
df['dvx'] = (df.vx_1 - df.vx_2).astype(np.float32)
df['dvy'] = (df.vy_1 - df.vy_2).astype(np.float32)
df['rel_speed'] = np.sqrt(df.dvx ** 2 + df.dvy ** 2).astype(np.float32)
# closing speed: negative derivative of distance
df['closing'] = (-(dx * df.dvx + dy * df.dvy) / (df.dist + 1e-6)).astype(np.float32)
df['same_team'] = (df.team_home_1 == df.team_home_2).astype(np.float32)
df.loc[df.is_g == 1, 'same_team'] = np.nan
# is p2 in p1's facing direction
ang = np.arctan2(-dx, -dy)  # from p1 to p2, angle measured like direction (0 = +y)
df['face_1'] = np.cos(ang - np.deg2rad(df.orientation_1)).astype(np.float32)
df['face_2'] = np.cos(ang + np.pi - np.deg2rad(df.orientation_2)).astype(np.float32)
df['speed_sum'] = (df.speed_1 + df.speed_2).astype(np.float32)
df['acc_sum'] = (df.acceleration_1 + df.acceleration_2).astype(np.float32)
df['sa_diff'] = (df.sa_1 - df.sa_2).astype(np.float32)
df['x_rel'] = (df.x_position_1 - df.groupby(['game_play', 'step']).x_position_1.transform('mean')).astype(np.float32)

df = df.sort_values(['game_play', 'p1', 'p2', 'step']).reset_index(drop=True)
gp = df.groupby(['game_play', 'p1', 'p2'])
for k in [1, 2, 3, 5, 10]:
    df[f'dist_lag{k}'] = gp.dist.shift(k).astype(np.float32)
    df[f'dist_lead{k}'] = gp.dist.shift(-k).astype(np.float32)
for w in [5, 11, 21]:
    df[f'dist_rmin{w}'] = gp.dist.transform(lambda s: s.rolling(w, center=True, min_periods=1).min()).astype(np.float32)
    df[f'dist_rmean{w}'] = gp.dist.transform(lambda s: s.rolling(w, center=True, min_periods=1).mean()).astype(np.float32)
df['dist_min_play'] = gp.dist.transform('min').astype(np.float32)
df['step_from_min'] = (df.step - df.step.where(df.dist == df.dist_min_play).groupby([df.game_play, df.p1, df.p2]).transform('max')).astype(np.float32)

# number of nearby players for each of p1 and p2 at the step
pl = trk[['game_play', 'nfl_player_id', 'step', 'x_position', 'y_position']]
mm = pl.merge(pl, on=['game_play', 'step'], suffixes=('', '_o'))
mm = mm[mm.nfl_player_id != mm.nfl_player_id_o]
mm['d'] = np.sqrt((mm.x_position - mm.x_position_o) ** 2 + (mm.y_position - mm.y_position_o) ** 2)
nb = mm.groupby(['game_play', 'nfl_player_id', 'step']).d.agg(
    nn_min='min', n_1=lambda s: (s < 1).sum(), n_2=lambda s: (s < 2).sum()).reset_index()
del mm
for w in ['1', '2']:
    t = nb.rename(columns={'nfl_player_id': f'p{w}', 'nn_min': f'nn_min_{w}', 'n_1': f'n1_{w}', 'n_2': f'n2_{w}'})
    df = df.merge(t, on=['game_play', f'p{w}', 'step'], how='left')

# ---------------- helmets ----------------
hm = pd.read_csv(f'{D}/{split}_baseline_helmets.csv', usecols=['game_play', 'view', 'frame', 'nfl_player_id', 'left', 'width', 'top', 'height'])
vm = pd.read_csv(f'{D}/{split}_video_metadata.csv')
vm['start_time'] = pd.to_datetime(vm.start_time)
step0 = tr[tr.step == 0].groupby('game_play').datetime.first().rename('t0').reset_index()
vm = vm.merge(step0, on='game_play', how='left')
vm['off'] = (vm.start_time - vm.t0).dt.total_seconds()
hm = hm.merge(vm[['game_play', 'view', 'off']], on=['game_play', 'view'], how='inner')
hm['step'] = np.round((hm.frame / FPS + hm.off) * 10).astype(int)
hm['cx'] = hm.left + hm.width / 2
hm['cy'] = hm.top + hm.height / 2
hb = hm.groupby(['game_play', 'view', 'nfl_player_id', 'step'])[['cx', 'cy', 'width', 'height']].mean().reset_index()
hb = hb.sort_values(['game_play', 'view', 'nfl_player_id', 'step'])
hg = hb.groupby(['game_play', 'view', 'nfl_player_id'])
hb['dcy'] = hb.cy - hg.cy.shift(1)
hb['dcy_f'] = hg.cy.shift(-1) - hb.cy
hb['dh'] = hb.height - hg.height.shift(1)
for view in ['Endzone', 'Sideline']:
    v = view[0]
    hv = hb[hb.view == view].drop(columns='view')
    for w in ['1', '2']:
        t = hv.rename(columns={'nfl_player_id': f'p{w}', **{c: f'{v}_{c}_{w}' for c in ['cx', 'cy', 'width', 'height', 'dcy', 'dcy_f', 'dh']}})
        df = df.merge(t, on=['game_play', f'p{w}', 'step'], how='left')
    hmean = (df[f'{v}_height_1'] + df[f'{v}_height_2']) / 2
    df[f'{v}_bdist'] = (np.sqrt((df[f'{v}_cx_1'] - df[f'{v}_cx_2']) ** 2 + (df[f'{v}_cy_1'] - df[f'{v}_cy_2']) ** 2) / hmean).astype(np.float32)
    df[f'{v}_bdx'] = (np.abs(df[f'{v}_cx_1'] - df[f'{v}_cx_2']) / hmean).astype(np.float32)
    df[f'{v}_bdy'] = ((df[f'{v}_cy_1'] - df[f'{v}_cy_2']) / hmean).astype(np.float32)
    df[f'{v}_hratio'] = (df[f'{v}_height_1'] / df[f'{v}_height_2']).astype(np.float32)
    df[f'{v}_aspect_1'] = (df[f'{v}_width_1'] / df[f'{v}_height_1']).astype(np.float32)
    df[f'{v}_cy_rel_1'] = (df[f'{v}_cy_1'] / df[f'{v}_height_1']).astype(np.float32)
    for c in ['cx_1', 'cx_2', 'cy_2']:
        df = df.drop(columns=f'{v}_{c}')
df = df.sort_values(['game_play', 'p1', 'p2', 'step']).reset_index(drop=True)
gp = df.groupby(['game_play', 'p1', 'p2'])
for v in ['E', 'S']:
    for k in [1, 3]:
        df[f'{v}_bdist_lag{k}'] = gp[f'{v}_bdist'].shift(k).astype(np.float32)
        df[f'{v}_bdist_lead{k}'] = gp[f'{v}_bdist'].shift(-k).astype(np.float32)
    df[f'{v}_bdist_rmin5'] = gp[f'{v}_bdist'].transform(lambda s: s.rolling(5, center=True, min_periods=1).min()).astype(np.float32)
df['bdist_min'] = df[['E_bdist', 'S_bdist']].min(axis=1)
df['n_views'] = df[['E_height_1', 'S_height_1']].notna().sum(axis=1) + df[['E_height_2', 'S_height_2']].notna().sum(axis=1)

for c in df.columns:
    if df[c].dtype == np.float64:
        df[c] = df[c].astype(np.float32)
print(df.shape)
df.to_parquet(OUT)
