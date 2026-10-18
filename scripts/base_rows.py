"""Minimal rows (contact_id, game_play, step, p1, p2, is_g, dist, contact) for crop extraction."""
import sys, numpy as np, pandas as pd
split = sys.argv[1]
df = pd.read_csv('data/train_labels.csv', usecols=['contact_id', 'game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2', 'contact'])
df['is_g'] = (df.nfl_player_id_2 == 'G').astype(np.int8)
df['p1'] = df.nfl_player_id_1.astype(int)
df['p2'] = pd.to_numeric(df.nfl_player_id_2, errors='coerce').fillna(-1).astype(int)
tr = pd.read_csv('data/train_player_tracking.csv', usecols=['game_play', 'nfl_player_id', 'step', 'x_position', 'y_position'])
for w in '12':
    df = df.merge(tr.rename(columns={'nfl_player_id': 'p' + w, 'x_position': 'x' + w, 'y_position': 'y' + w}), on=['game_play', 'p' + w, 'step'], how='left')
df['dist'] = np.sqrt((df.x1 - df.x2) ** 2 + (df.y1 - df.y2) ** 2)
df[['contact_id', 'game_play', 'step', 'p1', 'p2', 'is_g', 'dist', 'contact']].to_parquet('work/feats/train_base.parquet')
