import pandas as pd, numpy as np, sys
d = sys.argv[1]; out = sys.argv[2]
for name in ['train', 'test']:
    df = pd.read_csv(f'{d}/{name}.csv')
    df = df.sort_values(['breath_id', 'time_step']).reset_index(drop=True)
    sizes = df.groupby('breath_id').size()
    assert (sizes == 80).all(), sizes.value_counts()
    n = len(sizes)
    cols = ['R', 'C', 'time_step', 'u_in', 'u_out'] + (['pressure'] if name == 'train' else [])
    arr = df[cols].values.astype(np.float32).reshape(n, 80, len(cols))
    np.save(f'{out}/{name}_arr.npy', arr)
    np.save(f'{out}/{name}_ids.npy', df['id'].values.reshape(n, 80))
    np.save(f'{out}/{name}_bid.npy', df['breath_id'].values.reshape(n, 80)[:, 0])
    print(name, arr.shape)
    if name == 'train':
        p = np.sort(df['pressure'].unique()); np.save(f'{out}/pressure_grid.npy', p); print('unique pressures', len(p), p[:3], np.diff(p)[:3])
        print(df.groupby(['R','C']).size()/80)
    else:
        print(df.groupby(['R','C']).size()/80)
