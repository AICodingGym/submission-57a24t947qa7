"""Parse QM9 xyz files for all train+test molecules into work/qm9_structures.csv,
and check that train molecules match the competition's structures.csv."""
import sys
import numpy as np
import pandas as pd

xyzdir = sys.argv[1]
tr = pd.read_csv('data/train.csv', usecols=['molecule_name']).molecule_name.unique()
te = pd.read_csv('data/test.csv', usecols=['molecule_name']).molecule_name.unique()
names = np.concatenate([tr, te])
rows = []
for m in names:
    with open(f'{xyzdir}/{m}.xyz') as f:
        lines = f.read().splitlines()
    n = int(lines[0])
    for i in range(n):
        p = lines[2 + i].split()
        xyz = [float(v.replace('*^', 'e')) for v in p[1:4]]
        rows.append((m, i, p[0], *xyz))
q = pd.DataFrame(rows, columns=['molecule_name', 'atom_index', 'atom', 'x', 'y', 'z'])
q.to_csv('work/qm9_structures.csv', index=False)

s = pd.read_csv('data/structures.csv')
c = s.merge(q, on=['molecule_name', 'atom_index'], how='left', suffixes=('', '_q'))
print('train atoms', len(s), 'matched', c.x_q.notna().sum(), 'atom equal', (c.atom == c.atom_q).mean())
d = np.abs(c[['x', 'y', 'z']].values - c[['x_q', 'y_q', 'z_q']].values)
print('max abs coord diff', np.nanmax(d), 'mean', np.nanmean(d))
print('test molecules', len(te), 'test atoms', q.molecule_name.isin(te).sum())
