"""Preprocess CSVs into dense per-molecule arrays (saved to work/prep.pt)."""
import numpy as np
import pandas as pd
import torch

D = 'data/'
TYPES = ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']
ELEM = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5}
NMAX = 29

# data/structures.csv only covers train molecules; this file = same coords for train + QM9 coords for test
s = pd.read_csv('work/qm9_structures.csv')
mols = s.molecule_name.unique()
mid = {m: i for i, m in enumerate(mols)}
M = len(mols)
s['m'] = s.molecule_name.map(mid).values
Z = np.zeros((M, NMAX), np.int64)
P = np.zeros((M, NMAX, 3), np.float32)
Z[s.m.values, s.atom_index.values] = s.atom.map(ELEM).values
P[s.m.values, s.atom_index.values] = s[['x', 'y', 'z']].values
nat = (Z > 0).sum(1)

# atom-level aux targets (train molecules only)
mull = np.full((M, NMAX), np.nan, np.float32)
mc = pd.read_csv(D + 'mulliken_charges.csv')
mull[mc.molecule_name.map(mid).values, mc.atom_index.values] = mc.mulliken_charge.values
shield = np.full((M, NMAX, 4), np.nan, np.float32)
ms = pd.read_csv(D + 'magnetic_shielding_tensors.csv')
T = ms[['XX', 'YX', 'ZX', 'XY', 'YY', 'ZY', 'XZ', 'YZ', 'ZZ']].values.reshape(-1, 3, 3)
Ts = 0.5 * (T + T.transpose(0, 2, 1))
ev = np.linalg.eigvalsh(Ts)  # sorted rotation-invariant eigenvalues
iso = np.trace(T, axis1=1, axis2=2) / 3
shield[ms.molecule_name.map(mid).values, ms.atom_index.values] = np.c_[iso, ev]

tr = pd.read_csv(D + 'train.csv')
te = pd.read_csv(D + 'test.csv')
sc = pd.read_csv(D + 'scalar_coupling_contributions.csv')
tr = tr.merge(sc[['molecule_name', 'atom_index_0', 'atom_index_1', 'fc', 'sd', 'pso', 'dso']],
              on=['molecule_name', 'atom_index_0', 'atom_index_1'], how='left')
assert tr.fc.notna().all()
te['scalar_coupling_constant'] = np.nan
for c in ['fc', 'sd', 'pso', 'dso']:
    te[c] = np.nan
tr['is_test'] = 0
te['is_test'] = 1
df = pd.concat([tr, te], ignore_index=True)
tmap = {t: i for i, t in enumerate(TYPES)}
out = dict(
    mols=list(mols), Z=torch.from_numpy(Z), P=torch.from_numpy(P), nat=torch.from_numpy(nat),
    mull=torch.from_numpy(mull), shield=torch.from_numpy(shield),
    pm=torch.from_numpy(df.molecule_name.map(mid).values.astype(np.int64)),
    a0=torch.from_numpy(df.atom_index_0.values.astype(np.int64)),
    a1=torch.from_numpy(df.atom_index_1.values.astype(np.int64)),
    ptype=torch.from_numpy(df.type.map(tmap).values.astype(np.int64)),
    y=torch.from_numpy(df[['scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso']].values.astype(np.float32)),
    pid=torch.from_numpy(df.id.values.astype(np.int64)),
    is_test=torch.from_numpy(df.is_test.values.astype(np.int64)),
    types=TYPES,
)
torch.save(out, 'work/prep.pt')
print(M, len(tr), len(te), np.bincount(nat).nonzero())
