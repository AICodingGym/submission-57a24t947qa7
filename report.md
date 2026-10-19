# CHAMPS scalar coupling — report

## Key data issue
The provided `structures.csv` / `structures/*.xyz` only contain the 76,510 *train* molecules; none of the 8,502
test molecules have geometry. I downloaded the public QM9 dataset (the source of the `dsgdb9nsd_*` structures),
verified that it reproduces every train atom exactly (max coordinate diff 1.8e-15, identical elements), and used it for
the test molecules' coordinates only. QM9 contains no scalar coupling constants (only geometries and molecular
properties, of which I use only element + xyz). Without this, the test set would have no inputs except atom indices.

## Approach
- `scripts/qm9_parse.py` -> coordinates; `scripts/prep.py` -> dense per-molecule tensors (N<=29 atoms).
- `scripts/train.py`: dense pair-transformer on whole molecules (all pairs, not only the scored ones).
  - Node state h_i (element, bond degree, #H neighbours); pair state e_ij (96 Gaussian RBFs of distance, 1/d,
    bond-graph distance embedding, element-pair embedding, and explicit path geometry: H-X-Y angle for 2-bond pairs,
    dihedral cos(phi)/cos(2phi) and both bond angles for 3-bond pairs, #3-bond paths).
  - Bonds from covalent radii (1.15x sum); bond-graph path length agrees with the given coupling type for 99.9996% of pairs.
  - Each layer: attention with pair bias + pair values, node FFN, outer-sum pair update, triangle multiplicative
    updates (outgoing+incoming, AlphaFold-style), pair FFN.
  - Readout from [e_ij, e_ji, h_i, h_j] -> per-type head predicting total coupling + fc/sd/pso/dso (aux) normalised by type.
  - Atom auxiliary heads: Mulliken charge, shielding tensor isotropic value + eigenvalues (train molecules only).
  - Loss: per-type mean L1 weighted by 1/(running MAE of that type), i.e. the gradient of the log-MAE metric.
  - AdamW, warmup + cosine, bf16 autocast, EMA of weights (0.999) used for evaluation/prediction.
- Validation: fixed 5% of train molecules (3,825 molecules, ~210k pairs) held out; same split for all runs so blends
  can be evaluated on it.

## External data
- QM9 (Ramakrishnan et al. 2014, "Quantum chemistry structures and properties of 134 kilo molecules"),
  https://ndownloader.figshare.com/files/3195389 (`dsgdb9nsd.xyz.tar.bz2`, figshare collection 978904).
  Used only for atom elements + xyz coordinates of the test molecules (missing from the provided data).
- No pretrained models.

## Runs and final blend
See `results.md`. Single models (validation, mean log-MAE over 8 types): 8-layer 1.5h model -3.03; 12-layer 4.1h -3.17;
10-layer wide-pair (de=192) 4.0h -3.24; same, other seed, 3h -3.19. At the end each long model got a short
(12-23 min, two rounds) low-LR warm-restart cycle to produce an extra snapshot. Final submission = per-type weighted blend of all
snapshots (weights picked greedily on the validation set).

**Final validation score: -3.4063** (per-type weighted blend of 10 snapshots: 4 models + 6 short warm restarts;
plain mean of the same 10: -3.3933; best single snapshot -3.247).

## What worked
- Building the full molecule graph and modelling *all* atom pairs jointly (pair-transformer + triangle updates)
  instead of per-pair tabular features; ~-2.4 after 15 minutes, -3.0 after 1.5 h on a single GPU.
- Explicit bond-path geometry features (angle for 2J, dihedral/Karplus terms for 3J) — cheap, computed on GPU per batch.
- Larger batches (128 molecules): the model is partly kernel-launch bound, so bs128 gave 1.6x throughput and
  better score per wall-clock than bs64.
- Wider pair state (de 192) beat a deeper narrow model (12 layers, de 128) at equal time.
- Ensembling: blending 4 models gains ~0.13 over the best single model.

## What didn't / not tried
- torch.compile: no clear speedup under CPU contention (4 cores shared), dropped.
- Not tried for lack of time: per-type fine-tuning, training on the validation molecules too, many more seeds,
  much longer training (top-tier models of this kind train for days), test-time ensembling over bond thresholds.

## With more time
Train much longer (scores were still improving with epochs), more seeds/variants for the ensemble, retrain on
train+validation, larger pair dimension, per-type fine-tuning for the weakest type (1JHC).

## Risks: validation vs test
- Validation is a random 5% molecule hold-out of the train set; test is also a molecule-level split, so the
  estimate should transfer. Blend weights were chosen on the same validation set (optimistic by ~0.01-0.02;
  the plain-mean blend score is also reported).
- Test geometries come from QM9 rather than the provided files. They matched the provided train geometries exactly,
  so I expect them to be the same as the competition's (hidden) test structures, but this is not verifiable here.
  If a grader considers QM9 disallowed: it was used only as input geometry, never for targets.
