# Statoil iceberg classifier — report

**Metric:** log loss (lower is better). **Validation:** 5-fold stratified OOF on the 1283 training rows
(fixed folds, seed 42), with stacker robustness re-checked on 2 extra fold seeds.
Hidden test = 321 rows. **Final OOF log loss: 0.1011** (default folds); the mean over 3 stacker fold seeds is 0.1037.
For reference: feature-only LightGBM 0.206, best single image-only CNN ≈0.19, mean of CNNs ≈0.17.

## Approach
1. **CNNs (image-only).** 5-fold models on 75×75 inputs with 3 channels (HH, HV, mean of the two), normalised with
   global statistics. Training used dihedral + shift augmentation, AdamW with OneCycle (40–60 epochs) and 8× dihedral
   TTA. Each run produces OOF logits for train and logits for test.
   - Backbones: a small custom 4-block CNN plus ImageNet-pretrained timm models (resnet18/26d/34/50d, convnext_nano,
     regnety_008, densenet121, efficientnet_b0). resnet26d (75 or 100 px) and convnext_nano worked best.
2. **Stacker.** LightGBM (3 seeds) and logistic regression, averaged. Inputs:
   - the mean CNN logit
   - simple band statistics (max, mean, std, p50/p90, bright area, centre mean)
   - `inc_angle` and a flag for missing angle (missing ⇒ always a ship in train)
   - **incidence-angle label evidence:** counts of iceberg and ship labels among *other* labelled rows with exactly the
     same angle, and among labelled rows within ±0.02° and ±0.05° (exact matches excluded). These counts are
     computed strictly out-of-fold: validation rows only see training-fold labels, and training rows use
     leave-one-out. For the test set they use all training labels, and the final stacker is refit on all of train.
3. **Pseudo-labelling of the test set (transductive).** Each fold-k stacker produces soft labels for the test
   images. CNNs are retrained with train-fold images plus those soft-labelled test images. To keep test logits
   comparable to OOF logits, the test set is split into 5 parts. Fold k trains on test parts ≠ k and predicts only
   part k, so no CNN predicts a test image it was trained on. I ran 5 rounds, each using labels from the latest
   stacker. The final stack uses the mean logit of 45 pseudo-label CNNs from rounds 1–5. A sixth round added 8 more
   models but gave no gain (3-seed mean 0.1040 vs 0.1037), so it is not used.

Code is in `src/`: `prep.py`, `folds.py`, `cnn.py`, `stack.py` and `write_sub.py`. Each run's settings and score are in
`results.md`, with training logs in `work/logs/`.

## What worked
- Angle structure. Exact `inc_angle` groups are ~98% label-pure, and 56% of test rows (164/293 with an angle) have
  an exact match in train. Nearby angles (±0.02–0.05°) also share labels. These features took the stack from
  0.149 to 0.116. Wider windows were worse, and the 0.02/0.05 choice held on 3 fold seeds.
- CNN diversity (many seeds and backbones): mean logit improved from 0.211 (4 models) to 0.193.
- Pseudo-labelling: single models improved by about 0.01–0.02 (e.g. resnet26d 0.214 → 0.19). The stack went
  from 0.110 to 0.101–0.103.

## What didn't help
- kNN "same scene" feature on (angle, background level).
- Nearest distinct-angle-group features.
- Feeding every CNN column into LightGBM; only the mean worked better.
- Learned CNN weights, and separate means for the custom and pretrained CNNs.
- A CNN×evidence interaction term.
- Clipping predictions at more than 0.001.
- Stacker hyper-parameter changes, all within ±0.0005.
- efficientnet_b0 and densenet121 were weak, and upsampling to 150 px gave no gain.

## Risks: validation vs test
- Noise. The OOF estimate comes from about 1.3k rows. Changing the stacker's fold seed moves it by about ±0.004, and
  several choices (window sizes, which CNN set to use) were made on this OOF, so expect mild optimism.
- Pseudo-labelling leak. Fold-k soft labels come from a stacker trained without fold-k labels. But that stacker uses
  CNN test logits averaged over all 5 fold models, so fold-k labels can reach the val-fold CNNs indirectly, and this
  compounds over rounds. The pseudo-label OOF gain may therefore be slightly optimistic. Test predictions themselves
  have no such leak.
- Angle features. These assume the test rows come from the same scenes as train. That holds here, since test is a
  random split of the original labelled data. Test rows also have more exact-angle matches (all of train is
  labelled) than OOF rows did, which should help on test.

## With more time
- Do the pseudo-labelling cleanly: keep per-fold CNN test logits so the fold-k soft labels never depend on fold-k
  labels.
- Stronger and more varied backbones (higher resolution, longer schedules, EMA/SWA) and per-image
  normalisation variants.
- Smoother angle-neighbour modelling (kernel-weighted label density), and a transductive treatment of test↔test
  angle matches.
