# Experiments
Score = 0.4*dice + 0.6*(1-HD3D) proxy, computed on 5-fold GroupKFold (by case) OOF; masks subsampled to 192x192 for scoring (rows 1-7). Rows 1-4 used effb3 folds 0,2,3 only (fold 1 finished later).

| # | change | validation | time |
|---|---|---|---|
| 0 | smoke: UNet effnet-b3 2.5D (5 slices), 300 steps, fold0 | 2D dice@0.5 = 0.714 (fold0) | 1.5 min |
| 1 | UNet effnet-b3 (timm ra2), 2.5D 5ch, 384px, 16 ep, bs24, lr1e-3 cosine, GPU aug; thr 0.5 no post | OOF folds 0,2,3: score 0.9292 (dice 0.9119, HD 0.0592); per-fold 2D dice f0 .906 f1 .929 f2 .908 f3 .922 | ~45 min/fold (fold1 ~70 min, shared GPU) |
| 2 | #1 + per-class thr (.45/.6/.4; flat, ~no effect) | OOF 0,2,3: 0.9292 | 1 min |
| 3 | #2 + min area 50px@192 (200@384) per slice | OOF 0,2,3: 0.9319 | 3 min |
| 4 | #3 + drop 3D connected components < 3% of largest (per day-class); tried 1/3/10/20% → .9364/.9381/.9365/.9323 | OOF 0,2,3: **0.9381** (dice .9157, HD .0469) | 5 min |
| 5 | UNet tf_efficientnetv2_s (in21k), same recipe, 14 ep; ensemble avg with effb3 (2 models/fold) | OOF 0,2,3 no post: 0.9362 (vs .9292 b3 alone) | ~38 min/fold |
| 6 | #5 + thr .55/.55/.5 + min area 25px@192 + 3D CC < 1% removed | OOF 0,2,3: **0.9396** (dice .9169, HD .0453) | 12 min tuning |
| 7 | #6 evaluated on all 4 OOF folds (0-3, 25,161 slices); same params selected | OOF 0-3: no post 0.9388 → **0.9417** (dice .9211, HD .0445) | 12 min tuning |
| 8 | Final: #7 ensemble + effb3 & v2s trained on all data (12 ep each, no val) = 10 models, post params of #7 | n/a (all-data models unvalidated); OOF reference 0.9417 | 38 min train + 3 min infer |
