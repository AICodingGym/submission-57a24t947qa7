# Experiments
Validation: 5-fold GroupKFold by game_play on all 216 train plays; MCC computed over ALL label rows (rows filtered out by the distance>3yd rule count as predicted 0); single global threshold tuned on OOF.

| # | change | val MCC | time |
|---|---|---|---|
| 0 | heuristic: pair dist<0.8yd → contact, ground → 0 (early safety submission) | not validated | 1 min |
| 1 | XGBoost (GPU) on tracking + baseline-helmet box features (194 feats), rows dist<=3yd + all ground rows | 0.7284 | feats 12 min + 1 min train |
| 2 | + resnet18 2.5D CNN OOF probs (3 gray frames ±0.1s + helmet-mask channel, 64px crops→128, 2-fold by play, 3 ep) as stacked features w/ temporal lags | 0.7783 | CNN 8 min/fold (parallel GPUs) |
| 3 | + efficientnet_b0 CNN (same setup) features as well | 0.7822 | CNN 13-20 min/fold |
| 4 | GBDT lr 0.02 depth 7 (vs 0.05/8) on #3 features | 0.7828 (0.7851 smoothed) | 2.5 min |
| 5 | temporal smoothing of #3 probs within pair, rolling mean window 3 (win 5 similar) | 0.7854 | <1 min |
| 6 | + r18 CNN on 5-frame (±0.2s) crops, 4 ep — CNN alone worse (overfits), no stack gain | 0.7821 (0.7851 sm) | 12 min/fold |
| 7 | + cross-pair context feats (per player/step max/sum of CNN probs over other pairs, ground prob) | 0.7826 (0.7854 sm) | 3 min |
| 8 | r18 on larger-context 80px crops (res 160, 2 ep): best single CNN val AUC 0.945/0.941; all 5 CNN feat sets stacked | 0.7819 (0.7851 sm) | 7 min/fold |
| 9 | average CNN probs of r18+eb0+r18b80 before feature building | 0.7828 (0.7853 sm) | 3 min |
| 10 | blend (mean prob) of GBDTs #3,#9,#7 + smoothing w3 → **submission** | 0.7868 | - |
| 11 | eb0 CNN, **4-fold** by play, 80px crops, res 128, 2 ep (val AUC .944–.954) → stack alone | 0.7847 (0.7882 sm) | ~9 min/fold, 4 GPUs parallel |
| 12 | blend GBDTs: k4 + (k4 + 2-fold avg feats) + (avg of all CNN probs) | 0.7896 | - |
| 13 | r18 4-fold 80px res160 (AUC .942–.952); GBDTs on avg(eb0k4,r18k4) / both sets | 0.7853 / 0.7851 raw; blend of 5 GBDTs 0.7902 | 7 min/fold |
| 14 | LightGBM stacker variant (same feats as k4c) | 0.7858 raw; adding to blend 0.7898 (no gain, dropped) | 3 min |
| 15 | eb0 4-fold 80px **res160** (AUC .947–.957, best CNN); GBDT on avg3 k4 CNNs / eb0r160+avg | 0.7873 / 0.7872 raw; blend of 7 GBDTs **0.7906** | 10 min/fold |
| 16 | efficientnet_b2 4-fold 80px res160 (AUC .949–.957); GBDT on avg of 4 k4 CNNs / eb2+eb0r160 sets | 0.7877 / 0.7872 raw | 13 min/fold |
| 17 | **FINAL**: mean prob of GBDTs k4d,k4e,k4f,k4g (all built on 4-fold CNN feats) + rolling-3 smoothing, thr 0.395 | **0.7916** | - |
