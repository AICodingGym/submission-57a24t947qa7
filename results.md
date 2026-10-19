# Experiments
Validation = MAE on u_out==0 steps of held-out breaths (fold 0 of 5 unless noted). Times are wall-clock per run.

| run | change | val MAE | time |
|---|---|---|---|
| base | BiLSTM 512-384-256-128, bs512 lr2e-3, 150 ep, masked L1 | 0.2260 | 22 min |
| conv_ew | base + conv front-end (64ch) + 0.1 loss weight on expiration | 0.2252 | 23 min |
| base_bs256 | base with bs256 lr1e-3 | 0.1869 | 23 min |
| amp256 | base_bs256 + fp16 autocast on LSTMs (~1.7x faster) | 0.1863 | 31 min (shared GPU) |
| bs128 | bs128 lr7e-4 (fp32) | 0.1744 | 60 min (shared GPU) |
| cls256 | bs256 + classification head over 950 pressure values (CE + L1 on expectation) | 0.1860 | 60 min (shared) |
| drop256 | bs256 + dropout 0.1 between LSTM layers | 0.1839 | 60 min (shared) |
| ew256 | bs256 + loss weight 0.25 on expiratory steps | 0.1826 | 60 min (shared) |
| ens6_f0 | median of the 6 fold-0 models above (bs128, ew256, drop256, cls256, amp256, base_bs256), snapped to grid | 0.1594 | - |
| cls128amp | cls head + bs128 + amp (killed at ~ep95/150 to free GPU; superseded by fincls) | 0.1906 (partial) | 40 min |
| long256 | bs256 lr1e-3 for 250 epochs instead of 150 | 0.1864 | 80 min (shared) |
| fin (5 folds) | bs128 lr7e-4 + exp_w 0.25 + drop 0.1 + amp, 150 ep | OOF 0.1749 (folds 0.1738/0.1746/0.1720/0.1796/0.1746) | ~43 min/fold (2 runs per GPU) |
| gru (2 folds) | fin config with GRU instead of LSTM | 0.1968 / 0.2023 -> dropped | ~42 min/fold |
| fin2 (5 folds) | fin + conv front-end (64ch) + seed 7 | OOF 0.1722 (snapped 0.1706) | ~45 min/fold |
| fincls (5 folds) | fin + classification head | OOF 0.1763 (snapped 0.1747) | ~45 min/fold |
| ens fin+fin2+fincls | per-step median of 3 configs, snapped | OOF 0.1603 (all 67,905 breaths) | - |
| ens +7 fold-0 screening models | pooled median | OOF 0.1595 | - |
| fin3 (5 folds) | fin with exp_w 0.5, seed 11 | OOF 0.1813 | ~45 min/fold |
| cls2 (fold 0) | fincls with seed 5 | 0.1724 | 45 min |
| pooled ens (12 runs, 28 models) | median over all fold models + fold-0 screening models | OOF 0.1585 | - |
| fin4 (5 folds) | fin, seed 13, only 90 epochs (time-limited final wave) | OOF 0.1888; adding it to the ensemble hurt (0.1594) -> excluded | ~25-30 min/fold |
| cls2 fold 1 (80 ep) | short cls2 run on the last free slot | 0.1914; hurt the ensemble -> excluded | 24 min |
| **FINAL** | pooled median of fin, fin2, fincls, fin3 (5 folds each), cls2 f0 + 7 fold-0 screening models (bs128, ew256, drop256, amp256, base_bs256, cls256, long256); 28 models; snapped to grid | **OOF 0.1585** | - |
