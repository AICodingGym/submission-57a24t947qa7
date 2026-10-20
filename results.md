# Experiments
Metric: mean ROC AUC over the 9 submission columns (val = patient-grouped fold 0 unless noted). Time = wall-clock training time.

| run | what changed | val AUC9 | time |
|---|---|---|---|
| prior | constant train-prior submission (placeholder) | 0.5 | 0 min |
| v2s_512_f0 | EffNetV2-S in21k, 512px, 12 ep, lr 5e-4, EMA, hflip TTA | 0.93848 (last; peak 0.94274 @ep9, overfits after) | 44 min |
| cnxs_512_f0 | ConvNeXt-S in22k, 512px, 12 ep, lr 1.5e-4, wd 0.05 | 0.94791 (peak 0.94812 @ep10; stable) | 60 min |
| v2s_640_f0 | EffNetV2-S, 640px, 12 ep | 0.94286 (peak 0.94635 @ep9, overfits after) | 56 min |
| v2m_512_f0 | EffNetV2-M in21k, 512px, 12 ep, lr 4e-4 | 0.94294 (peak 0.94624 @ep8) | ~63 min |
| ens fold0 | mean prob cnxs_512 + v2s_640 (fold 0) | 0.94988 | - |
| ens fold0 | mean of all 4 fold-0 models | 0.94926 | - |
| v2s_512seg_f0 | v2s 512 + aux line-seg head (annotations), 9 ep — KILLED at ep4 (CPU-bound slowdown, no time) | 0.93423 @ep4 (baseline 12-ep run: 0.92629 @ep4) | 34 min |
| cnxs_640_f1 | ConvNeXt-S 640px, 12 ep (fold 1) | 0.95307 | 125 min |
| cnxs_640_f0 | ConvNeXt-S 640px, 12 ep (fold 0) | 0.94999 | 151 min |
| cnxs_640_f3 | ConvNeXt-S 640px, 11 ep (fold 3), fused EMA | 0.95932 | 100 min |
| cnxs_640_f4 (1st try) | OOM at start (queued too early while previous job was still exiting) | - | 0 |
| ens fold0 | cnxs_640 + cnxs_512 + v2s_640 + v2m_512 (fold 0, equal weights) | 0.95155 (cnxs_640 alone 0.94999) | - |
| cnxs_640 OOF f0,f1,f3 | fold average | 0.95333 | - |
| v2s_640e8_f2 | EffNetV2-S 640, 8 ep (fold 2) | 0.95112 | 35 min |
| cnxs_640_f2 | ConvNeXt-S 640px, 12 ep (fold 2) | 0.95499 | 155 min |
| cnxs_640_f4 | ConvNeXt-S 640px, only 6 ep (time) (fold 4) | 0.94326 | 43 min |
| v2s_640e8_f3 | EffNetV2-S 640, 8 ep (fold 3) | 0.95716 | 34 min |
| cnxs_640 OOF (5 folds) | fold models averaged on test | 0.95143 | - |
| ens folds 0,2,3 | cnxs_640 + v2s_640(e8) equal weights | 0.95593 | - |
| v2s_640e8_f1 | EffNetV2-S 640, 8-ep schedule, stopped at ep5 by time cap (slow GPU) (fold 1) | 0.94660 | 66 min |
| v2s_640e8_f4 | EffNetV2-S 640, 8 ep (fold 4) | 0.95706 | 34 min |
| v2s_640e7_f0 | EffNetV2-S 640, 7 ep (fold 0) | 0.94468 | 30 min |
| v2s_512e7_f1 | EffNetV2-S 512, 7 ep (fold 1) | 0.94775 | 20 min |
| v2s_640 family OOF (5 folds: e7 f0, e8 f1-f4) | | 0.95100 | - |
| **ens 5-fold OOF** | cnxs_640 + v2s_640 families, equal weights (all 27,074 rows) | **0.95590** | - |
| ens fold0 | + v2s_512, cnxs_512, v2m_512 extras at weight 0.3 each (final submission) | 0.95218 vs 0.95128 without extras | - |
| v2s_512e4_f2/f3/f4 | EffNetV2-S 512, 4 ep, last-minute fill-ins (folds 2,3,4) | 0.93444 / 0.94479 / 0.94123 | 11 min each |
| ens 5-fold OOF + v2s_512 family (5 folds) at w=0.3 | no change (0.95590); at w=0.5 0.95563 -> not used, submission unchanged | 0.95590 | - |
