# Ventilator pressure prediction: report

## Approach
- **Data.** Each breath is one 80-step sequence (train has 67,905 breaths, test 7,545). I reshaped it to (breath, 80, cols) arrays.
- **Features (59 per step).** R and C as raw values and one-hots (including the R×C combination), time_step and dt, u_in and u_out, cumsum(u_in), area = cumsum(u_in·dt), inspiratory-only area, expanding mean, u_in lags/leads 1–4 with their differences, u_out lags/leads, breath max/mean of u_in, inspiration length, two EWMs of u_in, area/C and u_in·R. Standardized with train+test statistics (features only).
- **Model.** 4-layer bidirectional LSTM (hidden 512/384/256/128 per direction) followed by an MLP head (128, SELU). AdamW with 3 warmup epochs then cosine decay to 0 over 150 epochs. Gradient clipping at 5, fp16 autocast on the LSTMs (head in fp32). All data sits on the GPU, so there is no DataLoader.
- **Loss.** L1 on inspiratory steps (u_out==0, which is all the metric scores), plus 0.25× weight on expiratory steps. The extra weight acts as a regularizer and helped (0.1826 vs 0.1868 at bs256).
- **Classification variant ("fincls").** Softmax over the 950 distinct pressure values in train, trained with CE + L1 on the expected value. It learns faster early on but ends about the same as regression. I keep it for ensemble diversity.
- **Validation.** 5-fold random split by breath with a fixed seed (1234), shared by every run. The score is MAE on u_out==0 steps of held-out breaths.
- **Post-processing.** Snap predictions to the nearest of the 950 train pressure values (gives about −0.0015 MAE every time).
- **Ensemble.** Per config, test predictions are aggregated over its 5 fold models. Configs are then combined with a per-step median. The OOF ensemble score is computed the same way.

## Key decisions / findings (fold 0 unless noted)
- Batch size mattered most: bs512 lr2e-3 gave 0.226, bs256 lr1e-3 0.187, bs128 lr7e-4 0.174. The models overfit heavily (train MAE ~0.07), and smaller batches regularize.
- 250 vs 150 epochs at bs256: no gain (0.1864 vs 0.1868).
- Dropout 0.1 between LSTM layers: small gain. Conv front-end: neutral. GRU: clearly worse (~0.195), so I dropped it.
- fp16 autocast: same accuracy (0.1863 vs 0.1869), ~1.7x faster.
- Ensembling helps a lot: the median of 6 diverse fold-0 models was 0.1594, vs 0.1725 for the best single model.

## Final result
- **Submission:** per-step median of 28 models, snapped to the pressure grid. The models are fin, fin2 (conv front-end, new seed), fincls (classification head) and fin3 (exp_w 0.5), 5 folds each; plus cls2 fold 0 and 7 fold-0 screening models (bs128, ew256, drop256, amp256, base_bs256, cls256, long256).
- **OOF MAE (u_out==0, all 67,905 train breaths): 0.1585.** Best single config: fin2 at 0.1706 snapped. The 3-config median was 0.1603.
- For test, every breath gets the median of all 28 models. Each OOF breath only gets the ~5.6 models that held it out. So the test ensemble is larger than the one validated, and the test score should be equal or slightly better.
- **Excluded:** GRU, plus the late short-training runs (fin4 at 90 epochs, and an 80-epoch cls2 fold 1). Each made the OOF ensemble worse.

## What didn't work
- GRU instead of LSTM (~0.195).
- Training longer at bs256 (250 epochs gave nothing).
- Raising the expiratory loss weight to 0.5 (fin3: 0.1813 vs 0.1745), although fin3 still helped as an ensemble member.
- Short 80–90 epoch runs: too weak, and they hurt the median.
- A mean ensemble instead of the median (0.1601 vs 0.1585).

## Outside data / models
None. Only the competition files in `data/` were used.

## What I'd try with more time
- Much more ensembling (more seeds and folds) and stronger regularization or augmentation for the overfitting.
- Pseudo-labelling the test breaths; transformer/LSTM hybrids; per-(R,C) fine-tuning; smaller batches (64).

## Risks
- The split is random by breath, matching how the hidden test was carved out of the original train data, so CV should transfer well.
- The ensemble's median-vs-mean choice and its member selection were made on the same OOF predictions, which gives a slight optimistic bias (small; only a handful of choices).
- The fold-0 screening models only cover fold 0, so their share of the OOF median is uneven across breaths. They do improve the fold-0 ensemble.
- Test feature standardization used train+test feature statistics (no labels). This is harmless.

(Final numbers are in results.md and work/final.json.)
