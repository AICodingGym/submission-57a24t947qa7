# Report: UW-Madison GI tract segmentation

## Approach
- **Preprocessing** (`src/prep.py`):
  - Each 16-bit slice is normalized per scan volume (divided by its 99.8th percentile), resized to 384x384 and cached as float16.
  - Masks are decoded in C-order (H, W). I checked this on a non-square 310x360 image: decoded this way the mask is one compact component that is bright inside; the transposed decoding gives 31 fragments.
- **Model** (`src/train.py`):
  - 2.5D UNet (segmentation_models_pytorch). The input is 5 neighbouring slices of the same scan (offsets -2..2, clamped at the volume edges) stacked as channels.
  - 3 sigmoid outputs, because the classes can overlap. Loss is BCE + soft Dice. AdamW, lr 1e-3, cosine schedule with warmup, bf16 autocast, batch 24.
  - Two encoders: timm `efficientnet_b3.ra2_in1k` (16 epochs) and `tf_efficientnetv2_s.in21k_ft_in1k` (14 epochs).
- **Data loading**: the server has only 4 CPUs.
  - The whole training set (about 14 GB including bit-packed masks) is kept in GPU memory.
  - Augmentation runs on the GPU: random affine (rotation ±15°, scale ±20%, anisotropic ±8%, shift ±10%), contrast, brightness, gamma and noise.
  - There are no flips, because the anatomy is left/right asymmetric.
- **Validation**:
  - 5-fold GroupKFold by case. Folds 0-3 were trained (one GPU each); fold 4 was never used.
  - Local metric proxy (`src/metric.py`): 0.4 × mean per-slice-per-class Dice (empty/empty = 1) + 0.6 × (1 − 3D Hausdorff per day-class).
  - In the Hausdorff term, coordinates are normalized by volume shape, the distance is divided by sqrt(3), and an empty-vs-non-empty pair scores 1.
  - Post-processing was tuned at 192x192 to keep it cheap on CPU.
- **Post-processing** (`src/post.py`, tuned on OOF):
  - per-class probability threshold;
  - minimum mask area per slice;
  - per scan and class, removal of 3D connected components smaller than 1% of the largest component.
- **Final submission**: average of the probabilities of 10 models, then post-processing:
  - 4 B3 fold models + 4 v2s fold models;
  - 1 B3 and 1 v2s model trained on all training data with the same recipe (12 epochs, no validation).
  - Probabilities are resized bilinearly back to the original resolution before thresholding and RLE encoding.

## Results (OOF, see results.md)
| setup | OOF score |
|---|---|
| B3, threshold 0.5 (folds 0,2,3) | 0.9292 |
| + min-area + 3D CC filtering | 0.9381 |
| B3 + v2s ensemble, no post-processing | 0.9362 |
| B3 + v2s ensemble + post-processing (folds 0,2,3) | 0.9396 |
| same, all 4 folds | **0.9417** |

## What worked and what didn't
- Worked:
  - Removing small 3D connected components gave the largest single gain (+0.006). Most of it comes from the Hausdorff term, which a single stray false-positive blob far from the organ ruins.
  - The minimum-area filter was also positive.
  - Ensembling two encoders gave about +0.007 before post-processing and +0.0015 after.
- Didn't matter: per-class thresholds were essentially flat between 0.3 and 0.7 for one model.
- Too aggressive: removing components smaller than 10-20% of the largest hurt, because it deletes real bowel loops.

## With more time
- Train fold 4 and run more seeds.
- Train longer (losses were still falling) and at higher resolution.
- Try a 2.5D or 3D model with a larger slice context.
- Add a separate slice-level "organ present" classifier to cut false positives on empty slices.
- Use test-time augmentation (small scales or shifts).
- Use the competition's exact Hausdorff implementation.

## Risks: validation vs test
- **The metric is a proxy.**
  - My Hausdorff normalization (coords / shape, ÷ sqrt(3)) may differ from the official scorer, and I computed it on 2x-subsampled masks.
  - Absolute numbers may therefore differ from the leaderboard, though the ranking of methods should hold.
  - The description says Dice = 0 when both masks are empty; I used 1 (the usual Kaggle implementation). Either way the best action on an empty slice is to predict empty.
- **The split differs from the test.**
  - 19 of 28 test cases also appear in train (time split, later days).
  - My CV holds out whole cases, so it is if anything pessimistic for those cases.
- **Untuned final ensemble.** The two all-data models have no validation score. The post-processing parameters were tuned on the 8-fold-model ensemble and applied unchanged to the 10-model ensemble.
- **Tuned on the same OOF.** Thresholds and filters were selected on the same OOF predictions used to report the score. The grid was small, so the optimism should be small.
