# Report — TensorFlow Speech Recognition

**Final:** `submission.csv` is the mean probability of 9 models:
- 3 fold-0 models: ResNet, ResNet with mixup, EfficientNet-B0.
- 6 retrains on all training data: 3 ResNet seeds (1 with mixup) and 3 EfficientNet-B0 seeds.

**Validation:** 12-label accuracy of **0.9925** for the 3-model ensemble on held-out speakers (fold 0).

**Compute:** about 2.1 h total on one shared A6000 (cuda:0 only; another user's job was also running on it).

## Approach
- **Data understanding.** The test set here (6,473 clips) is a ~10% random hold-out of the original Speech Commands v0.01 training folders, not the original Kaggle test set. Its loudness and clip-length distributions match train. It holds one 60 s clip, which is clearly a background-noise recording. So the true label distribution is about 63% `unknown` (the 20 auxiliary words), about 37% the 10 commands, and essentially no `silence`.
- **Model target.** I train a **30-word softmax** on the natural class distribution, then sum the probabilities of the 20 non-command words into `unknown`. This keeps the train prior identical to the expected test prior, and it uses the word identity of unknown words as a richer training signal. I did not train a silence class, because quiet word clips in test still carry word labels. The only `silence` prediction is a rule for clips longer than 1 s.
- **Features and augmentation (all on GPU).** Audio is held on the GPU as int16. Features are log-mel (64 bins, 30 ms window, 10 ms hop). Augmentations: random time shift (±100 ms), gain (0.5–1.5x), background-noise mixing from `_background_noise_` (p=0.6, amplitude ≤ 0.3), speed perturbation (0.85–1.15) via linear resampling, SpecAugment (2 frequency + 2 time masks), and label smoothing 0.05. Only 4 CPU cores were available, so the GPU pipeline avoids data-loader bottlenecks.
- **Models.**
  - (a) A ResNet with 8 basic blocks, width 64, trained from scratch.
  - (b) An ImageNet-pretrained EfficientNet-B0 (timm, 1 input channel, mel image upsampled 2x).
  - (c) Model (a) with mixup.
  - All use AdamW with a one-cycle schedule and bf16 autocast.
- **Validation.** 5-fold GroupKFold by **speaker**; for time reasons only fold 0 (~12k clips) is used. Metric: 12-label accuracy.
- **Post-processing.**
  - 77 test clips are byte-identical to training clips (the same recording appears in both). For these I copy the training label when it is unambiguous; this changes ~2 predictions.
  - Clips longer than 1 s are predicted `silence`.
- **Final.** Equal-weight probability average of the 3 fold-0 models and 6 all-data retrains (9 models). The test predictions of the individual models agree with the ensemble on about 98.7–98.9% of clips (30-word argmax).

## What worked / didn't
- The small from-scratch ResNet (99.14%) and the pretrained EfficientNet-B0 (99.11%) score about the same. Averaging them gives +0.1% (3-model ensemble: 99.25% on fold 0).
- Mixup: about neutral alone (99.15% vs 99.14%), but it adds ensemble diversity.
- Running two trainings at once on the shared GPU roughly halved per-job speed. It gave no throughput gain, but it let different recipes train side by side.
- Down-weighting `unknown` at decision time (most residual errors are commands predicted as unknown): no gain on validation, so not used.

## With more time
- Full 5-fold CV for every model. More architectures (wider ResNet, raw-waveform 1D CNN, a pretrained audio model such as PANNs/AST).
- Pseudo-labelling the test clips, TTA over time shifts, and cleaning noisy or near-silent training labels.

## Risks: validation vs test
- Validation holds out **speakers**. The test set is most likely a random file-level split, so many test speakers also appear in train (77 exact duplicates support this). Test accuracy is therefore probably *at or above* the validation score.
- The validation score comes from a single fold (~12k clips), so its standard error is about 0.1%.
- Models trained on all data have no held-out estimate. Their quality is inferred from the same recipe's fold-0 score.
- If the grader's test labels treat any clips as `silence` beyond the long noise clip, those would be missed. I found no evidence of this.
