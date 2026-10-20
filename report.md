# RANZCR CLiP – report

## Result
- **Validation (5-fold OOF, 9-column mean AUC): 0.9559** for the ConvNeXt-S-640 + EffNetV2-S-640 fold ensemble.
- Best single family (ConvNeXt-S 640, 5 folds): OOF 0.9514. EffNetV2-S 640, 5 folds: OOF 0.9510.
- `submission.csv` = that ensemble plus three extra model families at weight 0.3 (14 models in total).

## Task / metric
Multi-label chest X-ray classification. The scored metric is the mean ROC AUC over the **9 columns in
sample_submission.csv**. CVC-Normal and Swan-Ganz are not in the submission, so they are not scored, but they are still
trained as auxiliary targets. Every number here is that 9-column AUC.

## Approach and key decisions
- **Preprocessing:** every JPEG was decoded once to 768x768 grayscale uint8 (aspect ratio ignored) and stored as a memmap. Training resizes
  on the GPU to 512 or 640. Augmentation is done entirely on the GPU: affine (±12° rotation, ±15% scale, ±8% shift), horizontal flip,
  gamma/contrast/brightness, and one coarse-dropout box. The reason: the shared server's CPU was the bottleneck.
- **Validation:** 5-fold StratifiedGroupKFold grouped by PatientID, stratified on the rarest positive label (`work/folds.csv`).
- **Training:** timm ImageNet backbones with 1-channel input, BCE on 11 labels, AdamW, warmup + cosine, bf16, and an EMA of the weights
  (decay 0.999; EMA weights are evaluated and used). Inference uses horizontal-flip TTA.
- **Model selection (fold 0, 12 epochs):** ConvNeXt-S 512 scored 0.948 and was stable. EffNetV2-S 512, EffNetV2-S 640 and EffNetV2-M 512 peaked at
  0.943–0.946 around epochs 8–9, then overfit, so later EfficientNet runs used 7–8 epochs. 640px beat 512px.
  ConvNeXt-S at 640 was the main model (0.950–0.959 per fold at 11–12 epochs).
- **Ensemble:** simple probability averaging. On fold 0, blending different architectures added about +0.002 over the best single model.
  Across all 5 folds, ConvNeXt-640 + EffNetV2-S-640 gives 0.9559, against 0.9514 and 0.9510 for each alone.

## Outside data / models
Only ImageNet-pretrained weights from the Hugging Face Hub via timm, used as initialisation. No competition-derived data or models.
- `convnext_small.fb_in22k_ft_in1k`: https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k
- `tf_efficientnetv2_s.in21k_ft_in1k`: https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k
- `tf_efficientnetv2_m.in21k_ft_in1k`: https://huggingface.co/timm/tf_efficientnetv2_m.in21k_ft_in1k
- Downloaded only for a speed benchmark and not used in any model: `resnet200d.ra2_in1k`, `tf_efficientnet_b5.ns_jft_in1k`.
- `train_annotations.csv` (part of the competition data) was rasterised into line masks for an auxiliary segmentation head experiment. That experiment was not finished, and no model in the submission uses it.

## What worked / what didn't
- Worked: higher resolution (640 > 512); ConvNeXt (stable, no late overfitting); EMA; mixing ConvNeXt with EfficientNet in the ensemble.
- EfficientNets overfit after about 9 epochs on a 12-epoch cosine schedule, so shorter schedules were better.
- **Infrastructure problems cost a lot of time.** The shared server's CPUs made the EfficientNet runs launch-bound. timm's per-tensor EMA update
  alone was most of the step time, and replacing it with a fused `_foreach_lerp_` made a profiled 20-step window about 4× faster. Throughput
  still varied 2–3× between GPUs and over time. Two queued jobs OOM'd because they started before the previous process had released its memory.
- Auxiliary segmentation from the annotations (EffNetV2-S 512): 0.934 at epoch 4/9, against 0.926 at epoch 4 for the baseline (different
  schedules, so not a clean comparison). It was killed for lack of time, so it is unproven.
- Last-minute 4-epoch EffNetV2-S-512 runs on folds 2–4 added nothing to the 5-fold OOF (0.95590 → 0.95590), so they are not in the submission.
- Because of time, some fold models are under-trained: ConvNeXt fold 4 got only 6 epochs (0.943), and EffNetV2-S fold 1 was cut at epoch 5 of 8.

## With more time
- Finish the annotation-guided auxiliary segmentation (or a teacher–student setup that uses the line masks). It looked promising, and CVC-Borderline
  (AUC ≈ 0.86) is the weakest column.
- Retrain ConvNeXt fold 4 and EffNetV2 fold 1 with full schedules. Try higher resolution (768+), more seeds, and stronger backbones.
- Use torch.compile / CUDA graphs to beat the CPU launch bottleneck.

## Risks: validation vs test
- The OOF score comes from fold models that each saw 80% of the data. The test predictions average those fold models, which usually does as well as OOF or slightly better.
- The weight-0.3 extras were only validated on fold 0 (+0.0009), so that small gain is a bit optimistic.
- Several fold models had shortened or truncated schedules, so per-fold quality is uneven (0.943–0.959).
- Rare columns (ETT-Abnormal has about 15 positives per fold) make the per-column AUC noisy. The test split (3,009 images) will be noisy too.
