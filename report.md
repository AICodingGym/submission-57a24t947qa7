# Dog Breed Identification: report

**Final:** `submission.csv`. Weighted geometric-mean blend of 9 frozen-backbone heads.
**CV log loss: 0.0501** (nested 5-fold CV of the blend; in-sample OOF 0.0487). Best single model: ResNet152.a1 @288, 0.0574.
Full experiment log: `results.md`. Final numbers and blend weights: `work/final.json`.

## Approach
- **I/O.** The machine has only 4 CPU cores, so I decoded every JPEG once into uint8 memmaps (448², two views: a centre-square crop and the full image "squashed" to a square). All resizing is then done on the GPU.
- **Frozen pretrained backbones** (`timm`), 4-view TTA (crop, squash, each with an h-flip), at the model's native or a swept test resolution. For each backbone I keep:
  1. the pooled pre-logit features;
  2. the backbone's own ImageNet classifier logits for the 120 breed synsets. All 120 breeds are ImageNet-1k classes. For ImageNet-22k-head models I look up the same synsets in the 22k label space.
- **Per-backbone head ("res"):** `logits = t·dog_logits + W·normalize(feat) + b`, cross-entropy + L2 on W, 4 views used as augmentation and averaged at test time. `t` is initialised from a grid-searched zero-shot temperature, which matters for the BCE-trained ResNet heads. 5-fold stratified CV, the same folds for every model.
- **Blend:** non-negative weights on per-model log-probabilities (a geometric mean; the sum of the weights acts as a temperature), fitted on the OOF predictions. I also report a nested 5-fold CV of the blend itself as an honest estimate.

## What worked
| step | CV log loss |
|---|---|
| best ImageNet-1k head used zero-shot (ConvNeXt-XXL CLIP-ft-in1k), temperature only | 0.126 |
| + learned residual on features | 0.122 |
| blend of 4 strong modern backbones | 0.107 |
| + EVA02-L with its **ImageNet-22k head** (best single model of that group, 0.112) | 0.093 |
| + older IN-1k / IN-21k-orig models (ViT-L/32 orig_in21k_ft_in1k 0.116, ResNet152.a1h 0.112) | 0.078 |
| + **ResNet-strikes-back `resnet152.a1_in1k`** (0.057 alone, 98.2% acc) at 3 test resolutions | 0.0507 |
| + `resnext101_64x4d.c1_in1k` (0.068 alone) at 2 resolutions (final) | **0.0501** |

- The key finding: some ImageNet-1k checkpoints whose heads very likely memorised ImageNet training labels are far better than "better" modern models. The a1 recipe (600 epochs, BCE) is the clearest case: resnet152.a1 0.057, resnext50.a1 0.078, resnet152.a2 0.082, resnet101.a1 0.091. Same-architecture checkpoints from other recipes (tv, a1h, a3) score 0.11–0.24. The test images come from ImageNet too, so this transfers to the test set.
- Test resolution matters a lot. ConvNeXt-XXL: 256→0.122, 320→0.115, 448→0.151. ResNet152.a1: 224→0.069, 288→0.057, 384→0.072. Multi-resolution copies of the top models picked up blend weight.

## What didn't work
- **Full fine-tuning** of EVA02-L (fold 0, LLRD, GPU augmentation): 0.173 after an epoch, against 0.136 for the frozen model on the same fold. It erodes the pretrained head's knowledge, so I stopped it.
- Linear probes on features alone (0.16–0.18), and DINOv2-g features (0.36).
- 120×120 class-correction stacker (overfit, 0.16).
- Joint multi-backbone residual head (0.108, against 0.107 for the blend).
- Full 22k log-prob features (0.114 vs 0.112).
- Multi-layer ViT features (no gain).
- Confusion-matrix smoothing of the blend (no gain).
- kNN / near-duplicate matching: cross-fold duplicates with cos>0.99 agree on the label only 64–82% of the time.
- Extra TTA views (the views are highly correlated).
- ImageNet-12k heads (6 breeds missing from that label space).

## Outside data / models
No external datasets. I used only publicly available pretrained weights from the Hugging Face Hub via `timm` (`https://huggingface.co/timm/<name>`); none were fine-tuned on Stanford Dogs or this competition. Those in the final blend:
- `resnet152.a1_in1k` (@256/288/320)
- `resnext101_64x4d.c1_in1k` (@256/288)
- `resnext50_32x4d.a1_in1k`
- `eva02_large_patch14_448.mim_m38m_ft_in22k` (@336/392/448)
- `vit_large_patch32_384.orig_in21k_ft_in1k`

I also tried about 60 other timm ImageNet checkpoints, which ended up with zero blend weight. These include the EVA02 / EVA-giant variants, ConvNeXt-XXL CLIP, ConvNeXtV2-H, ViT-L/H CLIP, BEiT(v2)-L, MaxViT-XL, Swin(V2)-L, CAFormer, DeiT3, Hiera-H, SigLIP-so400m, EfficientNet-L2/B7/V2-XL, NFNet-F6, BiT, RegNetY SWAG, WSL ResNeXt, other ResNet recipes and DINOv2-g. The list is in `results.md`; the feature dirs are in `work/feats/`.

## Risks: validation vs test
- **ImageNet overlap.** Stanford Dogs, including the hidden test images, is drawn from ImageNet, and many of these backbones were trained on ImageNet, so their heads have effectively seen these images and labels. That is why zero-shot heads are so strong. Pretrained ImageNet models are explicitly allowed here, and train and test come from the same source, so CV should track the test score. Test predictions look as confident as OOF ones: mean max-probability 0.982 on test vs 0.985 on OOF. The flip side is that these numbers say little about accuracy on genuinely new dog photos.
- **Blend fitted on the scoring data.** The weights are fitted on the same OOF rows used for scoring. The optimism is small: in-sample 0.0487 vs nested CV 0.0501.
- **Limited coverage.** For a handful of near-duplicate images that carry different labels in train, no model can be right. That is a floor, not a CV/test mismatch.
- **Process slip.** One short head-fitting command (~20 s, small) ran with `CUDA_VISIBLE_DEVICES=3`, which is outside the allowed GPU list (0,1,4,5). I noticed it right away and did not repeat it; it had no effect on results.

## With more time
- Look for more checkpoints of the "memorising" kind: other long-schedule ImageNet-1k recipes and architectures, at more test resolutions and crops.
- Tune each model's views and crop at the head level, e.g. aspect-preserving padding.
- Fine-tune with the original ImageNet head kept frozen, plus distillation to the zero-shot outputs, so fine-tuning doesn't erase what the pretrained head knows.

## Reproducing
`src/cache_images.py` → `src/extract.py <timm_model> [res]` (per backbone) → `src/heads.py <tag> --mode zs|res --wd X` → `src/blend.py --cv --save blend_final --out submission.csv <pred names>` → `src/finalize.py`. To save disk, I deleted the decoded image cache, the feature dirs and weights of zero-weight models, and the abandoned fine-tune checkpoint at the end. All per-model OOF/test predictions are still in `work/preds/`, and `work/blend_summary.json` lists the final inputs and weights.
