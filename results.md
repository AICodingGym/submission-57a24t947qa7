# Experiment log
Validation = 5-fold stratified CV OOF multi-class log loss on train (9199 imgs) unless noted; fine-tunes use fold 0 (1840 imgs). Lower is better.

| run | what changed | val logloss | time |
|---|---|---|---|
| convnext_xxlarge.clip_laion2b_soup_ft_in1k@256 zs | IN1k head restricted to 120 dog classes, temperature fit; 4-view TTA (crop/squash x flip) | 0.12627 | <1 min (+3 min extract) |
| convnext_xxlarge@256 lr | multinomial LR on L2-normed pooled features, wd 1e-4 | 0.17118 | 8 s |
| convnext_xxlarge@256 res | zs logits * learned T + LR residual on features, wd 1e-2 | 0.12208 | 8 s |
| eva02_large_448 (m38m_ft_in22k_in1k) zs / res wd3e-3 / lr | same heads | 0.12936 / 0.12543 / 0.15616 | 8 min extract |
| vit_huge_clip_336 (laion2b_ft_in12k_in1k) zs / res wd1e-2 / lr | same heads | 0.13043 / 0.12654 / 0.16160 | 6 min extract |
| blend v1 | geometric-mean blend of 3 res heads, weights fit on OOF (CV of blend 0.10876) | 0.10831 | <1 min |
| convnextv2_huge@512 res wd1e-2 | new backbone | 0.12458 | 10 min extract |
| tf_efficientnet_l2@475 res | new backbone (weak, weight 0 in blend) | 0.17188 | 7 min |
| eva_giant_560 (m30m_ft_in22k_in1k) res | new backbone | 0.13080 | 15 min |
| maxvit_xlarge_512 / eva02_L mim_in22k_ft_in1k / beit_large_512 / so400m_siglip_378 | new backbones | 0.13393 / 0.13504 / 0.14559 / 0.15185 | ~8 min each |
| stack.py (120x120 class-correction stacker) | over 4 best heads | 0.16092 (worse) | 1 min |
| kNN / near-duplicate check | cos>0.99 cross-fold duplicates agree on label only 64-82% -> dropped | - | - |
| multihead (joint residual over 4 backbones) | concat features + per-model zs offsets, wd 1e-2 | 0.10800 | 1 min |
| eva02_L mim_m38m_ft_in1k res wd1e-2 | new backbone (best single) | 0.12004 | 8 min |
| vit_large_clip_336 openai_ft_in12k_in1k res | new backbone | 0.12264 | 6 min |
| convnext_large_mlp clip 384 res | new backbone | 0.16583 | 5 min |
| blend v2 | geometric blend of best head per backbone (13 backbones) | 0.10397 (blend-CV 0.10502) | <1 min |
| ft_eva02L_448 fold0 (full fine-tune, lr 5e-6, LLRD 0.85, GPU aug) | epoch1 val 0.1730 vs zero-shot/res on same fold 0.144/0.136 -> killed, fine-tuning loses the pretrained head's knowledge | 0.1730 (fold0) | 25 min |
| deit3_huge / swinv2_large_384 | new IN1k backbones | 0.16530 / 0.14512 | ~8 min |
| eva02_L mim_m38m_ft_in22k (IN-22k head mapped to the 120 synsets) res wd3e-3 | 22k head instead of 1k head | 0.11171 | 8 min |
| blend v3 | + 22k-head model -> weight 0.46 | 0.09234 (blend-CV 0.09320) | <1 min |
| full-22k-logit residual head (top-K 22k log-probs as features) on eva02 m38m_ft_in22k | K=300 | 0.11428 (vs 0.11171 plain res) - dropped | 1 min |
| more backbones: eva02_L mim_in22k_ft_in22k (22k head) / beit_large_224 in22k_ft_in22k (22k head) / beitv2_L in22k_in1k / vit_L clip laion2b 336 / eva02 in22k_ft_in22k_in1k | | 0.12472 / 0.13163 / 0.12284 / 0.12536 / 0.12113 | ~8 min each |
| weak: caformer_b36 22k / convnext_xlarge 22k @224,320 / swin_large 22k 384 / effnetv2_xl 21k / maxvit_xl 21k 224 / volo_d5 / deit3_huge / caformer in1k 384 | | 0.18-0.26 | |
| test-resolution sweep convnext_xxlarge clip_ft_in1k | 256 / 320 / 384 / 448 | 0.12208 / 0.11542 / 0.12882 / 0.15104 | 5-12 min each |
| eva02 at 518 (pos-embed interpolated) | m38m_ft_in1k / in22k_ft_in22k | 0.12061 / 0.12396 (~= 448) | 10 min |
| blend v4 | 36 inputs, Adam-fitted non-negative geometric weights; 5 get weight | 0.08960 (blend-CV 0.09017) | 2 min |
| confusion-matrix smoothing post-proc on blend | p'=(1-a)p + a p@C, nested CV | 0.08973 vs 0.08959 (no gain) | 1 min |
| new backbones: hiera_huge_224 / eva_giant_336 m30m / convnextv2_huge_384 / regnety_1280 SWAG / vit_L clip openai_ft_in1k 224 / vit_L,H clip laion2b_ft_in1k / eva_giant clip_ft_in1k 336 | | 0.13567 / 0.13755 / 0.13521 / 0.15136 / 0.12768 / 0.16175,0.17146 / 0.17414 | 5-15 min each |
| 12k-head CLIP models (6 breeds missing from IN-12k) | convnext_xxlarge / vit_L / vit_H in12k | 0.236-0.242 | |
| res-variants of 22k models: beit_large 22k @288/@384, eva02 m38m 22k @336 | | 0.12975 / 0.17316 / 0.11648 | |
| blend v5 | 50 inputs, 9 with weight | 0.08939 (blend-CV 0.09018) | 2.5 min |
| resolution sweep round 2: eva02 m38m 22k @280/@392, convnext_xxlarge @288/@352, vit_L openai @280, vit_L laion @392, beit_L 22k @256 | | 0.13706 / 0.11363 / 0.11506 / 0.12136 / 0.12542 / 0.12696 / 0.12783 | 5-10 min each |
| multi-layer features (blocks -2,-5,-9 mean tokens) residual head on eva02 m38m 22k | | 0.11185 (vs 0.11171) no gain | 10 min |
| vit_giant dinov2 @448 linear probe (no IN head) | | 0.36237 | 35 min |
| blend v6 | 57 inputs, 12 with weight | 0.08925 (blend-CV 0.09042; v5 0.09018 -> noise-level) | 2.5 min |
| older IN-1k-trained models: resnet152.a1h@288 zs / vit_large_patch32_384.orig_in21k_ft_in1k res / resnext101_32x32d WSL@288 zs / resnext101_32x16d SWSL | strong (likely memorised ImageNet-train labels) and diverse | 0.11342 / 0.11640 / 0.12170 / 0.16304 | 2-4 min each |
| blend v7 | + those models (ViT-L/32 orig gets weight 0.36, ResNet152 0.18) | 0.07717 (blend-CV 0.07829) | 2.5 min |
| heads.py fix: residual temperature initialised from zero-shot grid-fit T (BCE-trained heads have T far from 1) | resnet152.a1h res 0.12413 -> 0.11157; others unchanged | | 15 min refit |
| more IN-1k models: resnet152.a1_in1k@288 zs / res wd3e-2 | ResNet-strikes-back a1 recipe weights; acc 98.2% | **0.05814 / 0.05739** | 3 min |
| resnet101.a1h@288 / resnet152.tv / dm_nfnet_f6 / deit3_L_384 in1k / beit_L_384 / convnext_xl 22k_ft_in1k_384 / eva02 22k@392 / eva02 m38m_in1k@392 | | 0.13843 / 0.23075 / 0.15232 / 0.15285 / 0.14285 / 0.18082 / 0.12593 / 0.12308 | |
| blend v8 | + resnet152.a1 (weight 0.68) | 0.05085 (blend-CV 0.05131) | 3 min |
| ResNet-strikes-back family @288 (best head) | resnet152.a1 res 0.05739 / resnext50_32x4d.a1 0.07838 / resnet152.a2 0.08151 / resnet101.a1 0.09059 / resnet101.a2 0.10961 / ecaresnet50t.a1 0.12243 / resnet50d.a1 0.12857 / resnet50.a1 0.15137 | | ~2 min each |
| resnet152.a1 test-res sweep | 224 / 256 / 288 / 320 | 0.06857 / 0.06080 / 0.05739 / 0.05803 | |
| other IN-1k models this round: resnetv2_101x3_bit 21k_ft_in1k@448 / resnetrs420@320 / vgg19_bn / inception_resnet_v2 etc. | | 0.12296 / 0.12964 / >0.15 | |
| blend v9 | 82 inputs, 8 with weight | 0.04971 (blend-CV 0.05074) | 3 min |
| final batch: resnext101_64x4d.c1 @224/256/288/320 / regnety_320.tv2 / seresnext101_32x8d.ah / resnet152d.ra2 / resnext101_32x8d.tv2 / resnet152.gluon / wide_resnet101_2.tv2 / resnet152.tv2 / resnet50.c1 | | 0.0759 / 0.0689 / 0.0683 / 0.0698 / 0.1042 / 0.1189 / 0.1377 / 0.1313 / 0.1642 / 0.1850 / 0.2069 / 0.2060 | ~2-3 min each |
| **blend FINAL** (submission.csv) | 64 inputs (single OOF < 0.135), 9 with weight: resnet152.a1 @256/288/320, resnext101_64x4d.c1 @256/288, resnext50.a1, eva02-L 22k @336/392, ViT-L/32 orig | **0.04867 in-sample, 0.05006 nested blend-CV** | 3 min |
