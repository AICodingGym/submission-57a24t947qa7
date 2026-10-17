# Experiments
Validation: 5-fold GroupKFold by speaker (fold 0 unless noted); metric = 12-label accuracy (10 commands + unknown) on held-out speakers.

| run | change | val12 (fold0) | val30 | time |
|---|---|---|---|---|
| smoke | ResNet w32, 2 epochs (sanity) | 0.9691 | 0.9491 | 0.3 min |
| r64_e30 | ResNet w64, 30 ep, shift/gain/noise/speed aug + SpecAugment, LS 0.05 | 0.9914 | 0.9822 | 13.7 min |
| eb0_e25 | timm efficientnet_b0 (ImageNet pretrained, 1ch, 2x upsampled mel), 25 ep, lr 2e-3 | 0.9911 | 0.9820 | 19 min (shared GPU) |
| r64_mix | r64_e30 + mixup(alpha 0.4, p=0.5) | 0.9915 | 0.9821 | 26 min (shared GPU) |
| ens3 (fold0) | mean prob of r64_e30 + eb0_e25 + r64_mix | 0.9925 | - | - |
| r64_all | r64_e30 recipe on all training data (seed 1) | n/a (no holdout) | - | 32 min (shared GPU) |
| eb0_all | eb0_e25 recipe on all training data (seed 1) | n/a | - | 24 min (shared GPU) |
| r64mix_all | r64_mix recipe on all data (seed 2) | n/a | - | 33 min (shared GPU) |
| eb0_all2 | eb0 recipe on all data (seed 2) | n/a | - | 24 min (shared GPU) |
| r64_all3 | r64 recipe on all data (seed 3) | n/a | - | 33 min (shared GPU) |
| eb0_all3 | eb0 recipe on all data (seed 3) | n/a | - | 23 min (shared GPU) |
| **FINAL** | mean prob of 9 models (3 fold-0 + 6 all-data), 30->12 mapping, dup-copy + long-clip=silence rules | 0.9925 (3-model fold-0 ensemble; final ensemble has no holdout) | - | ~2.1 h total |
