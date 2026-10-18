# Experiments (validation = 5-fold stratified OOF log loss on 1283 train rows, fixed folds seed 42)

| # | what changed | OOF log loss | time |
|---|---|---|---|
| 1 | LightGBM on hand-crafted band stats + inc_angle + out-of-fold "same inc_angle" label counts (no CNN) | 0.2060 | 3 s |
| 2 | simple 4-block CNN (3ch: HH,HV,avg), dihedral+shift aug, 60 ep OneCycle, 8x TTA | 0.2292 | 103 s |
| 3 | same + inc_angle concatenated in head | 0.2350 | 103 s |
| 4 | simple CNN, 100 epochs, seed 1 | 0.2296 | 167 s |
| 5 | resnet18.a1_in1k pretrained @75px, 40 ep, lr 5e-4 | 0.2279 | 118 s |
| 6 | mean of CNN logits (2-5) | 0.2108 | - |
| 7 | stacker: LGB + logistic on CNN logits + band stats + OOF inc-group label counts; avg of both | 0.1486 (lgb 0.158, lr 0.1547) | 5 s |
| 8 | stacker + nearby-angle label features (labeled rows within ±0.05°, excl. exact) | 0.1333 (lgb 0.1348) | 5 s |
| 9 | window sweep {0.005..0.5}; best region 0.02–0.05; chose WINS=0.02,0.05, LR gets angle feats too | 0.1160 | ~1 min |
| 10 | CNN batch 2: simple s2 0.2261, s3 0.2414; rn18@150 0.2308; effb0@150 0.2767; rn34@75 0.2357 | - | 100–300 s each |
| 11 | stacker over all 9 CNNs (+WINS 0.02,0.05); test = full-train refit | 0.1133 | 10 s |
| 12 | kNN label feature on (inc, band medians) "same scene" | 0.1137 (no gain, dropped) | 10 s |
| 13 | LGB rounds 200/300/600 vs 400 | 0.1167/0.1135/0.1133 (keep 400) | 30 s |
| 14 | CNN batch 3: simple s4 0.2393, s5 0.2295, s6 0.2421; convnext_nano@75 0.2433; densenet121@75 0.2713; effb0@150 lr3e-4 0.3025 | - | 100–540 s |
| 15 | stacker over 15 CNNs (cnn_mean 0.1924) | 0.1148 | 10 s |
| 16 | nearest distinct-angle group (below/above) rate/dist features | 0.1154 (no gain, dropped) | 10 s |
| 17 | LGB uses only cnn_mean instead of every CNN column (MEANONLY) | 0.1118 (lgb 0.119, lr 0.114) | 10 s |
| 18 | stacker fold-seed robustness (seeds 1,2,3) for WINS 0.02,0.05 / 0.03 / 0.05 | 0.112–0.118 / 0.118–0.119 / 0.118–0.120 → keep 0.02,0.05 | 1 min |
| 19 | CNN batch 4: rn18 s1 0.2404, s2 0.2235, e60 0.2351; rn18@150 s1 0.2299; resnet26d 0.2141; cnxn s1 0.2193; regnety_008 0.2339; rn34 s1 0.2458; rn34@150 0.2398 | - | 110–360 s |
| 20 | stacker over 24 CNNs (cnn_mean 0.1931) | 0.1110 | 10 s |
| 21 | CNN batch 5: rn26d s1 0.2242, s2 0.2071, e60 0.2412; rn26d@100 0.2139; resnet26t 0.2701; seresnext26d 0.2666; cnxn s2 0.2270; resnet50d 0.2229 | - | 160–260 s |
| 22 | pseudo-labelling: test soft labels = fold-k stacker (st4) test preds; test split in 5 parts, fold k trains on 4 parts, predicts the held-out part (no self-memorisation). simple_s0 0.2292→0.2210, rn26d_s0 0.2141→0.2065 | - | 170–225 s |
| 23 | stacker over 32 non-pseudo CNNs (cnn_mean 0.1899) | 0.1105 | 10 s |
| 24 | stacker over all 34 CNNs incl. 2 pseudo (cnn_mean 0.1889) | 0.1100 | 10 s |
| 25 | pseudo round 1 (labels st4): simple s1 0.2165, s2 0.2145; rn18 0.2147; rn26d s1 0.1968, s2 0.1883; cnxn s0 0.2084, s1 0.1969; rn26d@100 0.1926; rn50d 0.2097 | - | 125–310 s |
| 26 | stacker over the 11 pseudo CNNs only (cnn_mean 0.1754); fold seeds 1/2: 0.1058/0.1120 | 0.1057 | 10 s |
| 27 | stacker over all 43 CNNs (cnn_mean 0.1848); fold seeds 1/2: 0.1083/0.1158 | 0.1086 | 10 s |
| 28 | CNN×evidence interaction in LR stacker | 0.1058 (no gain, dropped) | 30 s |
| 29 | pseudo round 2 (labels from st6p): rn26d s0/s1/s2 0.2012/0.2079/0.1946; rn26d@100 s0/s1 0.2003/0.2047; cnxn s0/s1 0.2055/0.2141; simple s0/s1 0.2159/0.2192 | - | 125–255 s |
| 30 | stacker: round-2 only (9) / round 1+2 (20); fold seeds default/1/2 | 0.1025/0.1023/0.1088 vs 0.1036/0.1048/0.1101 | 1 min |
| 31 | pseudo round 3 (labels st7): rn26d s3/s4/s5 0.1920/0.2011/0.1968; rn26d@100 s2/s3 0.1942/0.1958; cnxn 0.1985; rn50d 0.2089; rn18 0.2110; simple 0.2215 | - | 125–310 s |
| 32 | stack sets (mean over 3 fold seeds): p3 only / p2+p3 / ps+p2+p3 | 0.1018,0.1037,0.1090 / 0.1024,0.1031,0.1076 / 0.1033,0.1040,0.1097 | 1 min |
| 33 | output clipping 0.001/0.003/0.01/0.02 on OOF | 0.1034/0.1041/0.1082/0.1140 → keep 0.001 | - |
| 34 | stacker params (3-seed mean): base 0.1057; LR C 0.3/3 0.1059/0.1057; LGB leaves4 0.1057; min_data 30/8 0.1057/0.1054 → keep defaults | ≈0.1057 | 2 min |
| 35 | pseudo round 4 (labels st8): rn26d s6/s7/s8 0.1999/0.1995/0.2088; rn26d@100 s4/s5 0.1931/0.1860; cnxn s3/s4 0.2079/0.2125; simple 0.2222 | - | 120–250 s |
| 36 | stack sets (3-fold-seed mean): ps..p3 0.1057; ps..p4 (37 CNNs) 0.1054; p2..p4 0.1050; p3+p4 0.1038 — all within noise; chose all 37 pseudo CNNs for robustness (st9) | 0.1030 (default folds) | 2 min |
| 37 | pseudo round 5 (labels st9): rn26d@100 s1..s4 0.2006/0.1932/0.1931/0.1891; rn26d@128 s0..s3 0.2029/0.1971/0.1962/0.2009 | - | ~200 s |
| 38 | stack all 45 pseudo CNNs (st10; cnn_mean 0.1733); fold seeds 1/2: 0.1022/0.1078 (3-seed mean 0.1037) | 0.1011 | 10 s |
| 39 | pseudo round 6 (labels st10): 8× rn26d@100 0.1908–0.2071; stack of 53 pseudo CNNs: default/seed1/seed2 0.1018/0.1020/0.1081 (mean 0.1040 vs 0.1037 for st10) → no gain, not used | 0.1018 | ~200 s each |

**Final submission:** st10 = LGB+logistic stacker over 45 pseudo-labelled CNNs + band stats + exact/±0.02°/±0.05° inc_angle label evidence, refit on all train. OOF log loss 0.1011 (default folds), 0.1037 mean over 3 stacker fold seeds.
