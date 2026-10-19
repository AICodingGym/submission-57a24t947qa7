# Experiments (validation = word-level Jaccard on out-of-fold chaii train predictions, 5-fold stratified by language)

| # | experiment | val Jaccard | time |
|---|---|---|---|
| 0 | smoke: xlm-r-large-squad2, fold0 only, 0.1 epoch | 0.603 (fold0) | 0.5 min |
| 1 | xlmr_a ep1 (fold0 only, early submission) | 0.670 (fold0) | 4 min |
| 2 | xlmr_a: xlm-roberta-large-squad2, chaii(4/5)+MLQA-hi+XQuAD-hi, 384/128, lr1.5e-5, 3 ep, neg 30%; ep2 checkpoint preds | 0.6857 (5-fold OOF) | 60 min |
| 3 | xlmr_a ep2 + decoding: max 80 chars, no span across newline, strip unbalanced brackets/quotes | 0.6982 | - |
| 4 | muril_s1: MuRIL-large stage-1 on SQuAD1.1-en + MLQA-hi + XQuAD-hi (1 epoch, lr3e-5) | (pretraining only) | 30 min |
| 5 | muril_a: MuRIL-large (from muril_s1), chaii+hindi ext, lr2e-5, 3 ep; ep1 preds | 0.7038 (ep2 0.6947, ep3 0.6891) | 53 min |
| 6 | ensemble xlmr_a:ep2 + muril_a:ep1 (char-level logit average) | 0.7138 | - |
| 7 | xlmr_b: xlm-r-large-squad2, 512/192, lr1e-5, 2 ep; final | 0.7002 (ep1 0.6888) | 48 min |
| 8 | ensemble xlmr_a:ep2 + xlmr_b:final + muril_a:ep1 + muril_a:ep2 | 0.7238 | - |
| 9 | muril_b: MuRIL, lr1e-5, 2 ep, seed 7; ep1 / final | 0.6996 / 0.6927 | 35 min |
| 10 | muril_c: MuRIL, chaii only (no Hindi ext in fold stage), Tamil rows x3, lr2e-5, 2 ep; ep1 / final | 0.6983 / 0.6959 | 20 min |
| 11 | rembert_a: RemBERT (stage-1 SQuAD+Hindi, 53 min) then chaii+Hindi ext, lr1.5e-5, 3 ep; ep1/ep2/final | 0.6731 / 0.6871 / 0.6818 | 110 min |
| 12 | ensemble xlmr_a:ep2+xlmr_b:final+muril_a:ep1+muril_c:final+rembert_a:ep2, raw logit avg | 0.7263 | - |
| 13 | same 5 members, per-model softmax-prob averaging (adopted) | 0.7326 | - |
| 14 | infoxlm_a: InfoXLM-large (stage-1 SQuAD+Hindi, 32 min), chaii+Hindi ext, 384/128, lr1.5e-5, 2 ep; ep1 / final | 0.6919 / 0.7058 | 75 min |
| 15 | muril_d: MuRIL 512/192, lr2e-5, 2 ep, seed 21; ep1 / final | 0.7073 / 0.7111 | 42 min |
| 16 | ensemble of 7 (xlmr_a:ep2, xlmr_b, muril_a:ep1, muril_c, muril_d, infoxlm_a, rembert_a:ep2), prob avg | 0.7420 | - |
| 17 | infoxlm_b: InfoXLM 512/192, 2 ep, seed 3; ep1 / final | 0.6876 / 0.6969 | 52 min |
| 18 | muril_e: MuRIL 512/192, Tamil x2, lr2e-5, 2 ep, seed 33; ep1 / final | 0.7087 / 0.6994 | 45 min |
| 19 | rembert_b: RemBERT, Tamil x2, 2 ep, seed 5; ep1 / final | 0.6897 / 0.6699 | 80 min |
| 20 | ensemble 9 runs (all but rembert_b), equal weights | 0.7432 | - |
| 21 | ensemble 10 runs, equal weights (leave-one-out range 0.735-0.744 -> member choice is within noise) | 0.7365 | - |
| 22 | greedy (Caruana) selection on OOF: in-sample 0.7492, but NESTED (select on 4 folds, score 5th) 0.7270 vs equal-weight-all 0.7365 -> greedy rejected | 0.7270 (nested) | - |
| 23 | ensemble of 9 using final checkpoints only (no epoch selection) | 0.7364 | - |
| 24 | muril_f: MuRIL 512/192, lr1.5e-5, 2 ep, seed 44; ep1 / final | 0.6953 / 0.6984 | 45 min |
| 25 | muril_g: MuRIL 384/128, lr2e-5, 2 ep, seed 55; ep1 / final | 0.7086 / 0.7038 | 37 min |
| 26 | infoxlm_c: InfoXLM 512/192, Tamil x2, 2 ep, seed 13; ep1 / final | 0.7024 / 0.7075 | 55 min |
| **27** | **FINAL: equal-weight prob-average of all 14 runs (best epoch per run): xlmr_a,b; muril_a-g; infoxlm_a-c; rembert_a,b** | **0.7381** (Hindi 0.790, Tamil 0.637) | - |
