# Experiments (validation = 5% of train molecules held out, fixed seed 1234; metric = mean over 8 types of log(MAE), lower is better)

| run | change | val score | time |
|---|---|---|---|
| B_L12 | 12 layers, bs64 (killed at ep6 for being slow) | -2.165 @ep6 (A: -2.09 @ep6) | 15 min |
| A_base | 8 layers dn256 de128 bs64 lr5e-4 (killed at ep8, dominated by bs128) | -2.271 @ep8 | 15 min |
| C_bs128 | 8 layers dn256 de128 bs128 lr7e-4, 84 epochs (1.5h) | -3.0347 | 90 min |
| D_L12_bs128_long | 12 layers dn256 de128 bs128 lr5e-4, 168 epochs | -3.1724 | 4.1 h |
| E_L10_de192_long | 10 layers dn256 de192 bs128 lr5e-4, 165 epochs | -3.2405 | 4.0 h |
| F_L10_de192_s3 | same as E, seed 3, 121 epochs | -3.1905 | 3.0 h |
| blend C+D+E+F (mean) | plain average | -3.3742 | - |
| blend C+D+E+F (per-type weights) | greedy per-type weights fit on the validation set (optimistic) | -3.3951 | - |
| D_wr / E_wr / F_wr | warm restart from D/E/F final ckpt, lr 1.5e-4 cosine, 12-15 epochs (~15-23 min) | -3.1740 / -3.2440 / -3.1947 | 0.3 h each |
| blend all 7 (mean) | plain average of C, D, E, F, D_wr, E_wr, F_wr | -3.3956 | - |
| blend all 7 (per-type weights) | greedy per-type weights fit on validation (optimistic by ~0.01) | -3.4028 | - |
| C_wr / E_wr2 / F_wr2 | second short warm restarts (C from C, E_wr2 from E_wr, F_wr2 from F_wr), 12 min | -3.0439 / -3.2468 / -3.1986 | 0.2 h each |
| blend all 10 (mean) | plain average | -3.3933 | - |
| **blend all 10 (per-type weights) = submission** | greedy per-type weights fit on validation (optimistic by ~0.01) | **-3.4063** | - |
