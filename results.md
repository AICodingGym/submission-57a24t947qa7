# Experiments
| # | change | val Kendall tau | time |
|---|---|---|---|
| 0 | heuristic: md cells spread evenly between code cells | 0.5633 | 1 min |
| A-ep1 | unixcoder-base cell encoder + 3-layer notebook transformer, reg(L1)+pointer heads, epoch 1/6 (median-gap decoding) | 0.8490 (reg 0.8386, ptr argmax 0.8361) | 33 min |
| A-ep2 | same run, epoch 2/6, median decoding | 0.8654 | 68 min |
| A-ep3 | epoch 3/6, median decoding (ep2+ep3 avg: 0.8742) | 0.8743 | 103 min |
| A-ep4 | epoch 4/6, median decoding | 0.8799 | 138 min |
| A-ep5 | epoch 5/6, median decoding (ep4+ep5 avg: 0.8823, no gain) | 0.8826 | 173 min |
| A-ep6 | epoch 6/6 (final of run A), median decoding; ep5+ep6 avg 0.8831; medm0.15 decoding 0.8833 (noise, not used) | 0.8831 | 208 min |
| B-ft | fine-tune A-ep6 for 1 more epoch (lr 1e-5, pointer loss weight x3, new seed) | 0.8836 | 34 min |
| A6+B | average of A-ep6 and B predictions (reg + pointer probs), median decoding | 0.8841 | - |
| C-ft | fine-tune B for 0.8 epoch (lr 8e-6, seed 11) | 0.8833 | 27 min |
| **A6+B+C** | **average of three checkpoints, median decoding (FINAL SUBMISSION)** | **0.8844** | - |
