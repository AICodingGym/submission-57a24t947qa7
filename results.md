# Experiments (validation = random 5% holdout of train, fold 0, 90,255 rows; competition bias metric)

| exp | what | val score | time |
|---|---|---|---|
| tfidf | TF-IDF word 1-2gram (300k) + LogReg, 600k-row subsample, binary target | 0.8864 | 1.7 min |
| rb_smoke | roberta-base, 32k samples, weighted soft-BCE + aux subtype heads | 0.9159 | 1.3 min |
| db1 | deberta-v3-base, full train, lr 2e-5, bf16 autocast | 0.4999 at 25% (collapsed to constant output) -> killed | 15 min wasted |
| rb1 | roberta-base, full train (1.71M) 1 epoch, bs64, lr2e-5, maxlen256, identity weights + aux heads | 0.9450 | 35 min |
| bb1 | bert-base-uncased, same recipe as rb1 | 0.9455 | 35 min |
| ens1 | rank-average rb1+bb1 (equal-ish weights); tfidf gets ~0 weight and hurts | 0.9467 | - |
| db2 | deberta-v3-base retry, lr 1e-5, 1.2M | 0.5000 at 12.5% (collapsed again; suspect deberta-v2 + bf16 autocast issue in this transformers version) -> killed | 6 min |
| el1 | electra-base-discriminator (bert tokens), lr 3e-5, full train 1 epoch | 0.9448 | 35 min |
| ens2 | rank-avg rb1+bb1+el1 | 0.9468 | - |
| rl1 | roberta-large, full train 1 epoch, lr 1e-5 | 0.9482 | 90 min |
| ens3 | rank-avg rb1+bb1+el1+rl1 (opt weights shrunk 20% to equal) | 0.9488 | - |
| rl2 | roberta-large seed 1, 1.5M samples | 0.9480 | 82 min |
| bl1 | bert-large-uncased-wwm, 950k samples, lr 1.5e-5 | 0.9445 | 52 min |
| ens4 | rank-avg of 6 models (rb1,bb1,el1,rl1,rl2,bl1); equal weights 0.9484 | 0.9490 | - |
| rb2 | roberta-base seed 1, full train | 0.9449 | 35 min |
| **ens5 (final)** | rank-avg of 7 models; weights tuned on holdout (Nelder-Mead) shrunk 20% to equal: rl1 .275, rl2 .272, bb1 .236, el1 .092, rb1 .067, rb2 .029, bl1 .029. Equal weights: 0.9482. Half-split CV of the weight tuning (tune on one half, score on the other): 0.9486 tuned vs 0.9481 equal | 0.9489 | - |
