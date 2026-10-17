# Jigsaw Unintended Bias: Report

**Final:** `submission.csv` is a rank-weighted ensemble of 7 fine-tuned transformers. Holdout score is **0.9489**; a half-split cross-check of the weight tuning gives 0.9486. The TF-IDF baseline scored 0.886 and the best single model (roberta-large) 0.9482.

## Approach and key decisions
- **Validation:** a random 5% holdout of train.csv (fold 0, 90,255 rows). The competition metric is reimplemented in `src/metric.py`: 0.25·overall AUC + 0.25·each power mean (p=-5) of the subgroup, BPSN and BNSP AUCs over the 9 evaluated identities, with identity >= 0.5. All models train on the other 95%.
- **Models** (`src/train.py`): pretrained transformers fine-tuned for 1 epoch, one model per GPU in parallel.
  - Inputs are pre-tokenized once per vocabulary (`src/tok.py`); bert, electra and bert-large share the uncased WordPiece vocab, and both roberta sizes share one BPE vocab. Head truncation to 256 tokens.
  - Batches are length-bucketed with dynamic padding. Training uses bf16 autocast, AdamW, 5% warmup then linear decay, and grad clipping at 1.0.
  - Every run has a wall-clock deadline guard so it always finishes and writes predictions.
- **Head:** (CLS + masked-mean pooling)/2 -> linear layer with 8 outputs (main target, auxiliary target, 6 toxicity subtypes).
- **Loss:** BCE on the soft target (fraction of raters) for the main output, multiplied by identity-aware sample weights: `w = 1 + [identity] + [toxic & no identity] + 2·[non-toxic & identity]`, normalized to mean 1. This targets BPSN, the weakest part of the baseline. Auxiliary subtype BCE is added with weight 0.5. Only the main output is used for prediction.
- **Ensemble** (`src/ensemble.py`): predictions are rank-normalized, then weighted-averaged. Weights come from Nelder-Mead on the holdout metric and are then shrunk 20% toward equal weights.

| model | samples | holdout |
|---|---|---|
| roberta-large (lr 1e-5) | 1.71M | 0.9482 |
| roberta-large seed 1 | 1.5M | 0.9480 |
| bert-base-uncased | 1.71M | 0.9455 |
| roberta-base (2 seeds) | 1.71M | 0.9450 / 0.9449 |
| electra-base | 1.71M | 0.9448 |
| bert-large-uncased-wwm | 0.95M | 0.9445 |
| **ensemble (weights rl1 .275, rl2 .272, bb1 .236, el1 .092, rb1 .067, rb2 .029, bl1 .029)** | | **0.9489** (equal weights 0.9482) |

## What worked / what didn't
- Transformers beat TF-IDF by a wide margin: 0.886 -> 0.945 for a base model after one epoch. The biggest gains came in BPSN (0.80 -> 0.94) and subgroup AUC.
- roberta-large beats the base models by about +0.003.
- Ensembling different backbones adds about +0.0007 over the best single model. Adding a second seed of the same architecture adds little.
- TF-IDF hurts the transformer ensemble even at small weight, so it gets weight 0.
- The weakest submetric is still subgroup AUC (about 0.915), driven by homosexual_gay_or_lesbian, black, white and muslim.
- deberta-v3-base collapsed to constant predictions twice (lr 2e-5 and 1e-5) under bf16 autocast with transformers 5.x. I dropped it rather than spend time debugging it.

## What I'd try with more time
- Train 2 epochs (scores were still rising at the end of epoch 1), use more roberta-large seeds, and retrain on 100% of the data.
- Fix deberta-v3 (fp32, or check the attention implementation). Add other model families such as xlnet or gpt2 for diversity.
- Tune the identity weighting scheme and the auxiliary loss weight, and try head+tail truncation for long comments.
- Use the multi-rater structure more (e.g., `toxicity_annotator_count`-based weighting).

## Risks: holdout vs test
- The holdout is a random split of the same distribution as test, so the score should transfer. However, the per-identity holdout subsets are small, and the p=-5 power mean is driven by the worst subgroup, so expect noise around ±0.002.
- Ensemble weights were tuned on the same holdout, which makes the reported score slightly optimistic. Half-split CV (tune on one half, score on the other) gives 0.9486 vs 0.9481 for equal weights, so the optimism is about 0.0003.
- The test set's identity labels may be distributed a bit differently from train's (the description says "similar, but not exact").
