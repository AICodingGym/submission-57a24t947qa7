# Tweet Sentiment Extraction: report

**Final:** `submission.csv` is an equal-weight ensemble of 10 soft-target span-extraction runs (7× DeBERTa-v3-large, 2× ELECTRA-large, 1× RoBERTa-large; 5 folds each). It uses label-shift post-processing and expected-Jaccard decoding. **OOF Jaccard 0.7309** over 5-fold CV (neutral 0.977, positive 0.564, negative 0.565). For reference, predicting the full text scores 0.589. Code is in `work/src/`; every experiment is listed in `results.md`.

## Approach and key decisions
- **Task framing:** extractive QA. Input is `sentiment [SEP] tweet`. A linear head predicts start and end logits over the tweet tokens.
- **Validation:** one 5-fold split (StratifiedKFold by sentiment, seed 42) shared by every run, so OOF predictions can be ensembled and scored directly.
- **Training:** 3 epochs, bf16 autocast, AdamW with 10% warmup and linear decay, batch size 32. The checkpoint is chosen by validation Jaccard every half epoch; the best was almost always at epoch 2.5–3.
- **Char-level ensembling:** each run maps its token start/end probabilities onto the first and last non-space character of each token. Runs with different tokenizers are then averaged per character, and the span is decoded at character level. Test predictions average all 5 fold models of every run.
- **Label-noise discovery (biggest insight):** about 11% of training labels start or end mid-word (`'o lonely'`, `'y adore'`, `'ly goo'`). This is systematic. When the text before the annotator's span (a,b) contains `ex >= 2` "extra" spaces (leading spaces plus surplus spaces inside runs of spaces), the published label is `text[a-ex : b-ex+2].strip()`.
  - I invert this to get clean training spans (`pp.clean_label`). Re-applying the forward shift to the clean spans reproduces the train labels at Jaccard 0.9997.
  - Models are trained on clean spans, and the forward shift is applied to predictions (`pp.shift_span`).
  - On OOF, rows with ex=2 go from 0.38 to 0.49 Jaccard, and rows with ex≥3 from 0.15 to 0.44. The test set has the same whitespace statistics as train.
- **Soft targets:** start/end targets are 50% one-hot and 50% a distribution proportional to the Jaccard of the span each alternative boundary would give. This gave +0.002–0.003 OOF for both DeBERTa and ELECTRA, consistently across folds.
- **MBR decoding:** consider the top-30 (start, end) pairs, with the span distribution sharpened by temperature 0.5, and output the candidate with the highest expected Jaccard. This helps diverse ensembles (+0.001) but not single models.

## What worked / didn't (OOF Jaccard)
| change | OOF |
|---|---|
| full-text baseline | 0.589 |
| RoBERTa-large, raw labels (4 folds) | 0.709 |
| RoBERTa-large, clean spans + shift | 0.715 |
| DeBERTa-v3-large, clean + shift (two seeds) | 0.7243 / 0.7239 |
| + soft targets | 0.7268 |
| + lr 1.5e-5 (3 seeds) | 0.7290 / 0.7266 / 0.7265 |
| ELECTRA-large clean → + soft | 0.7185 → 0.7206 |
| final equal-weight ensemble of 10 soft runs + MBR | **0.7309** |

- **Didn't help:** soft weight 0.8 (0.7226, worse than 0.5); lr 2e-5 (no better than 1.5e-5); geometric-mean ensembling (−0.002); MBR at temperature 1 for positive/negative (−0.003).
- **Weak spot:** ensembling gave modest gains overall (~+0.002 over the best single DeBERTa seed).
- **Not used:** greedy weighted ensemble selection reached 0.7314 on OOF. Its weights were fitted on OOF and leaned on one lucky seed, so I used equal weights (0.7309) as the more robust choice.
- **Problems hit:** transformers 5 loads the DeBERTa-v3 checkpoint as fp16, which caused NaN losses; fixed by forcing `dtype=float32`. ELECTRA diverged on 2 of 10 fold runs (Jaccard ~0.2); those folds were rerun with different seeds.

## Risks: validation vs test
- **Checkpoint selection:** each run's checkpoint is chosen on its own validation fold, so OOF is slightly optimistic (about 0.001 judging by best vs last checkpoint). The decoding settings and the choice of which runs to include were also made on OOF.
- **Shift rule:** it is inferred from train labels. If the hidden test labels were produced differently, it would hurt the ~20% of tweets with ≥2 extra spaces. The whitespace statistics match between train and test (19.7% vs 19.6%), and the rule reproduces train labels almost perfectly, so I consider this low risk.
- **Test vs OOF:** test predictions average 5 fold models per run, while OOF uses a single model per row. Test performance should therefore be similar to or slightly better than OOF.

## With more time
- More DeBERTa seeds, and other strong backbones (e.g. deberta-v2-xlarge), for ensemble diversity.
- A character-level second-stage model on top of the token-level probabilities, to recover partial-word boundaries more precisely.
- Pseudo-labelling the test set; multi-sample dropout; layer-wise LR decay; tuning epochs and the soft-target shape per backbone.
- Making ELECTRA training stable (lower LR or a longer warmup).
