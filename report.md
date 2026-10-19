# chaii Hindi/Tamil QA: report

**Final:** `submission.csv` is an equal-weight ensemble of 14 five-fold runs (4 model families). Out-of-fold word-level Jaccard is **0.7381** (Hindi 0.790, Tamil 0.637). The best single run scores 0.711.

## Approach
- **Task framing:** extractive QA. Each long Wikipedia context (median 7.5k chars) is split into sliding windows (384 or 512 tokens; stride 128 or 192). The models are trained with start/end cross-entropy. Negative windows (no answer) are subsampled to 30% per epoch, and their target is the CLS token.
- **Validation:** 5-fold CV on the 1002 train rows, stratified by language, with fixed folds for every model. The score is word-level Jaccard on out-of-fold (OOF) predictions over all 1002 rows. Test predictions average the 5 fold models.
- **Two-stage training:**
  - Stage 1: MuRIL-large, RemBERT and InfoXLM-large first trained 1 epoch on SQuAD v1.1 (English) plus MLQA-hi and XQuAD-hi.
  - XLM-R-large used the public SQuAD2 checkpoint (deepset) instead of stage 1.
  - Stage 2: per-fold fine-tuning on the chaii training folds plus MLQA-hi/XQuAD-hi, for 2–3 epochs. A prediction is saved after every epoch.
- **Char-level outputs:** every model's token logits are mapped onto context characters, using the max over overlapping windows. Token offsets are whitespace-trimmed, because RemBERT offsets include the leading space. This lets models with different tokenizers (SentencePiece XLM-R/InfoXLM, RemBERT, WordPiece MuRIL) be ensembled directly.
- **Ensembling:** take a per-model, per-example softmax over candidate start/end positions, average the probabilities, then decode the best span on log-probabilities. On OOF this was clearly better than averaging raw logits (0.7326 vs 0.7263 on the same 5 members).
- **Decoding and post-processing** (each choice validated on OOF):
  - Answers are capped at 80 characters (+0.004).
  - Spans may not cross a newline.
  - Unbalanced brackets or quotes at the span ends are stripped (+0.008).
  - Stripping trailing `.`/`,`/`।` hurt (answers like "एस. पी. जी." end with a period), so it is not used.

## Outside data and models
| name | URL | use |
|---|---|---|
| deepset/xlm-roberta-large-squad2 | https://huggingface.co/deepset/xlm-roberta-large-squad2 | initial weights (SQuAD 2.0 English QA) |
| microsoft/infoxlm-large | https://huggingface.co/microsoft/infoxlm-large | pretrained encoder |
| google/muril-large-cased | https://huggingface.co/google/muril-large-cased | pretrained encoder |
| google/rembert | https://huggingface.co/google/rembert | pretrained encoder |
| MLQA v1 (hi context / hi question, dev+test, 5.4k) | https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip | extra Hindi QA training data |
| XQuAD hi (1190) | https://huggingface.co/datasets/google/xquad | extra Hindi QA training data |
| SQuAD v1.1 train (87.6k) | https://huggingface.co/datasets/rajpurkar/squad | stage-1 QA training for MuRIL / RemBERT / InfoXLM |

None of these is derived from chaii data. No chaii-finetuned checkpoint or chaii mirror was used.

## What worked / what didn't
- **Ensembling diverse models gave the largest gain.** Single runs score 0.686–0.711 OOF. Two runs reach 0.714, five reach 0.733 (probability averaging) and 7–14 reach 0.737–0.743.
  - Beyond about 7 members the score plateaus. Leave-one-out changes are ±0.007, which is within noise.
- **Member selection:** I tested greedy (Caruana) selection honestly with nested CV (select on 4 folds, score the 5th). It reached 0.727, below equal weighting of all runs at 0.7365. Its in-sample 0.749 was overfit. The final ensemble therefore weights all 14 runs equally, using each run's best epoch.
- **MuRIL-large** (after the SQuAD+Hindi stage 1) was the best single family. Using 512-token windows helped it: 0.711 vs 0.704.
- **InfoXLM-large** after stage 1 (0.706) beat the public XLM-R SQuAD2 checkpoint (0.686–0.700).
- **Epochs:** after stage-1 pretraining, 1–2 epochs on chaii is enough. A third epoch usually lowered OOF.
- **Tamil is much harder than Hindi** (≈0.62 vs ≈0.79 OOF). Oversampling Tamil training rows ×3 did not improve the single model's Tamil score, but the run still added diversity to the ensemble.
- **RemBERT** was the slowest model and the weakest single model here (0.687). It still adds a little to the ensemble.

## What I'd try with more time
- More Tamil training data, such as SQuAD machine-translated into Tamil.
- Larger models (XLM-R XL, mT5-XL encoders).
- More seeds and full-data (no-holdout) refits.
- Smarter handling of the noisy "first occurrence" labels.
- Learned answer-length and number-format post-processing (Devanagari vs Arabic digits).

## Risks: validation vs test
- **Label quality differs.** Test labels come from 3 annotators with majority voting and are cleaner than the single-annotator train labels. Absolute test scores may therefore be higher than OOF, but the ranking of choices should hold.
- **Test set is small.** It has only 112 rows (84 Hindi, 28 Tamil), so the test score will have high variance (±0.03–0.04).
- **Mild selection bias.** The best epoch per run and the post-processing rules were chosen on the same OOF data, which biases the OOF score upward slightly.
  - The same ensemble built from final checkpoints only scores 0.736 (9-run version), so the bias is likely under 0.005–0.007.
  - Weights are equal, and greedy weight search was not used.
- **Test-time model differs from validation.** Test predictions average 5 fold models, which is somewhat stronger than any single fold model used for OOF.
