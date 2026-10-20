# Random Acts of Pizza — report

## Result
- Final validation: **0.8216 ± 0.003** OOF ROC-AUC. This is 5-fold stratified CV, with the level-2 stage repeated over 10 split seeds; the figure is the mean per-seed AUC of the final rank blend. The seed-averaged OOF gives 0.8234. The best single level-2 model (LightGBM) scores 0.8177.
- Submission (`submission.csv`): rank average of the level-2 LightGBM and LogReg (`stk_final`), each averaged over 10 seeds × 5 folds for test. Giver-known rows are forced to the top.

## Approach and key decisions
- **Only test-time fields are used.** Train has extra `*_at_retrieval` fields, flair, request_text, etc. that are not in test; they were dropped.
- **Validation:** 2878 train rows, test is a random sample with the same date range, so stratified random 5-fold CV is a fair proxy. Every model saves out-of-fold (OOF) predictions; level-2 uses different fold seeds from level-1.
- **`giver_username_if_known`** is given in test. In train it is non-"N/A" only for successful requests (208/208), so it is used as a feature. Those rows already rank at the top; I also force them to the top in post-processing (it changes nothing in OOF).
- **Two-level stack:**
  - Level 1 (text only, each gives an OOF logit): LogReg on frozen embeddings from Qwen2.5-7B/14B/32B-Instruct, Llama-3.1-8B-Instruct, e5-mistral-7b, bge-large and e5-large. The decoder-LLM embeddings are mean-pooled middle/late layers of a prompt-wrapped post. Also: fine-tuned DeBERTa-v3-large; **zero-shot LLM scores** (log-odds of yes/no, or the expected 1–9 rating, for prompts such as "did this person get a pizza?", "is this person genuinely in need?" and "rate 1–9"); and a LogReg on the bag of the requester's subreddits.
  - Level 2: LightGBM (7 leaves, lr 0.01, 500 trees) and L2 LogReg on request-time metadata (account age, karma, RAOP history, time features, text length, lexicon counts for money/family/gratitude/reciprocity/student/job, image links, subreddit SVD) plus the level-1 logits. The final prediction is a rank average.

## What worked
- Metadata alone: ~0.78 (LightGBM 0.783, TF-IDF+meta LogReg 0.791). Adding text through stacking: 0.81. More diverse text sources: 0.818. Zero-shot LLM scores: 0.820. Qwen-32B features: 0.822.
- Frozen LLM embeddings with strong L2 (C ≈ 1e-4) did about as well as or better than fine-tuning: each source scored about 0.65–0.667 AUC text-only, against 0.638 for fine-tuned DeBERTa.
- Zero-shot scores are weak alone (0.52–0.63) but complementary, and the bigger model was better (32B 0.632 vs 7–14B ~0.61).

## What didn't work
- LoRA fine-tuning of Qwen2.5-7B as a classifier (text only): 0.616 OOF, noisy across folds, no gain in the stack. Two launches were also lost to bugs: OOM because the model was in eval mode with gradient checkpointing off, and a random score head that gave huge initial logits.
- CatBoost (0.805) and ExtraTrees (0.810) at level 2 (both below LightGBM). RoBERTa-large fine-tunes (0.639) and extra zero-shot prompt sets (give/story/deserve/proof/reciprocity/quality), phi-4 zero-shot scores and a LogReg over all zero-shot scores: none beat the chosen stack, though they were within noise. Tuning level-2 LightGBM parameters (all within ±0.002). Training level-2 without the giver-known rows (worse). Adding the TF-IDF or meta+embedding models to the final blend gave only noise-level gains.
- Text-only signal saturates at ~0.67 AUC. The non-giver-row AUC of the full model is ~0.74.

## With more time
- Few-shot prompting: 6 in-context train examples raised Qwen-14B's yes/no score from 0.597 to 0.632 AUC alone. It added nothing measurable to the stack, but few-shot prompts on a larger model (the Qwen-32B run hit OOM at the end) and prompt ensembles look promising.
- Larger zero-shot models (e.g. 70B-class; skipped because of disk and time limits). Prompt ensembles with few-shot examples.
- Multi-seed LoRA with better regularisation, or an LLM fine-tune with the metadata included in the prompt.
- Nested CV for the level-1 hyperparameters, to remove the small optimism left by selection.

## Risks: validation vs test
- Mild optimism: the level-1 C values and the choice of stack sources were selected on the same OOF predictions (about 30 stack/blend variants tried, all within 0.818–0.8225; seed-to-seed std ≈ 0.003). The true test AUC may be ~0.005 lower.
- The test set has 79 giver-known rows (6.8%, vs 7.2% in train). If the hidden labels for those rows behave as in train, the giver feature carries over; if not, the score would drop.
- The test set is small (1162 rows), so the test AUC itself has noise of about ±0.015.

## Outside data and pretrained models (no external datasets)
All are public Hugging Face checkpoints, used frozen for embeddings or zero-shot scoring unless noted:
- Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B-Instruct, Qwen/Qwen2.5-32B-Instruct — https://huggingface.co/Qwen (embeddings and zero-shot scores; 7B also LoRA fine-tuned, not used in the final model)
- NousResearch/Meta-Llama-3.1-8B-Instruct — https://huggingface.co/NousResearch/Meta-Llama-3.1-8B-Instruct (embeddings and zero-shot scores)
- microsoft/phi-4 — https://huggingface.co/microsoft/phi-4 (zero-shot scores; experiments only, not in the final stack)
- FacebookAI/roberta-large — https://huggingface.co/FacebookAI/roberta-large (fine-tuned; experiments only, not in the final stack)
- intfloat/e5-mistral-7b-instruct — https://huggingface.co/intfloat/e5-mistral-7b-instruct (embeddings)
- BAAI/bge-large-en-v1.5 — https://huggingface.co/BAAI/bge-large-en-v1.5 (embeddings)
- intfloat/e5-large-v2 — https://huggingface.co/intfloat/e5-large-v2 (embeddings)
- microsoft/deberta-v3-large — https://huggingface.co/microsoft/deberta-v3-large (fine-tuned on the train folds)

Code: `work/scripts/` (common.py, embed.py, llm_feats.py, eval_emb.py, finetune.py, lora.py, stack.py, make_sub.py, ...).

## Housekeeping
- All background jobs have finished or been killed. Extracted features, OOF and test predictions are in `work/feats` and `work/oof`; training curves are in `work/curves`.
- The pretrained LLM / embedding checkpoints were downloaded to the shared `~/.cache/huggingface` early in the run, not to `./work/`. They were deleted at the end (≈136 GB). phi-4 is still in `work/hf`.
