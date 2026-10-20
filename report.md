# Spooky Author Identification — report

**Final validation: 0.14492 multi-class log loss** (5-fold OOF, lower is better). The best single model is Mistral-7B LoRA at 0.210 (temperature-scaled). For comparison, the sparse-only stack scores 0.284 and the uniform prior about 1.09.

## Approach
- **Validation:** a single fixed 5-fold stratified split (seed 42, `work/folds.npy`) shared by every model, so all out-of-fold (OOF) predictions are stackable. The score is log loss over the full 17,621-row OOF.
- **Level-1 models.** Each produces 5-fold OOF predictions plus test predictions averaged over the folds.
  1. Sparse: Multinomial NB and logistic regression on word 1-2/1-3-gram and char 1-5 / char_wb 1-6 TF-IDF or count features (10 models, each 0.35–0.43 alone, about 0.284 when stacked).
  2. Fine-tuned encoders (lr 1e-5–2e-5, 3 epochs, bs 16): DeBERTa-v3-large (0.280), RoBERTa-large (0.330), BERT-large-cased (0.314). A second DeBERTa seed (0.300) was trained but added nothing to the stack, so it was left out. ELECTRA-large was dropped because one fold collapsed to a constant prediction twice.
  3. **LoRA-fine-tuned 7-8B decoder LLMs as 3-way sequence classifiers** (r=16 on all linear layers, bf16, gradient checkpointing, 2 epochs, bs 16): Qwen2.5-7B (0.228), Mistral-7B-v0.3 (0.210), Qwen3-8B-Base (0.238). These were the biggest gain.
  4. **Author-conditional causal LMs as generative classifiers.** Fine-tune an LM on `"EAP:" + sentence`, `"HPL:" + sentence`, etc. Then score every held-out sentence under each of the 3 author prefixes, and use the 3 summed log-likelihoods as features. Models: GPT-2-medium (0.386), GPT-2-large (0.343), Qwen2.5-1.5B full FT (two seeds, 0.284 and 0.295, averaged), two Qwen2.5-3B LoRA runs (0.280 and 0.282; their log-likelihoods are averaged into one feature block). They are weaker alone but very complementary: adding Qwen-1.5B-LM moved the stack from 0.1575 to 0.1502.
- **Checkpoint choice for transformers:** logits are saved every quarter epoch. After epoch 1 the models become very overconfident, so raw log loss gets worse while temperature-scaled log loss keeps improving. For each model, a single eval point (the same for all folds) is chosen by temperature-scaled OOF log loss; in practice that was the end of training.
- **Level 2:** multinomial logistic regression (C=0.03) on the centred log-probabilities/logits of all level-1 models plus log text length, cross-validated on the same 5 folds. The final submission is 0.8 × LR stack + 0.2 × LightGBM stack (same inputs plus 6 simple count features, fixed 350 rounds). The 0.2 weight was picked on OOF: LR alone 0.14538 → 0.14492. The final stack has 20 level-1 input blocks from 22 trained models.

## What worked / what didn't
- Worked: LLM LoRA classifiers (0.199 → 0.161 in the stack); conditional-LM likelihood features (+0.004 from the GPT-2s, +0.0075 from Qwen-1.5B, +0.004 from the Qwen-3B runs); simple LR stacking; temperature-aware checkpoint selection.
- Didn't help or was dropped: ELECTRA-large (unstable); length-normalised LM features; LightGBM as the main meta-model (0.155 vs 0.150 for LR); averaging the logits of several checkpoints; a second DeBERTa seed; clipping the final probabilities.
- Issues hit along the way: transformers 5.x loads DeBERTa-v3 in fp16 by default, which gave NaNs until fp32 was forced. LoRA at lr 1e-4 diverged early on one Qwen2.5 fold, so folds 2-3 were rerun at lr 7e-5 / seed 1; the per-fold hyper-parameters therefore differ slightly.

## Outside data / models
All are public pretrained checkpoints from the Hugging Face Hub. They were used only as initialisation for fine-tuning on the competition train set. No external text data was used, and no source books were downloaded.
- microsoft/deberta-v3-large — https://huggingface.co/microsoft/deberta-v3-large (classifier)
- FacebookAI/roberta-large — https://huggingface.co/FacebookAI/roberta-large (classifier)
- google-bert/bert-large-cased — https://huggingface.co/google-bert/bert-large-cased (classifier)
- google/electra-large-discriminator — https://huggingface.co/google/electra-large-discriminator (tried, dropped)
- Qwen/Qwen2.5-7B — https://huggingface.co/Qwen/Qwen2.5-7B (LoRA classifier)
- mistralai/Mistral-7B-v0.3 — https://huggingface.co/mistralai/Mistral-7B-v0.3 (LoRA classifier)
- Qwen/Qwen3-8B-Base — https://huggingface.co/Qwen/Qwen3-8B-Base (LoRA classifier)
- openai-community/gpt2-medium, openai-community/gpt2-large — https://huggingface.co/openai-community (conditional LM)
- Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-3B — https://huggingface.co/Qwen (conditional LM)

## Risks: validation vs test
- Checkpoint selection and the temperature/meta-model are fit on the same OOF predictions used for scoring. This is only a few parameters per model, so the optimism should be small (maybe ~0.002). The LR stack itself is properly cross-validated.
- Large pretrained LMs have probably seen these public-domain books during pretraining. This helps on both validation and test in the same way, because the test set is a random split of the same data, so it should not bias the val/test gap.
- Test predictions average 5 fold-models while OOF comes from single fold-models, so the test may be slightly better calibrated than OOF suggests. That is a small, favourable shift.

## With more time
- More seeds / full-data refits of the 7-8B LoRA classifiers; larger LLMs (14B+); a larger conditional LM (7B) for the generative features.
- Per-author LMs instead of a shared prefix-conditioned LM; token-level likelihood-ratio features.
- Tuning LoRA rank/lr per model and a stronger meta-model with nested CV.

## Housekeeping
No jobs are left running. Downloaded checkpoints are in `work/hf_cache` (about 52 GB; safe to delete). The older encoders were already in the shared HF cache. Scripts are in `work/scripts`, OOF/test predictions in `work/oof`, and logs in `work/logs`. To rebuild the submission: `cd work/scripts && python final.py <21 names listed in results.md>`.
