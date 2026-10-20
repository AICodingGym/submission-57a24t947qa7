# Experiments (5-fold stratified CV, seed 42, metric = multi-class log loss, lower is better)

| run | what changed | OOF val logloss | time |
|---|---|---|---|
| nb_word | MultinomialNB word 1-2gram counts, alpha 0.03 | 1.10417 | 6s |
| nb_char | MultinomialNB char 1-5gram counts, alpha 0.1 | 3.13941 | 25s |
| lr_word | LogReg C=30 on word 1-2gram TF-IDF (first submission) | 0.39902 | 26s |
| nb_word_tfidf | MNB on word 1-2 TF-IDF (no sublinear), alpha .03 | 0.35136 | 6s |
| nb_char_tfidf | MNB on char 1-5 TF-IDF, alpha .015 | 0.38402 | 37s |
| nb_word13_tfidf | MNB on word 1-3 TF-IDF, alpha .02 | 0.35692 | 14s |
| nb_charwb_tfidf | MNB on char_wb 1-6 TF-IDF | 0.41342 | 32s |
| lr_char | LR C=30 char 1-5 sublinear TF-IDF | 0.39708 | 71s |
| lr_charwb | LR C=30 char_wb 1-6 | 0.40923 | 69s |
| lr_word13 | LR C=30 word 1-3 | 0.42856 | 64s |
| stack_classic | LR meta (C=0.3) on log-probs of 10 sparse models + log length | 0.28394 | 10s |
| deb_l/rob_l/ele_l (v1, killed) | first try; DeBERTa NaN because transformers 5 loads fp16 weights -> forced fp32; best-epoch-by-raw-logloss selection replaced by saving all eval points | rob fold0 0.355 @ep1 | - |
| ele_l | ELECTRA-large disc., lr 1e-5, 3 ep, bs16, maxlen 256; fold 2 collapsed to constant | 0.50685 (T-scaled; ~0.35 on healthy folds) | 35 min |
| deb_l | DeBERTa-v3-large, lr 1e-5, 3 ep, bs16, cosine; best global eval point ep 2.0 | 0.27981 (T-scaled; raw 0.343) | 45 min |
| stack2 | LR meta on 10 sparse + deb_l + ele_l logits | 0.20733 | 10s |
| bert_lc | BERT-large-cased, lr 2e-5, 3 ep | 0.31361 (T-scaled; raw 0.411) | 40 min |
| stack3 | LR meta: sparse + deb_l + ele_l + bert_lc | 0.19896 | 10s |
| stack_lgb (exp) | LightGBM meta on stack2 inputs + 6 count feats (early-stopped on val fold, optimistic) | 0.20608 | 1 min |
| rob_l | RoBERTa-large, lr 1e-5, 3 ep | 0.32966 (T-scaled; raw 0.379) | 60 min (shared GPU) |
| gpt2m | author-conditional GPT-2-medium LM ("EAP:"+text), class = argmax log-lik; 2 ep lr 5e-5 | 0.38601 (raw softmax of log-liks) | 75 min (shared GPU) |
| stack4 | LR meta: sparse + deb_l + bert_lc + rob_l + gpt2m (electra dropped) | 0.19359 | 10s |
| qwen7 | Qwen2.5-7B LoRA r16 (all linear) seq-cls, bs16, 2 ep, maxlen 256, grad-ckpt; folds 0,1,4 lr 1e-4 seed0, folds 2,3 lr 7e-5 seed1 (fold 2 at 1e-4 diverged early, rerun) | 0.22821 (T-scaled; raw 0.313) | ~45-60 min/fold |
| stack5 | LR meta: stack4 inputs + qwen7 | 0.17022 | 10s |
| mis7 | Mistral-7B-v0.3 LoRA r16 seq-cls, lr 5e-5, 2 ep, bs16 (same recipe as qwen7) | 0.21028 (T-scaled; raw 0.348) | ~40-75 min/fold |
| stack6 | LR meta: stack5 inputs + mis7 | 0.16060 | 10s |
| gpt2l | author-conditional GPT-2-large LM, 2 ep lr 3e-5 | 0.34285 (raw) | ~2 h (shared GPU) |
| stack7 | stack6 inputs + gpt2l | 0.15795 | 10s |
| qwen3 | Qwen3-8B-Base LoRA r16 seq-cls, lr 7e-5, 2 ep | 0.23785 (T-scaled; raw 0.309) | ~45 min/fold |
| stack8 | stack7 inputs + qwen3 | 0.15748 | 10s |
| qw15lm | author-conditional Qwen2.5-1.5B LM (full FT, 1 ep, lr 3e-5) log-lik features | 0.28423 (raw) | ~10 min/fold |
| stack9 | stack8 inputs + qw15lm | 0.15016 | 10s |
| stack9 + len-norm LM feats | extra LL/sqrt(n_words) features | 0.15025 (no gain, dropped) | 10s |
| final9 | 0.8 * LR stack + 0.2 * LightGBM stack (fixed 350 rounds) | 0.14967 | 1 min |
| deb_l2 | DeBERTa-v3-large seed 1 (same recipe as deb_l, but with model.train() fixed from step 0) | 0.29952 (T-scaled) | 45 min (2 GPUs) |
| qw3lm | author-conditional Qwen2.5-3B LM, LoRA r32, 1 ep, lr 1e-4 | 0.27995 (raw) | ~25 min/fold (shared GPUs) |
| stack10 | stack9 inputs + qw3lm (LR only) | 0.14651 | 10s |
| stack10 + deb_l2 | adding 2nd DeBERTa seed | 0.14660 (no gain, dropped) | 10s |
| final10 | 0.8 * LR stack10 + 0.2 * LGB stack10 (fixed 350 rounds) -> submission.csv | 0.14591 | 1 min |
| qw3lm2 | 2nd Qwen2.5-3B conditional LM: seed 1, bs 32, lr 2e-4, 1 ep | 0.28182 (raw) | ~14 min/fold (4 GPUs in parallel) |
| stack11 | stack10 inputs + qw3lm2 as separate features (LR only) | 0.14587 | 10s |
| stack11b | qw3lm and qw3lm2 log-liks averaged into one feature block (LR only) | 0.14579 | 10s |
| final11b | 0.7 * LR stack11b + 0.3 * LGB stack11b (350 rounds); probability clipping tested, no gain | 0.14514 | 1 min |
| qw15lm2 | 2nd Qwen2.5-1.5B conditional LM: seed 1, bs 32, lr 6e-5, 1 ep | 0.29457 (raw) | ~7 min/fold (4 GPUs) |
| stack12 | stack11b with the qw15lm/qw15lm2 log-liks averaged into one block (LR only) | 0.14538 | 10s |
| **FINAL** | 0.8 * LR stack12 + 0.2 * LGB stack12 (350 rounds) -> submission.csv | **0.14492** | 1 min |

Notes: curves in work/curves use step = fold*100 + epoch. For tfm.py runs val_metric is the per-fold temperature-scaled log loss; for lmcls.py runs it is the raw log loss of softmax(log-liks); for the sparse models step = fold index.
