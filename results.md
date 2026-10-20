# Experiments (validation = OOF ROC-AUC, 5-fold stratified CV, mean over 3 seeds unless noted)

| # | experiment | val AUC | time |
|---|---|---|---|
| 1 | lgb_meta: LightGBM on request-time metadata + simple text stats + giver_known | 0.7829 | ~45s |
| 2 | lr_tfidf_meta: LogReg on word(1-2)+char(2-5) TF-IDF + log-scaled meta | 0.7911 | ~55s |
| 3 | rank-average of 1+2 (first submission) | 0.7975 | - |
| 4 | lr_e5l_emb: LogReg on e5-large-v2 mean-pooled embedding + meta | 0.7903 | ~1 min |
| 5 | lr_bgel_emb: LogReg on bge-large-en-v1.5 CLS embedding + meta | 0.7953 | ~1 min |
| 6 | lgb_meta_sub: LGBM meta + subreddit SVD(16) | 0.7783 | ~9 min (CPU contention) |
| 7 | lgb_meta_sub_bge: + bge PCA(32) | 0.7920 | ~1.5 min |
| 8 | text-only LogReg on frozen embeddings: e5 0.628, bge 0.638, Qwen2.5-7B-Instruct (prompted, layer 28 mean) 0.660, Qwen2.5-14B-Instruct (layer 36 mean) 0.662 | 0.662 best | ~5 min extraction each |
| 9 | debl_s0: DeBERTa-v3-large fine-tuned text-only, 3 ep, lr 1e-5, 1 seed | 0.6382 | 18 min |
| 10 | stk_a: level-2 LGBM / LogReg on meta + subreddit SVD + text-only OOF logits (q14, q7, deberta, bge); rank-avg of lgb+lr | 0.8103 / 0.8048 / 0.8127 | ~1 min |
| 11 | stk_b: level-2 with 7 text sources (q14 mean36/last36, q7 mean28/last28, deberta, bge, e5) lgb / lr / rank-avg | 0.8126 / 0.8120 / 0.8175 | ~1 min |
| 12 | stk_c: level-2 with only q14 mean36 text source (ablation) | 0.8099 (rank-avg) | ~1 min |
| - | Qwen2.5-7B LoRA attempts 1-2 failed (OOM: model left in eval mode so no grad-checkpointing; dtype mismatch in predict; huge init logits from random score head) | - | ~35 min lost |
| 13 | text-only frozen emb: Llama-3.1-8B-Instruct (Nous mirror) best layer16 mean 0.663; e5-mistral-7b-instruct layer16 mean 0.664 | 0.664 | ~10 min each |
| 14 | stk_d: stack with 11 text sources (+Llama, e5-mistral) | 0.8184 | ~1 min |
| 15 | q7lora_s0: Qwen2.5-7B-Instruct LoRA r16, 2 ep, lr 1e-4, text-only | 0.6161 (fold AUCs 0.59-0.66) | ~35 min on 4 GPUs |
| 16 | stk_e: stk_d + q7lora | 0.8184 (no gain) | ~1 min |
| 17 | level-2 LGBM param grid (leaves 4-15, lr .005-.01) | 0.8151-0.8171 (default best) | ~1 min |
| 18 | concatenated 5 embeddings text-only LogReg | 0.6683 | 3 min |
| 19 | level-2 trained excluding giver-known rows | 0.8178 (vs 0.8184; worse) | 1 min |
| 20 | lr_subs: LogReg on bag of subreddits | 0.5851 | 1 min |
| 21 | zero-shot LLM yes/no or 1-9 rating scores (Llama-8B, Qwen-14B; 3 prompts) | 0.597-0.616 each | ~6 min/prompt |
| 22 | stk_g: stk_d sources + lr_subs + 5 zero-shot scores | lgb 0.8147 / lr 0.8157 / avg 0.8201 | 1 min |
| 23 | CatBoost level-2 (depth 4, 1500 it) | 0.8054 (worse than LGBM) | 7.5 min |
| 24 | more zero-shot scores: Qwen-7B/14B/Llama prompt set 2 (give/story/deserve), phi-4; Qwen2.5-32B-Instruct (succ 0.632, rate 0.622, need 0.615) | 0.519-0.632 each | 6-30 min/model |
| 25 | zs_combo: LogReg on all 23 zero-shot scores (text-only) | 0.6502 | 1 min |
| 26 | Qwen-32B prompted embeddings text-only (rate prompt, layer 48 mean) | 0.6667 | (same run) |
| 27 | stk_k: stk_g + Qwen-32B zero-shot scores + 2 Qwen-32B embedding LRs | lgb 0.8176 / lr 0.8149 / avg 0.8217 | 1 min |
| 28 | stk_m: stk_g sources + zs_combo + 2 Qwen-32B embedding LRs | avg 0.8216 | 1 min |
| 29 | rank-avg of stk_k + stk_m (lgb+lr each) -> submission | see final | - |
| 30 | DeBERTa-v3-large seeds 1,2 (0.6286, 0.6300); 3-seed avg debl_avg | 0.6428 | 16 min each |
| 31 | prompt set 3 (proof / reciprocity / quality 1-9) on Qwen-14B, Llama-8B; Qwen-32B set 2 (give/story/deserve) | 0.559-0.627 each | 6-26 min |
| 32 | stk_n / stk_o / stk_q: + extra zero-shot sources or zs_combo | 0.8217 / 0.8220 / 0.8220 | 1 min |
| 33 | stk_p: stk_k sources with debl_avg instead of debl_s0 -> lgb 0.8183 / lr 0.8159 / rank-avg 0.8225 (submission) | 0.8225 | 1 min |
| 34 | RoBERTa-large fine-tune text-only, seeds 0/1 (0.6388/0.6372), 2-seed avg | 0.6467 | 8 min each |
| 35 | stk_r / stk_s: stk_p + robl_avg (+ q32 quality zero-shot) | 0.8222 / 0.8217 (no gain) | 1 min |
| 36 | **FINAL stk_final**: stk_p sources, level-2 over 10 split seeds; lgb 0.8177, lr 0.8172 (per-seed means); rank-avg per-seed 0.8216 ± 0.0029 (seed-averaged OOF 0.8234) | **0.8216** | 1 min |
| 37 | level-2 ExtraTrees (1000 trees, leaf 10) / LGBM extra_trees=True; added to final blend | 0.8100 / 0.8169; blend 0.8218 / 0.8227 vs 0.8234 (no gain, not used) | 1 min |
| 38 | q32emb_avg (avg of 9 Qwen-32B prompt-embedding LRs, 0.6662) replacing 2 q32 LRs in stack | per-seed 0.8208 vs 0.8216 (not used) | 2 min |
| 39 | few-shot (6 train examples in prompt) Qwen-14B yes/no score: 0.6324 alone (vs 0.597 zero-shot); stk_u = stk_final + it | per-seed 0.8217 vs 0.8216 (noise; not used) | 8 min on 2 GPUs |
| 40 | few-shot Qwen-32B: OOM on longest batches at the end of the run, no time to rerun | - | 30 min |
