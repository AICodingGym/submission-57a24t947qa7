# AI4Code: report

**Final validation Kendall tau: 0.8844** (grouped holdout, 5975 notebooks). The heuristic baseline scored 0.563. The submission is in `submission.csv` (20000 notebooks; every row checked to be a permutation of its notebook's cells).

## Approach
- **Model:** a notebook-level model, one forward pass per notebook.
  - Each cell is encoded separately by `microsoft/unixcoder-base` (mean-pooled; markdown ≤80 tokens, code ≤64 tokens; long notebooks get shorter limits so each notebook stays under ~12k tokens).
  - A 3-layer transformer then runs over all cell embeddings of the notebook. Code cells get position info (their known order); markdown cells get none, so the model treats them as an unordered set.
  - I picked this over the common "one markdown cell + sampled code context" pointwise model because every markdown cell sees the whole notebook, and it costs fewer tokens per epoch (about 33 min per epoch on 3× A6000).
- **Two heads:**
  1. Regression (L1) of each markdown cell's position in "code-index space": a markdown cell in gap *k* (after *k* code cells) gets a target in (k, k+1)/(n_code+1).
  2. A pointer head: a softmax over [code cells, END] that predicts which code cell comes right after the markdown cell.
- **Decoding:** each markdown cell goes in the **median** gap of its pointer distribution, which minimizes the expected number of code/markdown inversions. Markdown cells in the same gap are ordered by the regression output.
- **Training:** DDP, bf16, AdamW (encoder 3e-5, heads 2e-4), cosine schedule with 3% warmup, batches built by token budget.
  - Run A: 6 epochs.
  - B: 1 more epoch from A at lr 1e-5, with the pointer loss weighted 3× higher.
  - C: 0.8 more epoch from B at lr 8e-6, with a new seed.
- **Final prediction:** the average of A-ep6, B and C (both the regression values and the pointer probabilities), then median decoding.
- **Validation:** grouped holdout by `ancestor_id` (md5(ancestor) % 20 == 0, 5975 notebooks), never used for training. This matches the test setup, where no test notebook shares an ancestor with train.

## External data / models
- `microsoft/unixcoder-base`, https://huggingface.co/microsoft/unixcoder-base. A pretrained code/NL encoder used to initialize the cell encoder. No other external data was used.

## What worked / what didn't
- **Median-gap decoding of the pointer head** was the best decoder:
  - +0.007–0.01 over regression only, and +0.01 over pointer argmax.
  - Mixing the regression and pointer outputs (mean position, or using the pointer only when it is confident) landed in between.
  - Mixing a reg-centred distribution into the pointer before taking the median gave +0.0003. That is noise, so I didn't use it.
- **More epochs** helped steadily: 0.849 → 0.865 → 0.874 → 0.880 → 0.883 → 0.883 for epochs 1–6.
- **The extra low-LR fine-tunes plus averaging** added about +0.0013.
- **Averaging consecutive epochs of the same run** gave no gain.

## With more time
- Longer markdown context: pad markdown and code cells separately so longer markdown doesn't inflate padding.
- A larger encoder (e.g., a deberta-v3 or code-specific large model), and more epochs (still improving at epoch 6).
- Independently trained seeds or encoders for a real ensemble.
- An explicit markdown-markdown pairwise loss for ordering cells within a gap.

## Risks
- The validation split is grouped by ancestor, like the test set, and comes from the same notebook pool, so the score should transfer well.
- The decoder and ensemble were chosen on the same 5975 validation notebooks. Any optimism from that is small, because the choices were few and coarse.
- Text is truncated to 800 characters and 64/80 tokens per cell. That is the same for train and test, so it is not a source of mismatch.
