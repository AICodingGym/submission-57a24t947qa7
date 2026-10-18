# Task: tweet-sentiment-extraction

You are working alone on a machine-learning competition. No human will answer questions during this run, so make reasonable decisions yourself and write down why.

- **Problem description:** `data/description.md`. Everything you need is under `data/`.
- **Goal:** the best score you can get on the hidden test set, using the competition's metric.
- **Final output:** `./submission.csv`, in exactly the format of the sample submission (same ids, same columns).

## Budget and environment
- **Time:** hard stop at **2026-10-18 08:59** (server time). Check the clock with `date` regularly.
  - Write a valid `submission.csv` early (within the first ~15% of the time), then overwrite it whenever you have something better.
  - Create `report.md` and `work/final.json` early as drafts and keep them updated as you go.
  - Start wrapping up by **2026-10-18 08:32**: no new long training runs after that. Finalize `submission.csv`, `report.md`, `results.md` and `work/final.json`.
  - Keep each training run short enough to finish before the wrap-up time.
- **GPU:** use only the GPU(s) listed in `CUDA_VISIBLE_DEVICES` (there may be more than one). Don't change that setting or use other GPUs.
  - This is a shared lab server, so keep RAM and disk use reasonable.
- **Python:** the conda env `gym` is active. You may `pip install` packages and download freely available pretrained weights.
- **Background processes:** run training in the foreground, or wait for it to finish. Nothing you start should keep running after you stop.

## Rules
- Don't search for, read or copy existing solutions, notebooks or write-ups for this competition. General library documentation and pretrained models are fine.
- Don't run any `aicodinggym` command. A human submits after you finish.
- Work only inside this folder, and put large files (features, checkpoints, caches) under `./work/`.
- Don't create Claude Docs or artifacts. All notes go in local files.

## Required outputs
- `./submission.csv`: the best predictions you have.
- `./results.md`: one line per experiment, with what changed, the validation score, and the time taken.
- `./work/final.json` must contain:
  ```json
  {"metric": "...", "higher_is_better": true, "validation_score": 0.0,
   "validation_method": "...", "approach": "...", "elapsed_hours": 0.0}
  ```
- `./report.md`, kept short, covering:
  - your approach and key decisions
  - what worked and what didn't
  - what you'd try with more time
  - any risk that the validation score doesn't reflect the test score
