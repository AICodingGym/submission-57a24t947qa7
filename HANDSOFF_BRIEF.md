# Task: champs-scalar-coupling

You are working alone on a machine-learning competition. No human will answer questions during this run, so make reasonable decisions yourself and write down why.

- **Problem description:** `data/description.md`. The competition data is under `data/`.
- **Goal:** the best score you can get on the hidden test set, using the competition's metric.
- **Final output:** `./submission.csv`, in exactly the format of the sample submission (same ids, same columns).

## Budget and environment
- **Time:** hard stop at **2026-10-19 09:35** (server time). Check the clock with `date` regularly.
  - Write a valid `submission.csv` early (within the first ~15% of the time), then overwrite it whenever you have something better.
  - Create `report.md` and `work/final.json` early as drafts and keep them updated as you go.
  - Start wrapping up by **2026-10-19 08:41**: no new long training runs after that. Finalize `submission.csv`, `report.md`, `results.md` and `work/final.json`.
  - Plan every training run so it finishes before the wrap-up time.
  - **Use the whole budget.** Don't stop early: until the wrap-up time, keep running experiments that could improve the validation score (more training, more seeds, better models, ensembling).
- **GPU:** use only the GPU(s) listed in `CUDA_VISIBLE_DEVICES` (there may be more than one). Don't change that setting or use other GPUs.
  - This is a shared lab server, so keep RAM and disk use reasonable.
- **Python:** the conda env `gym` is active. You may `pip install` packages.

## Long jobs and checkpoints
A single tool call can time out, and a job that dies before it saves anything is lost. So:
- Start any job longer than about 10 minutes with `nohup ... > work/logs/<name>.log 2>&1 &`, append its PID to `work/pids.txt`, and check on it with `tail`, `sleep` and `nvidia-smi`. Don't wrap long jobs in a `timeout` shorter than the job.
- Make every training script save a checkpoint and its validation/test predictions periodically (at least every ~20 minutes and at the end), so a killed job still leaves usable output.
- Before you stop, make sure every job you started has finished or been killed. Nothing you start should keep running after you stop.

## Outside data and models
This follows the original competition's rules: external data and pre-trained models were allowed, provided they were disclosed (see the timeline in `data/description.md`). Here, disclosure means listing them in `report.md`.

Also don't use any outside source of scalar coupling constants for these molecules.

In all cases:
- Don't download this competition's original data (from Kaggle or any mirror or copy), or any dataset or model built from it, such as checkpoints fine-tuned on it. The hidden test set here was carved out of that data, so using it would leak the answers.
- Don't search for, read or copy existing solutions, notebooks, discussion posts or write-ups for this competition. General library documentation is fine.
- List every outside dataset or pretrained model you use in `report.md` (name, URL, what you used it for).

## Other rules
- Don't run any `aicodinggym` command. A human submits after you finish.
- This is a fresh attempt. Don't look at git history or at files from earlier attempts.
- Work only inside this folder, and put large files (data you download, features, checkpoints, caches) under `./work/`.
- Don't create Claude Docs or artifacts. All notes go in local files.

## Required outputs
- `./submission.csv`: the best predictions you have.
- `./results.md`: one line per experiment, with what changed, the validation score, and the time taken.
- `./work/final.json` must contain:
  ```json
  {"metric": "...", "higher_is_better": true, "validation_score": 0.0,
   "validation_method": "...", "approach": "...", "elapsed_hours": 0.0,
   "external_data": ["..."]}
  ```
- `./report.md`, kept short, covering:
  - your approach and key decisions
  - outside data and models used
  - what worked and what didn't
  - what you'd try with more time
  - any risk that the validation score doesn't reflect the test score
