# NFL Player Contact Detection — report

## Approach
1. **Candidate rows**: all ground (`G`) rows plus player pairs with tracking distance ≤ 3 yd (keeps 57,830 of 57,891 train positives, ~18% of rows). All other rows are predicted 0.
2. **Tabular features** (`scripts/feats.py`, 194 feats): per-player tracking (speed/accel/sa, lags/leads, rolling stats, orientation/direction sin/cos, direction changes), pair features (distance and its lags/leads/rolling min, relative velocity, closing speed, facing, same team, neighbor counts), and baseline-helmet features per view (box size, normalized helmet-center distance between the pair plus its lags, vertical helmet motion, aspect ratio).
3. **Video CNN** (`scripts/extract_crops.py`, `scripts/train_cnn.py`): for each candidate row and view (Endzone/Sideline) where the baseline helmets are present, a grayscale crop around the pair (or around the player's body for ground contacts) is cut at 3 frames (−6, 0, +6 video frames). A 4th channel holds Gaussian blobs at the pair's helmet positions, so the net knows which players to judge. These go into ImageNet-pretrained resnet18 / efficientnet_b0 (timm), with is_ground and view concatenated into the head, trained 2 epochs with flip/brightness/shift augmentation done on the GPU. The CNNs are trained out-of-fold by game_play (2-fold at first, 4-fold for the final models).
4. **Stacking**: per contact_id, the CNN probabilities per view (E/S/max/mean, plus lags/leads and rolling mean/max over ±1–4 steps) are added to the tabular features. XGBoost (GPU) is trained with 5-fold GroupKFold by game_play.
5. **Post-processing**: the probabilities of a few GBDT variants are averaged, smoothed with a centered rolling mean over 3 steps within each pair, and thresholded with one global threshold tuned on OOF MCC.

## What worked
- Tracking + helmet GBDT alone: 0.728 OOF MCC.
- Stacking 2-fold CNN features: 0.778–0.782. **4-fold CNNs** (more training data per model) did better: 0.785 raw for one model, 0.788 smoothed.
- A higher input resolution (80px crops upsampled to 160) and a bigger backbone (efficientnet_b2) gave the best CNNs (per crop-view AUC ~0.955).
- **Final: 0.7916 OOF MCC**. This is the mean of 4 XGBoost stackers built on 4-fold CNN features (eb0 res128, r18 res160, eb0 res160, eb2 res160), smoothed over 3 steps, with threshold 0.395. (MCC is computed over all 4.26M train rows. Elapsed time is about 2.1 h.)
- Larger-context 80px crops (more of the body, which matters for ground contact) beat the 64px crops.
- Smoothing over 3 steps: about +0.003. Blending GBDT variants: about +0.001–0.002.

## What didn't
- 5-frame (±0.2 s) crops: the CNN was worse and overfit; no gain in the stack.
- Adding more and more correlated CNN feature sets to one GBDT: no gain. Neither did cross-pair context features or a lower GBDT learning rate (each within ±0.0005).
- Separate thresholds for ground vs. player pairs: no gain.

## With more time
- More CNN folds or epochs with stronger augmentation, bigger backbones, a higher crop resolution, and fusing tracking features directly into the CNN; using the All29 view.
- Train a 3D/temporal model over longer clips; seed-averaging.

## Risks: validation vs. test
- The threshold and blend weights were tuned on the same OOF data (a mild optimistic bias, probably <0.005).
- At test time the CNN features are averages over the fold models, while training used single-model OOF predictions. So the test features are slightly better and differently calibrated than the ones the GBDT learned on. This usually helps a little or is neutral, but it can shift the best threshold.
- The test plays are a different set of plays/games; label noise (±1 step) limits the achievable MCC equally in both.

## Reproduce
`scripts/feats.py train|test` → `scripts/base_rows.py train` → `S=80 BIG=1 PFX=b80_ scripts/extract_crops.py train|test` → `scripts/train_cnn.py --nfolds 4 --fold k --pfx b80_ ...` (tags eb0b80k4, r18b80k4 res160, eb0b80r160k4 res160, eb2b80k4 res160) → `scripts/cnn_feats.py <tag or tag1+tag2...>` → `scripts/train_gbdt.py <name> work/cnnfeat_...` (k4d, k4e, k4f, k4g; see the log lines in results.md) → `scripts/make_sub.py k4d,k4e,k4f,k4g 3`.
