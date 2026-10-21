#!/bin/bash
# fit zs/res heads for every finished feature dir that has no preds yet
cd "$(dirname "$0")/.."
g=${1:-1}
for d in work/feats/*/; do
  t=$(basename $d)
  [ -f $d/train_squash_flip_feat.npy ] || continue
  ls work/preds/${t}__*_oof.npy >/dev/null 2>&1 && continue
  [ -f $d/train_squash_flip_logit.npy ] || { CUDA_VISIBLE_DEVICES=$g python src/heads.py $t --mode lr --wd 1e-4 2>&1 | grep RESULT; continue; }
  CUDA_VISIBLE_DEVICES=$g python src/heads.py $t --mode zs 2>&1 | grep RESULT
  for wd in 3e-3 1e-2 3e-2; do CUDA_VISIBLE_DEVICES=$g python src/heads.py $t --mode res --wd $wd 2>&1 | grep RESULT; done
done
