#!/bin/bash
cd "$(dirname "$0")/.."
g=$1; f=$2
while read t; do
  [ -f work/feats/$t/train_squash_flip_logit.npy ] || continue
  for wd in 3e-3 1e-2; do CUDA_VISIBLE_DEVICES=$g python src/heads.py $t --mode res --wd $wd 2>&1 | grep RESULT; done
done < $f
echo REFIT_DONE
