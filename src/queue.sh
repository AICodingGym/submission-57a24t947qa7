#!/bin/bash
# usage: queue.sh <gpu> <model[:res]> ...   runs extract.py sequentially on one GPU
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/work/hf
g=$1; shift
for spec in "$@"; do
  m=${spec%%:*}; r=""; [[ "$spec" == *:* ]] && r=${spec##*:}
  CUDA_VISIBLE_DEVICES=$g python src/extract.py $m $r > work/logs/ext_${m}_${r}.log 2>&1
done
