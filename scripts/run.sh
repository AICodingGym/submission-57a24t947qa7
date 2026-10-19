#!/bin/bash
# usage: scripts/run.sh <gpu_index_within_CUDA_VISIBLE_DEVICES> <name> [train.py args...]
gpu=$1; name=$2; shift 2
nohup python scripts/train.py --gpu $gpu --name $name "$@" > work/logs/${name}.log 2>&1 &
echo $! >> work/pids.txt
