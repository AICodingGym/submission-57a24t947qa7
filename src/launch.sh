#!/bin/bash
# usage: src/launch.sh <gpu> <name> [train.py args...]
cd "$(dirname "$0")/.."
G=$1; N=$2; shift 2
CUDA_VISIBLE_DEVICES=$G nohup python src/train.py --name $N "$@" > work/logs/$N.log 2>&1 &
echo $! >> work/pids.txt
echo "launched $N on gpu $G pid $!"
