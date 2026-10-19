#!/bin/bash
# usage: nohup scripts/queue.sh <gpu_index> <jobs_file> &   -- runs each line "<name> <train.py args>" in order.
# Lines are read lazily (one at a time by line number), so jobs can be appended/edited while the queue runs.
gpu=$1; jobs=$2; n=1
while true; do
  line=$(sed -n "${n}p" "$jobs")
  [ -z "$line" ] && break
  n=$((n+1))
  [[ "$line" == \#* ]] && continue
  name=$(echo $line | cut -d' ' -f1); rest=$(echo $line | cut -d' ' -f2-)
  echo "$(date) start $name" >> work/logs/queue_$gpu.log
  python scripts/train.py --gpu $gpu --name $name $rest > work/logs/${name}_$(basename $jobs .txt)_$((n-1)).log 2>&1
  echo "$(date) end $name" >> work/logs/queue_$gpu.log
done
