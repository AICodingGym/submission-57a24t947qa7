#!/bin/bash
# wait for effb3 fold $1 to finish, then launch second architecture on the same GPU if there is time left
d=$1
until grep -q -E "^done|Traceback" work/logs_train/effb3_f$d.log; do sleep 10; done
if [ $(date +%H%M) -lt 1150 ]; then
  python src/train.py --fold $d --epochs 14 --enc tu-tf_efficientnetv2_s.in21k_ft_in1k --name v2s_f$d --dev $d > work/logs_train/v2s_f$d.log 2>&1
fi
