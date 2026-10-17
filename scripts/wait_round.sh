#!/bin/bash
# wait until given runs have test_all.npy (or any log has a traceback), then print last log lines
cd /data/yidan/gym/tensorflow-speech-recognition-challenge
while true; do
  ok=1
  for r in "$@"; do [ -f work/runs/$r/test_all.npy ] || ok=0; done
  [ $ok = 1 ] && break
  grep -q Traceback work/logs_runs/*.log && break
  sleep 15
done
echo DONE
for r in "$@"; do tail -n 1 work/logs_runs/$r.log; done
date
