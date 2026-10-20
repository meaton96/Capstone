#!/bin/bash
# rq3-size-oracle (10-20): local lab run, W tasks at a time (one CPU each): the reproduction check per floor, then the
# oracle on m100 and m50 (B2 test seeds 0-39; m100 first, it is the longer). Finished seeds are skipped (rerun resumes).
set -uo pipefail
W="${1:-16}"
cd "$(dirname "$0")/../.."
H=results/rq3-size-oracle
PY=.venv/bin/python
for c in m50 m100; do $PY $H/run.py check $c || { echo "reproduction check failed on $c"; exit 1; }; done
mkdir -p $H/logs
$PY $H/run.py list > $H/tasks.txt
echo "$(wc -l < $H/tasks.txt) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '.venv/bin/python results/rq3-size-oracle/run.py "$@" > "results/rq3-size-oracle/logs/$2_s$3.log" 2>&1; echo "$2 s$3 rc=$?"' _ < $H/tasks.txt
echo "all tasks ended ($(date -Is))"
