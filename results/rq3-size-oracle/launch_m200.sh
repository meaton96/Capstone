#!/bin/bash
# rq3-size-oracle, 200 machines (10-20, user: added after the 50 / 100-machine oracle and the catalog screen): local lab
# run, W tasks at a time. Reproduction check on m200 first, then the oracle on B2 test seeds 0-39 (~2 h per seed:
# 144 episodes of ~50 s). Finished seeds are skipped (rerun resumes).
set -uo pipefail
W="${1:-16}"
cd "$(dirname "$0")/../.."
H=results/rq3-size-oracle
PY=.venv/bin/python
$PY $H/run.py check m200 || { echo "reproduction check failed on m200"; exit 1; }
mkdir -p $H/logs
$PY $H/run.py list m200 > $H/tasks_m200.txt
echo "$(wc -l < $H/tasks_m200.txt) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '.venv/bin/python results/rq3-size-oracle/run.py "$@" > "results/rq3-size-oracle/logs/$2_s$3.log" 2>&1; echo "$2 s$3 rc=$?"' _ < $H/tasks_m200.txt
echo "all tasks ended ($(date -Is))"
