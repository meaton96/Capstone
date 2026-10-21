#!/bin/bash
# rq3-size-portfolio, step 3a (10-20, user): exact regime-map policies at 100 / 50 / 30 / 15 machines, B2 test seeds
# 0-39, local lab run (W at a time). The m15 check (= rq2-rebase-v5's exact map values) runs first. Finished seeds skip.
set -uo pipefail
W="${1:-4}"
cd "$(dirname "$0")/../.."
H=results/rq3-size-portfolio
.venv/bin/python $H/run.py check || { echo "map check failed"; exit 1; }
mkdir -p $H/logs
.venv/bin/python $H/run.py list map > $H/tasks_map.txt
echo "$(wc -l < $H/tasks_map.txt) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '.venv/bin/python results/rq3-size-portfolio/run.py "$@" > "results/rq3-size-portfolio/logs/map_$2_s$3.log" 2>&1; echo "map $2 s$3 rc=$?"' _ < $H/tasks_map.txt
echo "all tasks ended ($(date -Is))"
