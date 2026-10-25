#!/bin/bash
# rq3-size-headroom (10-25): local lab run, W tasks at a time (one CPU each). The reproduction / job-stream check per
# floor, then the descriptor runs (fast) and the oracles, m200 first (the longest). Finished outputs are skipped
# (rerun resumes); r 0.52 outputs that rq3-size-oracle / rq3-size-portfolio already hold are not in the task list.
# One process per task, 8 GB address-space cap each (the rq3-alpha-size leak froze the machine 10-23).
set -uo pipefail
W="${1:-20}"
cd "$(dirname "$0")/../.."
H=results/rq3-size-headroom
PY=.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for m in 50 100 200; do $PY $H/run.py check $m || { echo "check failed on m$m"; exit 1; }; done
mkdir -p $H/logs
$PY $H/run.py list > $H/tasks.txt
echo "$(wc -l < $H/tasks.txt) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c 'ulimit -v 8000000; log="results/rq3-size-headroom/logs/$(echo "$@" | tr " " "_").log"; .venv/bin/python results/rq3-size-headroom/run.py "$@" > "$log" 2>&1; echo "$* rc=$?"' _ < $H/tasks.txt
echo "all tasks ended ($(date -Is))"
n=$(ls $H/out/*/*/*/s*.json 2>/dev/null | wc -l)
echo "$n output files"
