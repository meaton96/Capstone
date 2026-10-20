#!/bin/bash
# rq3-size-catalog (10-20): local lab run, W tasks at a time (one CPU each), m100 first. Each task plays the 84 catalog
# pairs on one (floor, seed) and checks MDD-PTECT against the grid. Finished seeds are skipped (rerun resumes).
set -uo pipefail
W="${1:-16}"
cd "$(dirname "$0")/../.."
H=results/rq3-size-catalog
mkdir -p $H/logs
.venv/bin/python $H/run.py list > $H/tasks.txt
echo "$(wc -l < $H/tasks.txt) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '.venv/bin/python results/rq3-size-catalog/run.py "$@" > "results/rq3-size-catalog/logs/$2_s$3.log" 2>&1; echo "$2 s$3 rc=$?"' _ < $H/tasks.txt
echo "all tasks ended ($(date -Is))"
