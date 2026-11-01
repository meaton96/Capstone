#!/bin/bash
# rq3-size-catalog (10-20): local lab run, W tasks at a time (one CPU each), m100 first. Each task plays the 84 catalog
# pairs on one (floor, seed) and checks MDD-PTECT against the grid. Finished seeds are skipped (rerun resumes).
set -uo pipefail
W="${1:-16}"; shift || true      # further arguments: floors (default m100 m50 m30)
cd "$(dirname "$0")/../.."
H=results/rq3-size-catalog
mkdir -p $H/logs
TASKS=$H/tasks${1:+_$(IFS=_; echo "$*")}.txt
.venv/bin/python $H/run.py list "$@" > $TASKS
echo "$(wc -l < $TASKS) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '.venv/bin/python results/rq3-size-catalog/run.py "$@" > "results/rq3-size-catalog/logs/$2_s$3.log" 2>&1; echo "$2 s$3 rc=$?"' _ < $TASKS
echo "all tasks ended ($(date -Is))"
