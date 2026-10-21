#!/bin/bash
# rq3-alpha-size (10-21, user: paper B gate): local lab run, W tasks at a time (one CPU each), largest floors first
# (run.py list orders m100 -> m15, held-out cells last). Each task plays 18 configs on one (cell, seed). Finished
# seeds are skipped, so relaunching resumes.
set -uo pipefail
W="${1:-16}"
cd "$(dirname "$0")/../.."
H=results/rq3-alpha-size
mkdir -p $H/logs
.venv/bin/python $H/run.py list > $H/tasks.txt
echo "$(wc -l < $H/tasks.txt) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '.venv/bin/python results/rq3-alpha-size/run.py "$@" > "results/rq3-alpha-size/logs/$1_s$2.log" 2>&1; echo "$1 s$2 rc=$?"' _ < $H/tasks.txt
echo "all tasks ended ($(date -Is))"
