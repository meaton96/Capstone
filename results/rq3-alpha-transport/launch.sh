#!/bin/bash
# rq3-alpha-transport (10-28, user: size / transport term in the alpha law): stage 1, local lab run, W tasks at a time
# (one CPU each), longest first (run.py list). Each task plays MDD-PTECT at every alpha (curve) or at alpha 3 (util) on
# one (cell, seed). Finished seeds are skipped, so relaunching resumes.
set -uo pipefail
W="${1:-20}"
cd "$(dirname "$0")/../.."
H=results/rq3-alpha-transport
mkdir -p $H/logs
.venv/bin/python $H/run.py list > $H/tasks.txt
echo "$(wc -l < $H/tasks.txt) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '.venv/bin/python results/rq3-alpha-transport/run.py "$@" > "results/rq3-alpha-transport/logs/$1_s$2_$3.log" 2>&1; echo "$1 s$2 $3 rc=$?"' _ < $H/tasks.txt
echo "all tasks ended ($(date -Is))"
