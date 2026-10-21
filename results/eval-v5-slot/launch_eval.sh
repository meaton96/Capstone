#!/bin/bash
# eval-v5-slot (10-21): local lab run, one task per slot arm (9), all in parallel (one CPU each). Finished arms
# (base/slot<X>/DONE) are skipped by run_task, so relaunching resumes.
set -uo pipefail
cd "$(dirname "$0")/../.."
H=results/eval-v5-slot
mkdir -p $H/logs
for X in 60 120 180 300 600 900 1200 1500 1800; do
    .venv/bin/python $H/task.py --slot $X > $H/logs/slot$X.log 2>&1 &
done
wait
echo "all arms ended ($(date -Is)): $(ls $H/base/*/DONE 2>/dev/null | wc -l) / 9 DONE"
