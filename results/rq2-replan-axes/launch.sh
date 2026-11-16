#!/bin/bash
# rq2-replan-axes (10-29): local lab run of one task set (run.py task_sets: check | proc | reg | axis1 | axis1-ext | size | fleet | axis6 | axis2), W tasks at a time
# (one CPU each), costliest first (run.py est_cost: re-plans x the level's re-plan cost). Finished (variant, seed)
# outputs are skipped, so relaunching resumes. Usage: bash launch.sh axis1 14
set -uo pipefail
SET="${1:-axis1}"
W="${2:-14}"
cd "$(dirname "$0")/../.."
H=results/rq2-replan-axes
mkdir -p $H/logs
.venv/bin/python $H/run.py list "$SET" > $H/tasks_$SET.txt   # costliest first (run.py orders them)
echo "$(wc -l < $H/tasks_$SET.txt) tasks ($SET), $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c 'nice -n 10 .venv/bin/python results/rq2-replan-axes/run.py "$@" > "results/rq2-replan-axes/logs/$1_s$2.log" 2>&1; echo "$1 s$2 rc=$?"' _ < $H/tasks_$SET.txt
echo "all tasks ended ($(date -Is))"
