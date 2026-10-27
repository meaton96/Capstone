#!/bin/bash
# eval-alpha-offset (10-25, user: "the law sets alpha, the head picks an offset"): 10 tasks (tasks.txt: 5 spread cells x
# {off-het, off-b2} on test seeds 0-39 through evaluate.py), W at a time locally. Finished tasks
# (out/<cell>/DONE_<group>_<seeds>) are skipped, so relaunching resumes. Waits on train-alpha-off-het / -off-b2. The
# references (final / het / b2 / grid) are ../eval-alpha-spread's outputs on the same cells and seeds.
set -uo pipefail
W="${1:-10}"
cd "$(dirname "$0")/../.."
H=results/eval-alpha-offset
mkdir -p $H/logs
echo "$(wc -l < $H/tasks.txt) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '.venv/bin/python results/eval-alpha-offset/task.py "$@" > "results/eval-alpha-offset/logs/s$1_$2_$3.log" 2>&1; echo "s$1 $2 $3 rc=$?"' _ < $H/tasks.txt
n=$(ls $H/out/*/DONE_* 2>/dev/null | wc -l)
echo "all tasks ended ($(date -Is)): $n / $(wc -l < $H/tasks.txt) DONE"
[ "$n" -eq "$(wc -l < $H/tasks.txt)" ]
