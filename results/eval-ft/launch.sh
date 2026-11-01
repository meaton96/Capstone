#!/bin/bash
# eval-ft (10-25, plan §2b): the fine-tune evaluation of one arm, W tasks at a time locally (task.py; tasks_<arm>.txt):
#   het   train-ft-het budgets 2 / 5 / 10 / 24 on s1 (B2 with speed spread 1) and s0 (B2): 8 tasks, ~20 min each
#   m100  train-ft-m100 budgets on m100 (4 chunks of 10 seeds, ~1.5-2.5 h each) and s0 (B2): 20 tasks
# Finished tasks (out/<cell>/<arm>-b<b>*/DONE) are skipped, so relaunching resumes. Lab: eval-ft-het waits on
# train-ft-het, eval-ft-m100 on train-ft-m100 (its checkpoints pulled from the cluster).
set -uo pipefail
ARM="${1:?usage: launch.sh het|m100 [W]}"
W="${2:-16}"
cd "$(dirname "$0")/../.."
H=results/eval-ft
T=$H/tasks_$ARM.txt
[ -f "$T" ] || { echo "no $T"; exit 1; }
mkdir -p $H/logs
echo "$ARM: $(wc -l < $T) tasks, $W at a time ($(date -Is))"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 xargs -P "$W" -L 1 bash -c \
    '.venv/bin/python results/eval-ft/task.py "$@" > "results/eval-ft/logs/$1_$2_b$3_$4.log" 2>&1; echo "$1 $2 b$3 $4 rc=$?"' _ < $T
n=$(ls $H/out/*/"$ARM"-b*/DONE 2>/dev/null | wc -l)
echo "$ARM: all tasks ended ($(date -Is)): $n / $(wc -l < $T) DONE"
[ "$n" -eq "$(wc -l < $T)" ]
