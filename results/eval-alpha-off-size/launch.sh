#!/bin/bash
# eval-alpha-off-size (10-26, user: the law + offset arms on the size grid): 39 tasks (tasks.txt: m200 / m100 / m50 x
# {off-b2, off-het, alpha} in the grid's seed chunks, test seeds 0-39, longest first), W at a time locally. Finished
# tasks (out/<floor>/<group>_s<seeds>/DONE) are skipped, so relaunching resumes. The references (MDD-PTECT, v5 pairs,
# MDD-TECT, the final policy, T100) are ../rq3-size-v5's and ../rq3-size-grid's outputs on the same floors and seeds.
set -uo pipefail
W="${1:-20}"
cd "$(dirname "$0")/../.."
H=results/eval-alpha-off-size
mkdir -p $H/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "$(wc -l < $H/tasks.txt) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '.venv/bin/python results/eval-alpha-off-size/task.py "$@" > "results/eval-alpha-off-size/logs/$1_$2_$3.log" 2>&1; echo "$1 $2 $3 rc=$?"' _ < $H/tasks.txt
n=$(ls $H/out/*/*/DONE 2>/dev/null | wc -l)
echo "all tasks ended ($(date -Is)): $n / $(wc -l < $H/tasks.txt) DONE"
[ "$n" -eq "$(wc -l < $H/tasks.txt)" ]
