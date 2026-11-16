#!/bin/bash
# eval-alpha-only-size (10-28, user: the alpha-head-only check): 30 tasks (tasks.txt: m200 / m100 / m50 / m15 x
# {ao-het, ao-b2}, plus the full arms on m15 as a harness check, in the grid's seed chunks, test seeds 0-39, longest
# first), W at a time locally. Finished tasks (out/<floor>/<group>_s<seeds>/DONE) are skipped, so relaunching resumes.
# The references (full arms and fixed alphas on m50-m200, MDD-PTECT, the final policy) are ../eval-alpha-off-size's,
# ../rq3-size-v5's, ../rq3-size-grid's and ../rq3-gen-v5's outputs on the same floors and seeds.
set -uo pipefail
W="${1:-20}"
cd "$(dirname "$0")/../.."
H=results/eval-alpha-only-size
mkdir -p $H/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "$(wc -l < $H/tasks.txt) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '.venv/bin/python results/eval-alpha-only-size/task.py "$@" > "results/eval-alpha-only-size/logs/$1_$2_$3.log" 2>&1; echo "$1 $2 $3 rc=$?"' _ < $H/tasks.txt
n=$(ls $H/out/*/*/DONE 2>/dev/null | wc -l)
echo "all tasks ended ($(date -Is)): $n / $(wc -l < $H/tasks.txt) DONE"
[ "$n" -eq "$(wc -l < $H/tasks.txt)" ]
