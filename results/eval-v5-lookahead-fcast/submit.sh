#!/bin/bash
# eval-v5-lookahead-fcast (10-15, user): the 5 train-v5-lookahead checkpoints with look-ahead features from a perturbed
# published schedule: levels jit1 / jit10 / jit30 / churn10 / churn25 / redrawn x {fitting seeds 2000-2159, test seeds
# 0-39} = 60 twin tasks (twin_array.sbatch). Level "exact" is eval-v5-lookahead. Checkpoints and the frozen code are on
# the cluster; the forecast model is rq2-perturbed-v5/run.py. Resubmitting resumes (tasks with DONE are left out).
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/eval-v5-lookahead-fcast
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H capstone/results/rq2-perturbed-v5"
rsync -az -e "$RSYNC_SSH" $H/run_task.py "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" results/rq2-perturbed-v5/run.py "$REMOTE:capstone/results/rq2-perturbed-v5/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
D=results/eval-v5-lookahead-fcast
for f in results/rq2-rebase-v5/run.py results/rq2-rebase-v5/sim.py results/rq2-rebase-v5/code/env/evaluate.py \
         results/dev-popart-smoke/floor7/des_floor.json; do
    [[ -f $f ]] || { echo "missing $f on the cluster"; exit 1; }
done
for L in jit1 jit10 jit30 churn10 churn25 redrawn; do
    for k in 0 1 2 3 4; do
        r=train-v5-lookahead_s$k
        [[ -f results/$r/checkpoint.pt ]] || { echo "missing $r/checkpoint.pt"; exit 1; }
        [[ -f $D/$L/${r}__checkpoint/DONE ]] || echo "--run $r --level $L"
        [[ -f $D/$L/${r}__checkpoint__test/DONE ]] || echo "--run $r --level $L --split test"
    done
done > $D/tasks.txt
N=$(wc -l < $D/tasks.txt); [[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd linux_server; mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-03:00:00 --mem=6g --array=0-$((N - 1)) --job-name=eval-v5-lafc \
    --export=ALL,SCRIPT=results/eval-v5-lookahead-fcast/run_task.py,TASKS=$HOME/capstone/$D/tasks.txt slurm/twin_array.sbatch
REMOTE
