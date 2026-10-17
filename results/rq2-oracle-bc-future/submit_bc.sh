#!/bin/bash
# rq2-oracle-bc-future steps 2-3 (cluster), after submit_collect.sh's data:
#  2. bc_train.py on one A100 per model (gpu_array.sbatch): arms state / future64 / future128 x BC seeds 0-4 (the main
#     comparison), plus a learning curve (state / future64 x 250 / 500 training seeds x BC seeds 0-2, per-slot scores only)
#  3. rollout.py (slurm/twin_array.sbatch, one CPU, 5 seeds per task) on test seeds 0-39 and fitting seeds 2000-2159,
#     every main checkpoint (15) at every forecast level; starts when the training array has left the queue
#     (afterany; rollout.py --expect 15 fails if a main model is missing)
# Resubmitting resumes: finished models and seeds are left out.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/rq2-oracle-bc-future
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H/code"
rsync -az -e "$RSYNC_SSH" $H/common.py $H/collect.py $H/bc_train.py $H/rollout.py $H/gpu_array.sbatch "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" $H/code/ "$REMOTE:capstone/$H/code/"
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H/out/future_redrawn"
rsync -az -e "$RSYNC_SSH" $H/out/future_redrawn/ "$REMOTE:capstone/$H/out/future_redrawn/"   # bc_train's per-slot check
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
D=results/rq2-oracle-bc-future
[[ $(ls $D/out/labels/s*.npz | wc -l) -eq 1000 && $(ls $D/out/future/s*.npz | wc -l) -eq 200 \
   && $(ls $D/out/future_redrawn/s*.npz | wc -l) -eq 200 ]] \
    || { echo "data incomplete: $(ls $D/out/labels | wc -l) labels, $(ls $D/out/future | wc -l) future tables"; exit 1; }
{
    for arm in state future64 future128; do for k in 0 1 2 3 4; do
        [[ -f $D/bc/scores_${arm}_s$k.json ]] || echo "--arm $arm --bc-seed $k"
    done; done
    for n in 250 500; do for arm in state future64; do for k in 0 1 2; do
        [[ -f $D/bc/scores_${arm}_s${k}_n$n.json ]] || echo "--arm $arm --bc-seed $k --n-train $n --tag _n$n"
    done; done; done
} > $D/tasks_train.txt
for lo in $(seq 0 5 35) $(seq 2000 5 2155); do
    for s in $(seq $lo $((lo + 4))); do
        [[ -f $D/out/rollout/s$s.json ]] || { echo "$lo-$((lo + 4)) --expect 15"; break; }
    done
done > $D/tasks_rollout.txt
cd linux_server; mkdir -p logs
DEP=()
N=$(wc -l < ../$D/tasks_train.txt)
if [[ $N -gt 0 ]]; then
    OUT=$(sbatch --account=drl-scheduling --partition=sporc --time=0-03:00:00 --array=0-$((N - 1)) --job-name=bcfut-train \
        --export=ALL,SCRIPT=$D/bc_train.py,TASKS=$HOME/capstone/$D/tasks_train.txt ../$D/gpu_array.sbatch)
    echo "$OUT"
    DEP=(--dependency=afterany:${OUT##* })
fi
N=$(wc -l < ../$D/tasks_rollout.txt)
if [[ $N -gt 0 ]]; then
    sbatch --account=drl-scheduling --partition=sporc --time=0-03:00:00 --mem=4g --array=0-$((N - 1)) "${DEP[@]}" \
        --job-name=bcfut-roll --export=ALL,SCRIPT=$D/rollout.py,TASKS=$HOME/capstone/$D/tasks_rollout.txt \
        slurm/twin_array.sbatch
fi
REMOTE
