#!/bin/bash
# rq2-oracle-bc-future data scaling (10-16, user), after submit_collect4k.sh: the BC arms retrained on 4,000 training
# seeds (3000-6999) -> bc4k/, then their rollouts -> out/rollout4k/. Same scripts and recipe as submit_bc.sh.
#  2. bc_train.py on one A100 per model: state / future64 / future128 x BC seeds 0-4 on all 4,000 seeds, plus the
#     learning-curve point at 2,000 seeds (3000-4999) for the same arms x BC seeds 0-2 (per-slot scores only)
#  3. rollout.py --bc-dir bc4k (one CPU, 5 seeds per task) on test 0-39 and fitting 2000-2159, the 15 main
#     checkpoints at every forecast level; starts when the training array has left the queue (afterany; --expect 15)
# Resubmitting resumes: finished models and seeds are left out.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/rq2-oracle-bc-future
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H/code capstone/$H/out/future_redrawn"
rsync -az -e "$RSYNC_SSH" $H/common.py $H/collect.py $H/bc_train.py $H/rollout.py $H/gpu_array.sbatch "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" $H/code/ "$REMOTE:capstone/$H/code/"
rsync -az -e "$RSYNC_SSH" $H/out/future_redrawn/ "$REMOTE:capstone/$H/out/future_redrawn/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
D=results/rq2-oracle-bc-future
[[ $(ls $D/out/labels/s*.npz | wc -l) -eq 4000 && $(ls $D/out/future/s*.npz | wc -l) -eq 200 \
   && $(ls $D/out/future_redrawn/s*.npz | wc -l) -eq 200 ]] \
    || { echo "data incomplete: $(ls $D/out/labels | wc -l) labels, $(ls $D/out/future | wc -l) future tables"; exit 1; }
T="--train-seeds 3000-6999 --out-dir $HOME/capstone/$D/bc4k"
{
    for arm in state future64 future128; do for k in 0 1 2 3 4; do
        [[ -f $D/bc4k/scores_${arm}_s$k.json ]] || echo "--arm $arm --bc-seed $k $T"
    done; done
    for arm in state future64 future128; do for k in 0 1 2; do
        [[ -f $D/bc4k/scores_${arm}_s${k}_n2000.json ]] || echo "--arm $arm --bc-seed $k $T --n-train 2000 --tag _n2000"
    done; done
} > $D/tasks_train4k.txt
for lo in $(seq 0 5 35) $(seq 2000 5 2155); do
    for s in $(seq $lo $((lo + 4))); do
        [[ -f $D/out/rollout4k/s$s.json ]] || { echo "$lo-$((lo + 4)) --bc-dir bc4k --out $HOME/capstone/$D/out/rollout4k --expect 15"; break; }
    done
done > $D/tasks_rollout4k.txt
cd linux_server; mkdir -p logs
DEP=()
N=$(wc -l < ../$D/tasks_train4k.txt)
if [[ $N -gt 0 ]]; then
    OUT=$(sbatch --account=drl-scheduling --partition=sporc --time=0-04:00:00 --mem=48g --array=0-$((N - 1)) \
        --job-name=bcfut-train4k \
        --export=ALL,SCRIPT=$D/bc_train.py,TASKS=$HOME/capstone/$D/tasks_train4k.txt ../$D/gpu_array.sbatch)
    echo "$OUT"
    DEP=(--dependency=afterany:${OUT##* })
fi
N=$(wc -l < ../$D/tasks_rollout4k.txt)
if [[ $N -gt 0 ]]; then
    sbatch --account=drl-scheduling --partition=sporc --time=0-03:00:00 --mem=4g --array=0-$((N - 1)) "${DEP[@]}" \
        --job-name=bcfut-roll4k --export=ALL,SCRIPT=$D/rollout.py,TASKS=$HOME/capstone/$D/tasks_rollout4k.txt \
        slurm/twin_array.sbatch
fi
REMOTE
