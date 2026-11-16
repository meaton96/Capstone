#!/bin/bash
# train-ft-m100 (10-25, plan §2b, user decision 10-25: fine-tune targets 100 machines + spread 1.0): the final policy
# (train-v5-cmix s0-s4) resumed on the 100-machine floor for 24 updates, a checkpoint per update. One Slurm job per seed
# (16 twin workers, 21 CPUs, MEM 32g as train-size-m100; 16 h limit for ~10 h of training, so the jobs backfill sooner).
# results/scripts/ft_train.sh via submit_dev_cluster.sh; resubmitting skips seeds with DONE_s<seed> on the cluster and
# resumes the others from their latest checkpoint_step*.pt.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$(cat results/train-ft-m100/code/target)" == m100 && "$(cat results/train-ft-m100/code/c_range)" == "1.25 3.5" ]] \
    || { echo "code/target or code/c_range wrong"; exit 1; }
for s in 0 1 2 3 4; do
    MEM=32g JOB_TIME=0-16:00:00 TRAIN_SCRIPT=ft_train.sh bash results/scripts/submit_dev_cluster.sh train-ft-m100 "$s" 16
done
