#!/bin/bash
# train-v5-cmix (10-15 relaunch): one Slurm job per seed (7 CPUs each), because the 5-seed 27-CPU job had an estimated
# start two days out. Same stack and frozen code (6d13412a in code/) as train-v5-nokl; due-date range c ~ U[1.25 3.5]
# (code/c_range).
# results/scripts/dev_b2_slot_train_eval_cluster_c.sh trains only (code/train_only); the evaluation is a separate lab
# experiment. Resubmitting skips seeds with DONE_s<seed> on the cluster.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$(cat results/train-v5-cmix/code/c_range)" == "1.25 3.5" && -f results/train-v5-cmix/code/train_only ]] || { echo "code/c_range or train_only wrong"; exit 1; }
for s in 0 1 2 3 4; do
    MEM=16g TRAIN_SCRIPT=dev_b2_slot_train_eval_cluster_c.sh bash results/scripts/submit_dev_cluster.sh train-v5-cmix "$s" 2 \
        --prior-init offset --discount-horizon-s 5400 --shared-baseline-group 4 --rollout-length 128 --batch-size 256 \
        --prior-pair "" --prior-kl-coef 0
done
