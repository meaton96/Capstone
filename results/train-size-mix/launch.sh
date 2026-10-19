#!/bin/bash
# train-size-mix (10-19, user: floor-size study): the final policy's stack (train-v5-cmix) trained on floor size(s) 15,50,100, 5 seeds,
# one Slurm job per seed (12 twin workers, MEM 32g). results/scripts/size_train_cluster.sh via submit_dev_cluster.sh;
# resubmitting skips seeds with DONE_s<seed> on the cluster.
set -euo pipefail
cd "$(dirname "$0")/../.."
for s in 0 1 2 3 4; do
    MEM=32g TRAIN_SCRIPT=size_train_cluster.sh bash results/scripts/submit_dev_cluster.sh train-size-mix "$s" 12 \
        --prior-init offset --prior-pair "" --num-envs 12
done
