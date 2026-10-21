#!/bin/bash
# train-v5-slot60 (10-21, user: PPO slot-length sweep, results/eval-v5-slot/README.md): the final policy's recipe
# (train-v5-cmix: frozen code 6d13412a, c ~ U[1.25, 3.5], 16 envs, shared baseline groups of 4, 5,400 s discount
# horizon, entropy 0.01 -> 0.001) with --slot-seconds 60 and 720 updates (1474560 agent steps, the same simulated
# experience as cmix's 48 updates at 900 s). Seeds 0-2 in parallel, locally, 2 twin workers each. Resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE=cpu PY=.venv/bin/python bash results/scripts/dev_b2_slot_train_eval_cluster_c.sh train-v5-slot60 "0 1 2" 2 \
    --prior-init offset --discount-horizon-s 5400 --shared-baseline-group 4 --rollout-length 128 --batch-size 256 \
    --prior-pair "" --prior-kl-coef 0 --slot-seconds 60 --total-timesteps 1474560
