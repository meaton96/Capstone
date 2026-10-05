#!/bin/bash
# train-due-twin (10-05, user: train on the twin first): PPO on the event-based twin, 3 seeds x 3M steps, one A100 each.
# Regime from rq2-twin-fleet / dev-load-calib: 21,600 s (6 h) agent windows, regime blocks of 5,400 s (c ~ U[1.75, 2.5],
# op mean, load profile normal 0.6-1.2 : surge 1.0-1.6 : short-fleet (2 AGVs on duty) = 2:1:1), failures off (twin),
# tardiness reward, obs v3 / H15 heads / PopArt. Runs `slurm/submit_train_twin.sh` on the cluster over the shared SSH
# connection (resubmitting resumes from the latest checkpoint). Outputs ~/capstone/results/train-due-twin_s<seed>.
set -euo pipefail
source "$(dirname "$0")/../../linux_server/slurm/rit_ssh.sh"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
cd ~/capstone/linux_server_v3
export RUN_NAME=train-due-twin TOTAL_TIMESTEPS=3000000 WINDOW=21600 SEEDS=3 TIME=0-10:00:00 GPU=0   # A100 queue backed up 10-05
export PARAMS='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400, "horizon_seconds": 30600, "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2}, {"name": "surge", "utilization": [1.0, 1.6]}, {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}'
slurm/submit_train_twin.sh
REMOTE
