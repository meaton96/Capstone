#!/bin/bash
# Submit (or resubmit, which resumes) a training run. Run on the cluster from ~/capstone/linux_server.
# Required: ACCOUNT RUN_NAME TOTAL_TIMESTEPS. Optional: SEEDS (array size, default 3), TRAIN_SEED_BASE (0),
# PARTITION (sporc), TIME (4-00:00:00), NUM_ENVS (6), SAVE_EVERY (25 updates), ENT_COEF, ENT_COEF_FINAL,
# EPISODE_SECONDS (5400), STALL_MINUTES (45), DRY=1 prints the command.
# Notifications: NOTIFY_SLACK=@<slack user> adds RIT's Slurm-to-Slack notices (start, end, fail, 80% of the time
# limit) for every seed; progress / stall / crash messages come from train.py via ~/.capstone_webhook (env/notify.py).
# Example:
#   ACCOUNT=drl-scheduling RUN_NAME=rnd01 TOTAL_TIMESTEPS=3000000 slurm/submit_train.sh
set -euo pipefail
: "${ACCOUNT:?}" "${RUN_NAME:?}" "${TOTAL_TIMESTEPS:?}"
SEEDS="${SEEDS:-3}"
mkdir -p logs
CMD=(sbatch --account="$ACCOUNT" --partition="${PARTITION:-sporc}" --time="${TIME:-4-00:00:00}"
     --job-name="$RUN_NAME" --array="0-$(( SEEDS - 1 ))"
     ${NOTIFY_SLACK:+--mail-user=slack:$NOTIFY_SLACK --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80,ARRAY_TASKS}
     --export=ALL,RUN_NAME="$RUN_NAME",TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" slurm/train.sbatch)
if [[ "${DRY:-0}" == 1 ]]; then echo "${CMD[@]}"; else "${CMD[@]}"; fi
