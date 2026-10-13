#!/bin/bash
# dev-oracle-bc-collect (10-06): oracle labels + full obs for behavior cloning, B2 seeds 0-39 and 2000-2159 (200 tasks, ~12 CPU-min each).
# Writes the task list on the cluster and submits slurm/twin_array.sbatch (one CPU per task; the script skips finished
# outputs, so resubmitting resumes). Code: env/ via linux_server/slurm/sync_env_to_rit.sh, scripts + inputs by rsync
# (see the experiment README). Pull: the lab's pull_cmd.
set -euo pipefail
source "$(dirname "$0")/../../linux_server/slurm/rit_ssh.sh"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone/linux_server_v3
mkdir -p logs ~/capstone/results/dev-oracle-bc
for s in $(seq 0 39) $(seq 2000 2159); do echo "--seeds $s --workers 1"; done > ~/capstone/results/dev-oracle-bc/tasks.txt
N=$(wc -l < ~/capstone/results/dev-oracle-bc/tasks.txt)
sbatch --account=drl-scheduling --partition=sporc --time=0-03:00:00 --array=0-$((N - 1)) --job-name=oracle-bc \
    --export=ALL,SCRIPT=results/dev-oracle-bc/collect.py,TASKS=$HOME/capstone/results/dev-oracle-bc/tasks.txt slurm/twin_array.sbatch
REMOTE
