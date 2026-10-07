#!/bin/bash
# rq2-oracle-slotlen (10-06): H15 greedy oracle at slot lengths 1,800 / 3,600 / 5,400 / 10,800 s, B2 seeds 0-39 (160 tasks).
# Writes the task list on the cluster and submits slurm/twin_array.sbatch (one CPU per task; the script skips finished
# outputs, so resubmitting resumes). Code: env/ via linux_server/slurm/sync_env_to_rit.sh, scripts + inputs by rsync
# (see the experiment README). Pull: the lab's pull_cmd.
set -euo pipefail
source "$(dirname "$0")/../../linux_server/slurm/rit_ssh.sh"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone/linux_server_v3
mkdir -p logs ~/capstone/results/rq2-oracle-slotlen
for g in 1800 3600 5400 10800; do for s in $(seq 0 39); do echo "--seed $s --segment $g"; done; done > ~/capstone/results/rq2-oracle-slotlen/tasks.txt
N=$(wc -l < ~/capstone/results/rq2-oracle-slotlen/tasks.txt)
sbatch --account=drl-scheduling --partition=sporc --time=0-02:00:00 --array=0-$((N - 1)) --job-name=slotlen \
    --export=ALL,SCRIPT=results/rq2-oracle-slotlen/run.py,TASKS=$HOME/capstone/results/rq2-oracle-slotlen/tasks.txt slurm/twin_array.sbatch
REMOTE
