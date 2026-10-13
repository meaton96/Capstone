#!/bin/bash
# dev-oracle-bc-ev collect (2026-10-07, handoff fix 7): expected-value oracle labels (collect_ev.py, 4 redrawn futures
# per slot) for B2 seeds 0-39 and 2000-2159, one CPU per seed (~75 CPU-min each, ~250 CPU-h in all).
# Before submitting, env/ must be synced (linux_server/slurm/sync_env_to_rit.sh: collect_ev.py imports
# env_wrappers/paired_slot_env.py, new on 10-07), and only when no cluster job runs from the synced code.
# Copies the scripts, writes the task list, submits slurm/twin_array.sbatch (collect_ev.py skips finished seeds, so a
# resubmit resumes). Pull: the lab's pull_cmd.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
"${SSH[@]}" "$REMOTE" "mkdir -p ~/capstone/results/dev-oracle-bc-ev ~/capstone/results/dev-oracle-bc"
rsync -az -e "$RSYNC_SSH" results/dev-oracle-bc-ev/collect_ev.py "$REMOTE":capstone/results/dev-oracle-bc-ev/
rsync -az -e "$RSYNC_SSH" results/dev-oracle-bc/collect.py "$REMOTE":capstone/results/dev-oracle-bc/
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone/linux_server_v3
mkdir -p logs
for s in $(seq 0 39) $(seq 2000 2159); do echo "--seeds $s --futures 4 --workers 1"; done > ~/capstone/results/dev-oracle-bc-ev/tasks.txt
N=$(wc -l < ~/capstone/results/dev-oracle-bc-ev/tasks.txt)
sbatch --account=drl-scheduling --partition=sporc --time=0-04:00:00 --array=0-$((N - 1)) --job-name=oracle-bc-ev \
    --export=ALL,SCRIPT=results/dev-oracle-bc-ev/collect_ev.py,TASKS=$HOME/capstone/results/dev-oracle-bc-ev/tasks.txt slurm/twin_array.sbatch
REMOTE
