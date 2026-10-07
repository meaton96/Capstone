#!/bin/bash
# rq2-oracle-steady (10-07): S6 oracle + full obs per slot, B2 seeds 0-39 (test) and 2000-2159 (BC training): 200
# tasks, ~1 h each (24 stages x 15 tails x ~8 s). Task file tasks_oracle.txt; twin_array.sbatch; run.py skips finished
# seeds, so resubmitting resumes. Code: env/ via linux_server/slurm/sync_env_to_rit.sh; scripts by rsync (README).
set -euo pipefail
source "$(dirname "$0")/../../linux_server/slurm/rit_ssh.sh"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone/linux_server_v3
D=~/capstone/results/rq2-oracle-steady
mkdir -p logs $D
for s in $(seq 0 39) $(seq 2000 2159); do echo "--mode oracle --seed $s --setting S6"; done > $D/tasks_oracle.txt
sbatch --account=drl-scheduling --partition=sporc --time=0-06:00:00 --array=0-199 --job-name=steady-oracle \
    --export=ALL,SCRIPT=results/rq2-oracle-steady/run.py,TASKS=$D/tasks_oracle.txt slurm/twin_array.sbatch
REMOTE
