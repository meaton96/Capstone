#!/bin/bash
# rq3-oracle-loose (10-14): run.py's 40 oracle tasks (loose c, B2 test seeds 0-39) as a cluster CPU array
# (twin_array.sbatch; ~2-5 min each). rq2-rebase-v5 (frozen code, sim.py, run.py) is already on the cluster.
# Tasks skip finished outputs, so resubmitting resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python results/rq3-oracle-loose/run.py list > results/rq3-oracle-loose/tasks.txt
source linux_server/slurm/rit_ssh.sh
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/results/rq3-oracle-loose"
rsync -az -e "$RSYNC_SSH" results/rq3-oracle-loose/run.py results/rq3-oracle-loose/tasks.txt "$REMOTE:capstone/results/rq3-oracle-loose/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone && .venv/bin/python results/rq3-oracle-loose/run.py check
cd linux_server; mkdir -p logs
D=~/capstone/results/rq3-oracle-loose
sbatch --account=drl-scheduling --partition=sporc --time=0-01:00:00 --mem=4g --array=0-39 --job-name=rq3-oracle-loose \
    --export=ALL,SCRIPT=results/rq3-oracle-loose/run.py,TASKS=$D/tasks.txt slurm/twin_array.sbatch
REMOTE
