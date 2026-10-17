#!/bin/bash
# rq3-gen-v5: pull the cluster outputs back (not the scripts).
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
rsync -az -e "$RSYNC_SSH" --exclude 'scenarios/' --exclude '__pycache__' --exclude 'run_task.py' --exclude 'analyze.py' \
    "$REMOTE:capstone/results/rq3-gen-v5/" results/rq3-gen-v5/
echo "pulled: $(ls results/rq3-gen-v5/*/*/DONE 2>/dev/null | wc -l)/33 tasks done"
