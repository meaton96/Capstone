#!/bin/bash
# rq3-gen-v5-cmix: pull the cluster outputs back (not the scripts).
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
rsync -az -e "$RSYNC_SSH" --exclude 'scenarios/' --exclude '__pycache__' --exclude 'task.py' --exclude 'analyze.py' \
    "$REMOTE:capstone/results/rq3-gen-v5-cmix/" results/rq3-gen-v5-cmix/
echo "pulled: $(ls results/rq3-gen-v5-cmix/*/*/DONE 2>/dev/null | wc -l)/16 tasks done"
