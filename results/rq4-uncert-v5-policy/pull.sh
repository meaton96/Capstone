#!/bin/bash
# rq4-uncert-v5-policy: pull the cluster outputs back (not the scripts).
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
rsync -az -e "$RSYNC_SSH" --exclude 'scenarios/' --exclude '__pycache__' --exclude 'task.py' --exclude 'analyze.py' \
    "$REMOTE:capstone/results/rq4-uncert-v5-policy/" results/rq4-uncert-v5-policy/
echo "pulled: $(ls results/rq4-uncert-v5-policy/*/base/*/DONE 2>/dev/null | wc -l)/96 tasks done"
