#!/bin/bash
# rq3-size-grid: pull the cluster outputs back (not the scripts).
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
rsync -az -e "$RSYNC_SSH" --exclude 'scenarios/' --exclude '__pycache__' --exclude '*.py' --exclude '*.sh' \
    "$REMOTE:capstone/results/rq3-size-grid/" results/rq3-size-grid/
echo "pulled: ref $(ls results/rq3-size-grid/*/{pdr,cmix}_s*/DONE 2>/dev/null | wc -l)/56, arms $(ls results/rq3-size-grid/*/size*_s*/DONE 2>/dev/null | wc -l)/120 tasks done"
