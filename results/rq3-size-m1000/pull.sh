#!/bin/bash
# rq3-size-m1000: pull the cluster outputs back (not the scripts).
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
rsync -az -e "$RSYNC_SSH" --exclude 'scenarios/' --exclude '__pycache__' --exclude '*.py' --exclude '*.sh' \
    --exclude 'picks.json' "$REMOTE:capstone/results/rq3-size-m1000/" results/rq3-size-m1000/
echo "pulled: ref $(ls results/rq3-size-m1000/m1000/{mdd,srt,t15}_s*/DONE 2>/dev/null | wc -l)/15, arms $(ls results/rq3-size-m1000/m1000/t{50,100,mix}_s*/DONE 2>/dev/null | wc -l)/15 tasks done"
