#!/bin/bash
# train-ft-m100: pull results/train-ft-m100/ and results/train-ft-m100_s*/ back without the code and with only the
# budget checkpoints (2 / 5 / 10 updates = steps 102,400 / 108,544 / 118,784; 24 updates = checkpoint.pt, 147,456).
set -uo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
N=train-ft-m100
rsync -az -e "$RSYNC_SSH" \
    --include 'checkpoint_step102400.pt' --include 'checkpoint_step108544.pt' --include 'checkpoint_step118784.pt' \
    --exclude 'checkpoint_step*.pt' --exclude 'code/' \
    --include "$N/***" --include "${N}_s*/" --include "${N}_s*/***" --exclude '*' \
    "$REMOTE:capstone/results/" results/
echo "pulled: $(ls results/$N/DONE_s* 2>/dev/null | wc -l)/5 seeds done, $(ls results/${N}_s*/checkpoint_step*.pt 2>/dev/null | wc -l)/15 budget checkpoints"
