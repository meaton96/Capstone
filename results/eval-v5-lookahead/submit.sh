#!/bin/bash
# eval-v5-lookahead (10-15): the 5 train-v5-lookahead checkpoints x {fitting seeds 2000-2159, test seeds 0-39} in the
# twin = 10 cluster tasks (twin_array.sbatch). Checkpoints and frozen code are on the cluster. Resubmitting resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/eval-v5-lookahead
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H"
rsync -az -e "$RSYNC_SSH" $H/run_task.py "$REMOTE:capstone/$H/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
D=~/capstone/results/eval-v5-lookahead
for k in 0 1 2 3 4; do
    r=train-v5-lookahead_s$k
    [[ -f ~/capstone/results/$r/checkpoint.pt ]] || { echo "missing $r/checkpoint.pt"; exit 1; }
    [[ -f $D/${r}__checkpoint/DONE ]] || echo "--run $r --ckpt checkpoint"
    [[ -f $D/${r}__checkpoint__test/DONE ]] || echo "--run $r --ckpt checkpoint --split test"
done > $D/tasks.txt
N=$(wc -l < $D/tasks.txt); [[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd ~/capstone/linux_server; mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-04:00:00 --mem=6g --array=0-$((N - 1)) --job-name=eval-v5-la \
    --export=ALL,SCRIPT=results/eval-v5-lookahead/run_task.py,TASKS=$D/tasks.txt slurm/twin_array.sbatch
REMOTE
