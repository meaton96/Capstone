#!/bin/bash
# rq3-gen-v5-cmix (10-17): 8 conditions x {cmix, loose} = 16 twin tasks on the cluster (twin_array.sbatch, ~20-30 min
# each). Ships task.py, analyze.py and rq3-gen-v5's run_task.py; checkpoints, frozen code and floors are on the cluster.
# Only unfinished tasks are listed, so resubmitting resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/rq3-gen-v5-cmix
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H capstone/results/rq3-gen-v5"
rsync -az -e "$RSYNC_SSH" $H/task.py $H/analyze.py "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" results/rq3-gen-v5/run_task.py "$REMOTE:capstone/results/rq3-gen-v5/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
for r in train-v5-loose train-v5-cmix; do for k in 0 1 2 3 4; do
    [[ -f results/${r}_s$k/checkpoint.pt ]] || { echo "missing results/${r}_s$k/checkpoint.pt"; exit 1; }
done; done
D=~/capstone/results/rq3-gen-v5-cmix
for g in cmix loose; do for c in load-0.8 load-1.2 agv5-bay9 agv7-bay9 agv9-bay9 lay-c lay-g lay-j; do
    [[ -f $D/$c/$g/DONE ]] || echo "--cond $c --group $g"
done; done > $D/tasks.txt
N=$(wc -l < $D/tasks.txt); [[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd linux_server; mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-03:00:00 --mem=6g --array=0-$((N - 1)) --job-name=rq3-cmix \
    --export=ALL,SCRIPT=results/rq3-gen-v5-cmix/task.py,TASKS=$D/tasks.txt slurm/twin_array.sbatch
REMOTE
