#!/bin/bash
# rq3-size-remap (10-19): remaps {size, wip, opt, all} x (m50 all 40 seeds + m100 in 4 ten-seed chunks), group cmix
# (final policy family) = 20 twin tasks on the cluster (twin_array.sbatch; ~1-1.5 h each). Ships task.py, analyze.py,
# rq3-size-v5/task.py and rq3-gen-v5/run_task.py; the floors and checkpoints are on the cluster (rq3-size-v5).
# Only unfinished tasks are listed, so resubmitting resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/rq3-size-remap
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H capstone/results/rq3-size-v5 capstone/results/rq3-gen-v5"
rsync -az -e "$RSYNC_SSH" $H/task.py $H/analyze.py "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" results/rq3-size-v5/task.py "$REMOTE:capstone/results/rq3-size-v5/"
rsync -az -e "$RSYNC_SSH" results/rq3-gen-v5/run_task.py "$REMOTE:capstone/results/rq3-gen-v5/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
for f in results/agv-ratio-scale/floors/m50/des_floor.json results/agv-ratio-scale/floors/m100/des_floor.json \
         results/train-v5-cmix_s{0,1,2,3,4}/checkpoint.pt; do
    [[ -f $f ]] || { echo "missing on the cluster: $f"; exit 1; }
done
D=~/capstone/results/rq3-size-remap
{ for r in all size wip opt; do
    for k in 0-9 10-19 20-29 30-39; do [[ -f $D/$r/m100/cmix_s$k/DONE ]] || echo "--remap $r --cond m100 --chunk $k"; done
    [[ -f $D/$r/m50/cmix/DONE ]] || echo "--remap $r --cond m50"
  done; } > $D/tasks.txt
N=$(wc -l < $D/tasks.txt); [[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd linux_server; mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-04:00:00 --mem=8g --array=0-$((N - 1)) --job-name=rq3-remap \
    --export=ALL,SCRIPT=results/rq3-size-remap/task.py,TASKS=$D/tasks.txt slurm/twin_array.sbatch
REMOTE
