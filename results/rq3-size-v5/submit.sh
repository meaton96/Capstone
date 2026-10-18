#!/bin/bash
# rq3-size-v5 (10-17): m50 x {pdr, ppo, cmix, loose} (4 tasks, ~1-1.5 h) + m100 x {pdr, ppo, cmix, loose} x 4 seed
# chunks of 10 (16 tasks, ~3 h; one 40-seed m100 task needs 12-13 h, over the limit). twin_array.sbatch. Ships task.py, analyze.py and rq3-gen-v5's run_task.py, plus the two floor exports.
# Checkpoints and frozen code are on the cluster. Only unfinished tasks are listed, so resubmitting resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/rq3-size-v5
F=results/agv-ratio-scale/floors
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H capstone/results/rq3-gen-v5 capstone/$F/m50 capstone/$F/m100"
rsync -az -e "$RSYNC_SSH" $H/task.py $H/analyze.py "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" results/rq3-gen-v5/run_task.py "$REMOTE:capstone/results/rq3-gen-v5/"
rsync -az -e "$RSYNC_SSH" $F/m50/des_floor.json "$REMOTE:capstone/$F/m50/"
rsync -az -e "$RSYNC_SSH" $F/m100/des_floor.json "$REMOTE:capstone/$F/m100/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
for r in train-v5-loose train-v5-cmix; do for k in 0 1 2 3 4; do
    [[ -f results/${r}_s$k/checkpoint.pt ]] || { echo "missing results/${r}_s$k/checkpoint.pt"; exit 1; }
done; done
for f in results/train-v5-nokl_s{0,1,2}/checkpoint.pt results/train-v5-final_s{3,4}/checkpoint.pt; do
    [[ -f $f ]] || { echo "missing $f"; exit 1; }
done
D=~/capstone/results/rq3-size-v5
{ for g in pdr ppo cmix loose; do for k in 0-9 10-19 20-29 30-39; do
    [[ -f $D/m100/${g}_s$k/DONE ]] || echo "--cond m100 --group $g --chunk $k"
  done; done
  for g in pdr ppo cmix loose; do [[ -f $D/m50/$g/DONE ]] || echo "--cond m50 --group $g"; done; } > $D/tasks.txt
N=$(wc -l < $D/tasks.txt); [[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd linux_server; mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-06:00:00 --mem=8g --array=0-$((N - 1)) --job-name=rq3-size \
    --export=ALL,SCRIPT=results/rq3-size-v5/task.py,TASKS=$D/tasks.txt slurm/twin_array.sbatch
REMOTE
