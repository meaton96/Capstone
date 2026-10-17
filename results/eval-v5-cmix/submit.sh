#!/bin/bash
# eval-v5-cmix (10-14): 54 twin tasks on the cluster (twin_array.sbatch, ~20-30 min each):
#   test x {loose, cmix} x 3 conditions  +  4 fitting chunks x {pdr, ppo, loose, cmix} x 3 conditions.
# Ships cmix_task.py, analyze.py and rq3-gen-v5's run_task.py; checkpoints, frozen code and floors are on the cluster.
# Only unfinished tasks are listed, so resubmitting resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/eval-v5-cmix
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H capstone/results/rq3-gen-v5"
rsync -az -e "$RSYNC_SSH" $H/cmix_task.py $H/analyze.py "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" results/rq3-gen-v5/run_task.py "$REMOTE:capstone/results/rq3-gen-v5/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
for r in train-v5-loose train-v5-cmix train-v5-nokl train-v5-final; do
    for k in 0 1 2 3 4; do
        case $r in train-v5-nokl) [[ $k -le 2 ]] || continue;; train-v5-final) [[ $k -ge 3 ]] || continue;; esac
        [[ -f results/${r}_s$k/checkpoint.pt ]] || { echo "missing results/${r}_s$k/checkpoint.pt"; exit 1; }
    done
done
D=~/capstone/results/eval-v5-cmix
{ for c in due-tight base due-loose; do
    for g in loose cmix; do echo "test $c $g"; done
    for s in fit2000 fit2040 fit2080 fit2120; do for g in pdr ppo loose cmix; do echo "$s $c $g"; done; done
  done; } | while read -r s c g; do [[ -f $D/$s/$c/$g/DONE ]] || echo "--split $s --cond $c --group $g"; done > $D/tasks.txt
N=$(wc -l < $D/tasks.txt); [[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd linux_server; mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-03:00:00 --mem=6g --array=0-$((N - 1)) --job-name=eval-v5-cmix \
    --export=ALL,SCRIPT=results/eval-v5-cmix/cmix_task.py,TASKS=$D/tasks.txt slurm/twin_array.sbatch
REMOTE
