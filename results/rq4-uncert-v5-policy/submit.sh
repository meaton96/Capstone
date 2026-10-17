#!/bin/bash
# rq4-uncert-v5-policy (10-17): 24 uncertainty cells x {pdr, ppo, cmix, loose} = 96 twin tasks on the cluster
# (twin_array.sbatch, ~10-25 min each). Ships task.py, analyze.py and rq3-gen-v5's run_task.py; checkpoints, frozen code
# and floors are on the cluster. Only unfinished tasks are listed, so resubmitting resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/rq4-uncert-v5-policy
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H capstone/results/rq3-gen-v5"
rsync -az -e "$RSYNC_SSH" $H/task.py $H/analyze.py "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" results/rq3-gen-v5/run_task.py "$REMOTE:capstone/results/rq3-gen-v5/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
for r in train-v5-loose train-v5-cmix; do for k in 0 1 2 3 4; do
    [[ -f results/${r}_s$k/checkpoint.pt ]] || { echo "missing results/${r}_s$k/checkpoint.pt"; exit 1; }
done; done
D=~/capstone/results/rq4-uncert-v5-policy
CELLS="E1=realistic E1=severe E2=realistic E2=severe E2d=realistic E2d=severe E3=realistic E3=severe E4n=realistic E4n=severe O1=realistic O1=severe O2=realistic O2=severe O3=realistic O3=severe mix=realistic mix=severe E2=realistic,info=I1 E2=severe,info=I1 E2d=realistic,info=I1 E2d=severe,info=I1 mix=realistic,info=I1 mix=severe,info=I1"
for g in pdr cmix ppo loose; do for c in $CELLS; do
    d=$(echo "$c" | sed 's/,/_/g; s/=/-/g')
    [[ -f $D/$d/base/$g/DONE ]] || echo "--cell $c --group $g"
done; done > $D/tasks.txt
N=$(wc -l < $D/tasks.txt); [[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd linux_server; mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-03:00:00 --mem=6g --array=0-$((N - 1)) --job-name=rq4-unc-pol \
    --export=ALL,SCRIPT=results/rq4-uncert-v5-policy/task.py,TASKS=$D/tasks.txt slurm/twin_array.sbatch
REMOTE
