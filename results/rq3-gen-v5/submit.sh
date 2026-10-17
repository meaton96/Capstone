#!/bin/bash
# rq3-gen-v5 (10-14): submit the RQ3 twin conditions sweep on the v5 rules as a cluster CPU array
# (linux_server/slurm/twin_array.sbatch). 11 conditions x 3 groups (pdr, ppo, bc) = 33 tasks, one run_task.py call
# each (B2 seeds 0-39, ~10-20 min per task). Ships run_task.py and analyze.py; the frozen code
# (rq2-rebase-v5/code/env), the floors (rq3-gen-twin/floors), the BC checkpoints and all 5 PPO checkpoints
# (train-v5-nokl_s0-2, train-v5-final_s3-4, trained there) are already on the cluster. Only unfinished tasks are
# listed, so resubmitting resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/rq3-gen-v5
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H"
rsync -az -e "$RSYNC_SSH" $H/run_task.py $H/analyze.py "$REMOTE:capstone/$H/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
for f in results/train-v5-nokl_s{0,1,2}/checkpoint.pt results/train-v5-final_s{3,4}/checkpoint.pt \
         results/rq2-oracle-bc-v5/bc/bc_full_s{0,1,2}.pt results/rq2-rebase-v5/code/STAMP; do
    [[ -f $f ]] || { echo "missing on the cluster: $f"; exit 1; }
done
D=~/capstone/results/rq3-gen-v5
for g in pdr ppo bc; do
    for c in base load-0.8 load-1.2 due-tight due-loose agv5-bay9 agv7-bay9 agv9-bay9 lay-c lay-g lay-j; do
        [[ -f $D/$c/$g/DONE ]] || echo "--cond $c --group $g"
    done
done > $D/tasks.txt
N=$(wc -l < $D/tasks.txt)
[[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd linux_server
mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-03:00:00 --mem=6g --array=0-$((N - 1)) --job-name=rq3-gen-v5 \
    --export=ALL,SCRIPT=results/rq3-gen-v5/run_task.py,TASKS=$D/tasks.txt slurm/twin_array.sbatch
REMOTE
