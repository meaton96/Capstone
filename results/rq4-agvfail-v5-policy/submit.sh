#!/bin/bash
# rq4-agvfail-v5-policy (10-14): the RQ4 policy half on the v5 rules, Unity on the cluster. The 5-seed PPO stack
# (train-v5-nokl s0-2, train-v5-final s3-4) and BC of the v5 oracle (rq2-oracle-bc-v5 s0-2) under AGV breakdowns:
#   l3000 / l1500 x {ppoa (PPO s0-s2 + MDD-PTECT check), ppob (PPO s3-s4), bc (BC s0-s2)} x B2 seeds 0-39
#   + none x ppob x seeds 0-39 (s0-s2 and BC at none are in ../eval-transfer-unity-v5) = 280 tasks.
# Task script, array script and analysis are ../rq4-agvfail-v5-rules/{task.sh,array.sbatch,analyze.py} (one setup for
# both halves). Only unfinished tasks are listed, so resubmitting resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/rq4-agvfail-v5-policy
RH=results/rq4-agvfail-v5-rules
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H capstone/$RH"
rsync -az -e "$RSYNC_SSH" --exclude 'runs/' --exclude '*.out' --exclude smoke $RH/ "$REMOTE:capstone/$RH/"
rsync -az -e "$RSYNC_SSH" --exclude 'runs/' --exclude '*.out' $H/ "$REMOTE:capstone/$H/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
for f in results/train-v5-nokl_s{0,1,2}/checkpoint.pt results/train-v5-final_s{3,4}/checkpoint.pt \
         results/rq2-oracle-bc-v5/bc/bc_full_s{0,1,2}.pt; do
    [[ -f $f ]] || { echo "missing on the cluster: $f"; exit 1; }
done
H=results/rq4-agvfail-v5-policy
{ for lv in l3000 l1500; do for g in ppoa ppob bc; do for s in $(seq 0 39); do echo "$lv $g $s"; done; done; done
  for s in $(seq 0 39); do echo "none ppob $s"; done; } \
  | while read -r lv g s; do [[ -f $H/runs/$lv/$g/s$s/DONE ]] || echo "$lv $g $s"; done > $H/tasks.txt
N=$(wc -l < $H/tasks.txt)
[[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd linux_server
mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-04:00:00 --array=0-$((N - 1))%120 \
    --job-name=rq4-v5-policy --export=ALL,ROOT=$H,TASKS=$HOME/capstone/$H/tasks.txt,WID_BASE=9000 \
    ../results/rq4-agvfail-v5-rules/array.sbatch
REMOTE
