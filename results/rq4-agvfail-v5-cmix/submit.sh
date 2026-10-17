#!/bin/bash
# rq4-agvfail-v5-cmix (10-17, user: the final policy is the PPO trained on mixed due dates): the RQ4 AGV-breakdown
# policy half for the cmix family (train-v5-cmix s0-s4; final pick s3) and the loose pick (train-v5-loose s2), Unity on
# the cluster:  {none, l3000, l1500} x {cmixa (cmix s0-s2 + MDD-PTECT check), cmixb (cmix s3, s4, loose s2)} x B2 seeds
# 0-39 = 240 tasks. At none, cmixb's s3 / loose s2 repeat ../eval-v5-cmix-unity's base runs (run-to-run check).
# Task script, array script and the rules / B2-family runs are ../rq4-agvfail-v5-rules (one setup for every RQ4 half).
# Only unfinished tasks are listed, so resubmitting resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/rq4-agvfail-v5-cmix
RH=results/rq4-agvfail-v5-rules
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H capstone/$RH"
rsync -az -e "$RSYNC_SSH" --exclude 'runs/' --exclude '*.out' --exclude smoke $RH/ "$REMOTE:capstone/$RH/"
rsync -az -e "$RSYNC_SSH" --exclude 'runs/' --exclude '*.out' $H/ "$REMOTE:capstone/$H/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
for f in results/train-v5-cmix_s{0,1,2,3,4}/checkpoint.pt results/train-v5-loose_s2/checkpoint.pt linux_server/capstone.x86_64; do
    [[ -f $f ]] || { echo "missing on the cluster: $f"; exit 1; }
done
H=results/rq4-agvfail-v5-cmix
Q=$(squeue --me --noheader --name=rq4-v5-cmix --format=%i)
[[ -z $Q ]] || { echo "an rq4-v5-cmix array is still queued or running ($(echo $Q)); not rewriting tasks.txt"; exit 1; }
{ for lv in none l3000 l1500; do for g in cmixa cmixb; do for s in $(seq 0 39); do echo "$lv $g $s"; done; done; done; } \
  | while read -r lv g s; do [[ -f $H/runs/$lv/$g/s$s/DONE ]] || echo "$lv $g $s"; done > $H/tasks.txt
N=$(wc -l < $H/tasks.txt)
[[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
echo "$N unfinished tasks"
cd linux_server
mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-04:00:00 --array=0-$((N - 1))%120 \
    --job-name=rq4-v5-cmix --export=ALL,ROOT=$H,TASKS=$HOME/capstone/$H/tasks.txt,WID_BASE=15000 \
    ../results/rq4-agvfail-v5-rules/array.sbatch
REMOTE
