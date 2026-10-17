#!/bin/bash
# rq4-agvfail-v5-rules (10-14): the RQ4 fixed-rule half on the v5 rules, Unity on the cluster.
#   l3000 / l1500 x {pa, pb} x B2 seeds 0-39 + none x pn x seeds 0-39 = 200 tasks (4-5 episodes each, ~1 h per task
#   at eval-transfer-unity-v5's ~12 min per episode). Ships this folder; the code (rq2-rebase-v5/code/env) and the
#   glibc-patched player (linux_server/, 5e6e0f87) are already on the cluster. Only unfinished tasks are listed, so
#   resubmitting resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=results/rq4-agvfail-v5-rules
source linux_server/slurm/rit_ssh.sh
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H"
rsync -az -e "$RSYNC_SSH" --exclude 'runs/' --exclude '*.out' $H/ "$REMOTE:capstone/$H/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
H=results/rq4-agvfail-v5-rules
{ for lv in l3000 l1500; do for g in pa pb; do for s in $(seq 0 39); do echo "$lv $g $s"; done; done; done
  for s in $(seq 0 39); do echo "none pn $s"; done; } \
  | while read -r lv g s; do [[ -f $H/runs/$lv/$g/s$s/DONE ]] || echo "$lv $g $s"; done > $H/tasks.txt
N=$(wc -l < $H/tasks.txt)
[[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd linux_server
mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-04:00:00 --array=0-$((N - 1))%120 \
    --job-name=rq4-v5-rules --export=ALL,ROOT=$H,TASKS=$HOME/capstone/$H/tasks.txt,WID_BASE=7000 \
    ../$H/array.sbatch
REMOTE
