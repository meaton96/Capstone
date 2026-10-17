#!/bin/bash
# eval-v5-cmix-unity (10-15): Unity tasks on the cluster (3-4 episodes, ~1 h each), per B2 seed:
#   due-loose x {pa, pb, pc}  +  base x pc  (4 tasks per seed; seeds 0-19 = the first pass, 80 tasks).
# Usage: submit.sh [smoke|full] [FIRST LAST]   (seed range, default 0 19; or env SEED_FIRST / SEED_LAST)
#   `submit.sh full 0 39` (10-16) adds seeds 20-39 to the finished first pass: 80 more tasks.
# `submit.sh smoke` submits only base / pc / s0 (the 1-task smoke; its output counts toward the full run).
# Ships this folder and eval_wrap.py; the checkpoints, rq2-rebase-v5's frozen code and the glibc-patched player
# (linux_server/, 5e6e0f87) are already there (checked below). Only unfinished tasks are listed, so resubmitting resumes.
# Refuses to rewrite tasks.txt while a cmix-unity array is still queued or running (its tasks read it by line number).
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE=${1:-full}
FIRST=${2:-${SEED_FIRST:-0}}
LAST=${3:-${SEED_LAST:-19}}
[[ $MODE == smoke || $MODE == full ]] || { echo "usage: submit.sh [smoke|full] [FIRST LAST]"; exit 2; }
[[ $FIRST =~ ^[0-9]+$ && $LAST =~ ^[0-9]+$ && $FIRST -le $LAST ]] || { echo "bad seed range $FIRST $LAST"; exit 2; }
source linux_server/slurm/rit_ssh.sh
H=results/eval-v5-cmix-unity
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H capstone/results/rq4-agvfail-v5-rules"
rsync -az -e "$RSYNC_SSH" --exclude 'runs/' --exclude '*.out' --exclude 'tasks.txt' $H/ "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" results/rq4-agvfail-v5-rules/eval_wrap.py "$REMOTE:capstone/results/rq4-agvfail-v5-rules/"
"${SSH[@]}" "$REMOTE" MODE=$MODE FIRST=$FIRST LAST=$LAST bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
H=results/eval-v5-cmix-unity
for f in results/train-v5-final_s4/checkpoint.pt results/train-v5-loose_s2/checkpoint.pt \
         results/train-v5-cmix_s3/checkpoint.pt results/rq2-rebase-v5/code/env/evaluate.py linux_server/capstone.x86_64; do
    [[ -f $f ]] || { echo "missing on the cluster: $f"; exit 1; }
done
Q=$(squeue --me --noheader --name=cmix-unity --format=%i)
[[ -z $Q ]] || { echo "a cmix-unity array is still queued or running ($(echo $Q)); not rewriting tasks.txt"; exit 1; }
if [[ $MODE == smoke ]]; then
    L="base pc 0"
else
    L="$(for s in $(seq $FIRST $LAST); do for g in pa pb pc; do echo "due-loose $g $s"; done; echo "base pc $s"; done)"
fi
echo "$L" | while read -r c g s; do [[ -f $H/runs/$c/$g/s$s/DONE ]] || echo "$c $g $s"; done > $H/tasks.txt
N=$(wc -l < $H/tasks.txt)
[[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
echo "$N unfinished tasks (seeds $FIRST-$LAST)"
cd linux_server
mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-03:00:00 --array=0-$((N - 1)) --job-name=cmix-unity \
    ../$H/array.sbatch
REMOTE
