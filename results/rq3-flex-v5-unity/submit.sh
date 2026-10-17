#!/bin/bash
# rq3-flex-v5-unity (10-17): Unity tasks on the cluster (3-4 episodes, ~1 h each): {flex-p0.3, flex-p0.5} x {fa, fb, fc}
# x B2 seeds 0-19 = 120 tasks.  Usage: submit.sh [smoke|full]   (smoke = flex-p0.5 / fc / s0 only; it counts toward full)
# Ships this folder and eval_wrap.py; checkpoints, frozen code and the glibc-patched player are already there (checked).
# Only unfinished tasks are listed. Refuses to rewrite tasks.txt while an rq3-flex array is queued or running.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE=${1:-full}
[[ $MODE == smoke || $MODE == full ]] || { echo "usage: submit.sh [smoke|full]"; exit 2; }
source linux_server/slurm/rit_ssh.sh
H=results/rq3-flex-v5-unity
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H capstone/results/rq4-agvfail-v5-rules"
rsync -az -e "$RSYNC_SSH" --exclude 'runs/' --exclude '*.out' --exclude 'tasks.txt' $H/ "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" results/rq4-agvfail-v5-rules/eval_wrap.py "$REMOTE:capstone/results/rq4-agvfail-v5-rules/"
"${SSH[@]}" "$REMOTE" MODE=$MODE bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
H=results/rq3-flex-v5-unity
for f in results/train-v5-final_s4/checkpoint.pt results/train-v5-loose_s2/checkpoint.pt \
         results/train-v5-cmix_s3/checkpoint.pt results/rq2-rebase-v5/code/env/evaluate.py linux_server/capstone.x86_64; do
    [[ -f $f ]] || { echo "missing on the cluster: $f"; exit 1; }
done
Q=$(squeue --me --noheader --name=rq3-flex --format=%i)
[[ -z $Q ]] || { echo "an rq3-flex array is still queued or running ($(echo $Q)); not rewriting tasks.txt"; exit 1; }
if [[ $MODE == smoke ]]; then L="flex-p0.5 fc 0"; else
    L="$(for s in $(seq 0 19); do for lv in flex-p0.5 flex-p0.3; do for g in fa fb fc; do echo "$lv $g $s"; done; done; done)"; fi
echo "$L" | while read -r lv g s; do [[ -f $H/runs/$lv/$g/s$s/DONE ]] || echo "$lv $g $s"; done > $H/tasks.txt
N=$(wc -l < $H/tasks.txt)
[[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
echo "$N unfinished tasks"
cd linux_server
mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-03:00:00 --array=0-$((N - 1)) --job-name=rq3-flex ../$H/array.sbatch
REMOTE
