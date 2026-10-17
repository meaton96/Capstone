#!/bin/bash
# rq3-gen-v5-unity-cmix (10-17): Unity tasks on the cluster (3 episodes, ~1 h each): {load-1.2, due-tight, lay-g} x pc
# x B2 seeds 0-19 = 60 tasks. Ships this folder and eval_wrap.py; checkpoints, frozen code and the glibc-patched player
# are already there (checked). Only unfinished tasks are listed, so resubmitting resumes. Refuses to rewrite tasks.txt
# while an rq3u-cmix array is still queued or running (its tasks read it by line number).
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/rq3-gen-v5-unity-cmix
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H capstone/results/rq4-agvfail-v5-rules"
rsync -az -e "$RSYNC_SSH" --exclude 'runs/' --exclude '*.out' --exclude 'tasks.txt' $H/ "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" results/rq4-agvfail-v5-rules/eval_wrap.py "$REMOTE:capstone/results/rq4-agvfail-v5-rules/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
H=results/rq3-gen-v5-unity-cmix
for f in results/train-v5-final_s4/checkpoint.pt results/train-v5-loose_s2/checkpoint.pt \
         results/train-v5-cmix_s3/checkpoint.pt results/rq2-rebase-v5/code/env/evaluate.py linux_server/capstone.x86_64; do
    [[ -f $f ]] || { echo "missing on the cluster: $f"; exit 1; }
done
Q=$(squeue --me --noheader --name=rq3u-cmix --format=%i)
[[ -z $Q ]] || { echo "an rq3u-cmix array is still queued or running ($(echo $Q)); not rewriting tasks.txt"; exit 1; }
for s in $(seq 0 19); do for c in load-1.2 due-tight lay-g; do echo "$c pc $s"; done; done \
    | while read -r c g s; do [[ -f $H/runs/$c/$g/s$s/DONE ]] || echo "$c $g $s"; done > $H/tasks.txt
N=$(wc -l < $H/tasks.txt)
[[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
echo "$N unfinished tasks"
cd linux_server
mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-03:00:00 --array=0-$((N - 1)) --job-name=rq3u-cmix ../$H/array.sbatch
REMOTE
