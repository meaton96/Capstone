#!/bin/bash
# rq3-gen-v5-unity (10-14): 3 conditions x {pa, pb} x B2 seeds 0-19 = 120 Unity tasks on the cluster (3-4 episodes,
# ~1 h each). Ships this folder and eval-v5-final/selected.txt (the final policy; its checkpoint is already there).
# Only unfinished tasks are listed, so resubmitting resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ -f results/eval-v5-final/selected.txt ]] || { echo "no final policy yet (results/eval-v5-final/selected.txt)"; exit 1; }
source linux_server/slurm/rit_ssh.sh
H=results/rq3-gen-v5-unity
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H capstone/results/eval-v5-final"
rsync -az -e "$RSYNC_SSH" --exclude 'runs/' --exclude '*.out' $H/ "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" results/eval-v5-final/selected.txt "$REMOTE:capstone/results/eval-v5-final/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
H=results/rq3-gen-v5-unity
CK=$(cut -d' ' -f2 results/eval-v5-final/selected.txt); [[ -f $CK ]] || { echo "missing on the cluster: $CK"; exit 1; }
for c in load-1.2 due-tight lay-g; do for g in pa pb; do for s in $(seq 0 19); do
    [[ -f $H/runs/$c/$g/s$s/DONE ]] || echo "$c $g $s"; done; done; done > $H/tasks.txt
N=$(wc -l < $H/tasks.txt)
[[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd linux_server
mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-04:00:00 --array=0-$((N - 1)) --job-name=rq3-v5-unity \
    ../$H/array.sbatch
REMOTE
