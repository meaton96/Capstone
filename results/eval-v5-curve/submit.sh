#!/bin/bash
# eval-v5-curve (10-14): 5 PPO seeds x {checkpoint_init, checkpoint_step51200} on B2 fitting seeds 2000-2159 in the
# twin = 10 cluster tasks, + the 51.2k checkpoints on test seeds 0-39 (5 tasks, added 10-14 for early stopping) (twin_array.sbatch, ~15 min each). The checkpoints and frozen code are already on the cluster.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/eval-v5-curve
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H"
rsync -az -e "$RSYNC_SSH" $H/run_task.py "$REMOTE:capstone/$H/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
D=~/capstone/results/eval-v5-curve
for r in train-v5-nokl_s0 train-v5-nokl_s1 train-v5-nokl_s2 train-v5-final_s3 train-v5-final_s4; do
    for c in checkpoint_init checkpoint_step51200; do
        [[ -f ~/capstone/results/$r/$c.pt ]] || { echo "missing $r/$c.pt"; exit 1; }
        [[ -f $D/${r}__$c/DONE ]] || echo "--run $r --ckpt $c"
    done
    [[ -f $D/${r}__checkpoint_step51200__test/DONE ]] || echo "--run $r --ckpt checkpoint_step51200 --split test"   # 10-14: early stop
done > $D/tasks.txt
N=$(wc -l < $D/tasks.txt); [[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd ~/capstone/linux_server; mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-02:00:00 --mem=6g --array=0-$((N - 1)) --job-name=v5-curve \
    --export=ALL,SCRIPT=results/eval-v5-curve/run_task.py,TASKS=$D/tasks.txt slurm/twin_array.sbatch
REMOTE
