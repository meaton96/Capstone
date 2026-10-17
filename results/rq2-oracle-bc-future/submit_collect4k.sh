#!/bin/bash
# rq2-oracle-bc-future data scaling (10-16, user): 3,000 more training labels, B2 seeds 4000-6999 (collect.py label,
# 10 seeds per one-CPU task, slurm/twin_array.sbatch) -> out/labels/ next to seeds 3000-3999. Same code as
# submit_collect.sh. Resubmitting resumes: tasks whose seeds are all on the cluster already are left out.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/rq2-oracle-bc-future
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H/code"
rsync -az -e "$RSYNC_SSH" $H/common.py $H/collect.py "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" $H/code/ "$REMOTE:capstone/$H/code/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
D=results/rq2-oracle-bc-future
source .venv/bin/activate
python $D/collect.py list4k | while read -r kind seeds; do
    lo=${seeds%-*}; hi=${seeds#*-}
    for s in $(seq $lo $hi); do
        [[ -f $D/out/labels/s$s.npz ]] || { echo "$kind $seeds"; break; }
    done
done > $D/tasks_collect4k.txt
N=$(wc -l < $D/tasks_collect4k.txt); [[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd linux_server; mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-04:00:00 --mem=4g --array=0-$((N - 1)) \
    --job-name=bcfut-data4k \
    --export=ALL,SCRIPT=results/rq2-oracle-bc-future/collect.py,TASKS=$HOME/capstone/$D/tasks_collect4k.txt \
    slurm/twin_array.sbatch
REMOTE
