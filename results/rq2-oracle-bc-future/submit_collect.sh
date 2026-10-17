#!/bin/bash
# rq2-oracle-bc-future step 1 (cluster): the data. collect.py label on new B2 seeds 3000-3999 (10 seeds per task, the
# rq2-rebase-v5 oracle + the future table) and collect.py future on the evaluation seeds 0-39 / 2000-2159 (20 per
# task) = 110 one-CPU tasks (slurm/twin_array.sbatch). Resubmitting resumes: finished seeds are skipped, and tasks
# whose seeds are all on the cluster already are left out.
set -euo pipefail
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/rq2-oracle-bc-future
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H/code"
rsync -az -e "$RSYNC_SSH" $H/common.py $H/collect.py "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" $H/code/ "$REMOTE:capstone/$H/code/"
rsync -az -e "$RSYNC_SSH" results/rq2-perturbed-v5/run.py "$REMOTE:capstone/results/rq2-perturbed-v5/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
D=results/rq2-oracle-bc-future
for f in results/rq2-rebase-v5/run.py results/rq2-rebase-v5/sim.py results/rq2-rebase-v5/code/env/des_twin/engine.py \
         results/dev-congestion-signal/run.py results/eval-v5-lookahead-fcast/run_task.py; do
    [[ -f $f ]] || { echo "missing $f on the cluster"; exit 1; }
done
[[ $(ls results/rq2-rebase-v5/out/oracle/s*.npz | wc -l) -eq 200 ]] || { echo "rq2-rebase-v5 oracle labels incomplete"; exit 1; }
source .venv/bin/activate
python $D/collect.py list | while read -r kind seeds; do
    lo=${seeds%-*}; hi=${seeds#*-}; dir=labels; [[ $kind == future ]] && dir=future
    for s in $(seq $lo $hi); do
        [[ -f $D/out/$dir/s$s.npz ]] || { echo "$kind $seeds"; break; }
    done
done > $D/tasks_collect.txt
N=$(wc -l < $D/tasks_collect.txt); [[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd linux_server; mkdir -p logs
sbatch --account=drl-scheduling --partition=sporc --time=0-04:00:00 --mem=4g --array=0-$((N - 1)) \
    --job-name=bcfut-data \
    --export=ALL,SCRIPT=results/rq2-oracle-bc-future/collect.py,TASKS=$HOME/capstone/$D/tasks_collect.txt \
    slurm/twin_array.sbatch
REMOTE
