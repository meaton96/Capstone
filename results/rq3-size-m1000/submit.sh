#!/bin/bash
# rq3-size-m1000 (10-20): cluster submit, twin_array.sbatch, one (policy, seed) episode per task, test seeds 0-4.
#   ref   t15 (5 tasks; --nice=100 so the queued train-size-* jobs stay ahead). The rule groups mdd / srt run locally
#         in the oracle simulator since 10-20 (user; sim_rules.py, launch_sim.sh); array 21932264 tasks 0-9 cancelled.
#   arms  t50, t100, tmix (15 tasks; runs pick.py first: needs rq3-size-grid's m200 arms in its summary.csv)
# Limits: 2 days, 20 GB per task (one episode ~6 h / 7.7 GB locally on seed 0; ~1.6x slower on the cluster's nodes;
# episode time varies +-50% by seed). Only unfinished tasks are listed, so resubmitting resumes.
set -euo pipefail
PART="${1:?usage: submit.sh ref|arms}"
cd "$(dirname "$0")/../.."
H=results/rq3-size-m1000
case $PART in
    ref)  GRPS="t15"; NICE="--nice=100"; RUNS="train-v5-cmix" ;;
    arms) GRPS="t50 t100 tmix"; NICE=""; RUNS="train-size-m50 train-size-m100 train-size-mix"
          .venv/bin/python $H/pick.py ;;
    *) echo "usage: submit.sh ref|arms"; exit 1 ;;
esac
source linux_server/slurm/rit_ssh.sh
F=results/dev-size-floors/floors/m1000
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H capstone/$F capstone/results/rq3-size-grid capstone/results/rq3-size-v5 capstone/results/rq3-gen-v5"
rsync -az -e "$RSYNC_SSH" $H/task.py $H/analyze.py "$REMOTE:capstone/$H/"
[[ $PART == arms ]] && rsync -az -e "$RSYNC_SSH" $H/picks.json "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" results/rq3-size-grid/task.py "$REMOTE:capstone/results/rq3-size-grid/"
rsync -az -e "$RSYNC_SSH" results/rq3-size-v5/task.py "$REMOTE:capstone/results/rq3-size-v5/"
rsync -az -e "$RSYNC_SSH" results/rq3-gen-v5/run_task.py "$REMOTE:capstone/results/rq3-gen-v5/"
rsync -az -e "$RSYNC_SSH" $F/des_floor.json "$REMOTE:capstone/$F/"
"${SSH[@]}" "$REMOTE" "PART=$PART GRPS='$GRPS' NICE='$NICE' RUNS='$RUNS' bash -s" <<'REMOTE'
set -euo pipefail
cd ~/capstone
H=results/rq3-size-m1000
D=~/capstone/$H
for r in $RUNS; do for k in 0 1 2 3 4; do
    [[ -f results/${r}_s$k/checkpoint.pt ]] || { echo "missing results/${r}_s$k/checkpoint.pt"; exit 1; }
done; done
T=$D/tasks_$PART.txt
for g in $GRPS; do for s in 0 1 2 3 4; do
    [[ -f $D/m1000/${g}_s$s/DONE ]] || echo "--group $g --seed $s"
done; done > $T
N=$(wc -l < $T); [[ $N -gt 0 ]] || { echo "nothing to run"; exit 0; }
cd linux_server; mkdir -p logs
echo "$PART: $N tasks"
sbatch --account=drl-scheduling --partition=sporc --time=2-00:00:00 --mem=20g $NICE --array=0-$((N - 1)) \
    --job-name=rq3-m1000-$PART --export=ALL,SCRIPT=$H/task.py,TASKS=$T slurm/twin_array.sbatch
REMOTE
