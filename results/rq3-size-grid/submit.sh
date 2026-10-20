#!/bin/bash
# rq3-size-grid (10-20): cluster submit, twin_array.sbatch, one CPU per task. Only unfinished tasks are listed, so
# resubmitting resumes.
#   ref   pdr + cmix on the new floors m20 m30 m75 (10-seed chunks, 6 h) and m150 m200 (5-seed chunks, 10 h): 56 tasks.
#         m15 / m50 / m100 pdr and cmix come from rq3-gen-v5, eval-v5-cmix and rq3-size-v5.
#   arms  size50 / size100 / sizemix (train-size-m50 / -m100 / -mix s0-s4) on all 8 floors: 120 tasks. Needs the
#         checkpoints on the cluster (results/train-size-*_s<k>/checkpoint.pt).
# Ships task.py, analyze.py, rq3-size-v5's task.py, rq3-gen-v5's run_task.py and the floor exports.
set -euo pipefail
PART="${1:?usage: submit.sh ref|arms}"
[[ $PART == ref || $PART == arms ]] || { echo "usage: submit.sh ref|arms"; exit 1; }
cd "$(dirname "$0")/../.."
source linux_server/slurm/rit_ssh.sh
H=results/rq3-size-grid
F=results/dev-size-floors/floors
A=results/agv-ratio-scale/floors
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H capstone/results/rq3-size-v5 capstone/results/rq3-gen-v5 capstone/$A/m50 capstone/$A/m100 $(for m in 20 30 75 150 200; do printf 'capstone/%s/m%s ' $F $m; done)"
rsync -az -e "$RSYNC_SSH" $H/task.py $H/analyze.py "$REMOTE:capstone/$H/"
rsync -az -e "$RSYNC_SSH" results/rq3-size-v5/task.py "$REMOTE:capstone/results/rq3-size-v5/"
rsync -az -e "$RSYNC_SSH" results/rq3-gen-v5/run_task.py "$REMOTE:capstone/results/rq3-gen-v5/"
for m in 50 100; do rsync -az -e "$RSYNC_SSH" $A/m$m/des_floor.json "$REMOTE:capstone/$A/m$m/"; done
for m in 20 30 75 150 200; do rsync -az -e "$RSYNC_SSH" $F/m$m/des_floor.json "$REMOTE:capstone/$F/m$m/"; done
"${SSH[@]}" "$REMOTE" "PART=$PART bash -s" <<'REMOTE'
set -euo pipefail
cd ~/capstone
H=results/rq3-size-grid
D=~/capstone/$H
if [[ $PART == ref ]]; then
    GRPS="pdr cmix"; SMALL="m20 m30 m75"; RUNS="train-v5-cmix"
else
    GRPS="size50 size100 sizemix"; SMALL="m15 m20 m30 m50 m75 m100"; RUNS="train-size-m50 train-size-m100 train-size-mix"
fi
BIG="m150 m200"
for r in $RUNS; do for k in 0 1 2 3 4; do
    [[ -f results/${r}_s$k/checkpoint.pt ]] || { echo "missing results/${r}_s$k/checkpoint.pt"; exit 1; }
done; done
list() {   # floors chunks -> one task line per unfinished (floor, group, chunk)
    for c in $1; do for g in $GRPS; do for k in $2; do
        [[ -f $D/$c/${g}_s$k/DONE ]] || echo "--cond $c --group $g --chunk $k"
    done; done; done
}
list "$SMALL" "0-9 10-19 20-29 30-39" > $D/tasks_${PART}_small.txt
list "$BIG" "0-4 5-9 10-14 15-19 20-24 25-29 30-34 35-39" > $D/tasks_${PART}_big.txt
cd linux_server; mkdir -p logs
for sz in small big; do
    T=$D/tasks_${PART}_$sz.txt; N=$(wc -l < $T)
    [[ $N -gt 0 ]] || { echo "$sz: nothing to run"; continue; }
    LIM=0-06:00:00; [[ $sz == big ]] && LIM=0-10:00:00
    echo "$sz: $N tasks, limit $LIM"
    sbatch --account=drl-scheduling --partition=sporc --time=$LIM --mem=8g --array=0-$((N - 1)) \
        --job-name=rq3-grid-$PART --export=ALL,SCRIPT=$H/task.py,TASKS=$T slurm/twin_array.sbatch
done
REMOTE
