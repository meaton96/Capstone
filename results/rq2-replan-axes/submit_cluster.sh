#!/bin/bash
# rq2-replan-axes (10-29): one task set (run.py task_sets: check | proc | reg | axis1 | axis1-ext | size | fleet | axis6 | axis2) as
# cluster CPU arrays (twin_array.sbatch, one CPU per task), costliest first (run.py list orders them). Ships this folder
# (frozen code/env included, out/ and the dry runs excluded); tasks skip finished outputs, so resubmitting resumes. Arrays
# are split at 1,000 tasks. Time limit 3 h, 12 h for the axis 6 sets (D1 60 s at 100 machines takes ~3 h on one core,
# README §13); axis 2's longest task (D1 60 s at load 1.3) takes ~30 min (README §15). Usage: bash submit_cluster.sh axis1
set -euo pipefail
SET="${1:-axis1}"
cd "$(dirname "$0")/../.."
H=results/rq2-replan-axes
.venv/bin/python $H/run.py list "$SET" > $H/tasks_$SET.txt
case "$SET" in size|fleet|axis6) TIME=0-12:00:00 ;; *) TIME=0-03:00:00 ;; esac
source linux_server/slurm/rit_ssh.sh
rsync -az -e "$RSYNC_SSH" --exclude 'out/' --exclude 'dryrun/' --exclude 'dryrun6/' --exclude 'dryrun2/' --exclude 'logs/' --exclude '*.out' --exclude '__pycache__' \
    $H/ "$REMOTE:capstone/$H/"
N=$(wc -l < $H/tasks_$SET.txt)
"${SSH[@]}" "$REMOTE" SET="$SET" N="$N" TIME="$TIME" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone/linux_server
mkdir -p logs
D=~/capstone/results/rq2-replan-axes
for start in $(seq 0 1000 $((N - 1))); do
    end=$(( start + 999 < N - 1 ? start + 999 : N - 1 ))
    sed -n "$((start + 1)),$((end + 1))p" $D/tasks_$SET.txt > $D/tasks_${SET}_$start.txt
    sbatch --account=drl-scheduling --partition=sporc --time=$TIME --mem=4g --array=0-$((end - start)) \
        --job-name=replan-axes --export=ALL,SCRIPT=results/rq2-replan-axes/run.py,TASKS=$D/tasks_${SET}_$start.txt \
        slurm/twin_array.sbatch
done
REMOTE
