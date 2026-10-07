#!/bin/bash
# base-eplen (10-06): 15 fixed pairs at agent windows 6 / 24 / 96 / 200 h, B2 seeds 0-39: mix B2 at regime blocks
# 1.5 / 6 h (320 tasks; 200 h ~ 20 CPU-min) and the stable control mix B0 (normal profile only) at 1.5 h (160 tasks).
# One array per mix, each with its own task file (tasks_<mix>.txt; the first B2 array, job 21818326, reads tasks.txt
# with the same B2 lines). MIXES selects (default both). Writes the task lists on the cluster and submits
# slurm/twin_array.sbatch (one CPU per task; run.py skips finished outputs, so resubmitting resumes). Code: env/ via
# linux_server/slurm/sync_env_to_rit.sh, scripts + inputs by rsync. Pull: the lab's pull_cmd.
set -euo pipefail
source "$(dirname "$0")/../../linux_server/slurm/rit_ssh.sh"
"${SSH[@]}" "$REMOTE" MIXES="'${MIXES:-B2 B0}'" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone/linux_server_v3
D=~/capstone/results/base-eplen
mkdir -p logs $D
for m in $MIXES; do
    bl="5400 21600"; [[ $m == B0 ]] && bl=5400
    for b in $bl; do for w in 21600 86400 345600 720000; do for s in $(seq 0 39); do
        echo "--seed $s --window $w --block $b --mix $m"; done; done; done > $D/tasks_$m.txt
    N=$(wc -l < $D/tasks_$m.txt)
    sbatch --account=drl-scheduling --partition=sporc --time=0-04:00:00 --array=0-$((N - 1)) --job-name=eplen-$m \
        --export=ALL,SCRIPT=results/base-eplen/run.py,TASKS=$D/tasks_$m.txt slurm/twin_array.sbatch
done
REMOTE
