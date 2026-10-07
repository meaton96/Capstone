#!/bin/bash
# rq2-realizable (10-07): 375 observable policies (15 fixed, 216 regime maps, 144 threshold rules) per instance, B2 and
# S6, seeds 0-39 (test) and 2000-2159 (fitting): one array per setting, task files tasks_<setting>.txt. B2 ~10 min per
# task, S6 ~50 min. SETTINGS selects (default both). run.py skips finished outputs, so resubmitting resumes.
set -euo pipefail
source "$(dirname "$0")/../../linux_server/slurm/rit_ssh.sh"
"${SSH[@]}" "$REMOTE" SETTINGS="'${SETTINGS:-B2 S6}'" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone/linux_server_v3
D=~/capstone/results/rq2-realizable
mkdir -p logs $D
for st in $SETTINGS; do
    for s in $(seq 0 39) $(seq 2000 2159); do echo "--setting $st --seed $s"; done > $D/tasks_$st.txt
    T=0-02:00:00; [[ $st == S6 ]] && T=0-06:00:00
    sbatch --account=drl-scheduling --partition=sporc --time=$T --array=0-199 --job-name=realiz-$st \
        --export=ALL,SCRIPT=results/rq2-realizable/run.py,TASKS=$D/tasks_$st.txt slurm/twin_array.sbatch
done
REMOTE
