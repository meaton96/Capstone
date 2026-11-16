#!/bin/bash
# rq4-mfail-ccal-resume, stage 1 (10-28; copied from rq4-mfail-ccal 10-14): 7 calibration tasks (c scale S in 1.0..3.0,
# seeds 1000-1029, MDD-PTECT; S = 1 also without failures) as a cluster CPU array (twin_array.sbatch, ~3-4 min each),
# on this folder's code/env (the v5 snapshot + preempt-resume). Needs dev-pect-rule (harness.py, hetero.py,
# calib_models.json), rq3-gen-twin/run_task.py and the m50 floor on the cluster (shipped by rq4-mfail-ccal; resent here).
set -euo pipefail
cd "$(dirname "$0")/../.."
H=results/rq4-mfail-ccal-resume
.venv/bin/python $H/mfail.py list-calib > $H/tasks_calib.txt
source linux_server/slurm/rit_ssh.sh
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/$H/code capstone/results/rq4-mfail-ccal/out/calib capstone/results/dev-pect-rule capstone/results/agv-ratio-scale/floors/m50"
rsync -az -e "$RSYNC_SSH" $H/mfail.py $H/tasks_calib.txt "$REMOTE:capstone/$H/"
rsync -az --delete --exclude __pycache__ -e "$RSYNC_SSH" $H/code/ "$REMOTE:capstone/$H/code/"
rsync -az -e "$RSYNC_SSH" results/rq4-mfail-ccal/out/calib/s1.json "$REMOTE:capstone/results/rq4-mfail-ccal/out/calib/"
rsync -az -e "$RSYNC_SSH" results/dev-pect-rule/harness.py results/dev-pect-rule/hetero.py results/dev-pect-rule/calib_models.json \
    "$REMOTE:capstone/results/dev-pect-rule/"
rsync -az -e "$RSYNC_SSH" results/agv-ratio-scale/floors/m50/des_floor.json "$REMOTE:capstone/results/agv-ratio-scale/floors/m50/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone
[[ -f results/rq3-gen-twin/run_task.py ]] || { echo "missing rq3-gen-twin/run_task.py"; exit 1; }
.venv/bin/python results/rq4-mfail-ccal-resume/mfail.py check
cd linux_server; mkdir -p logs
D=~/capstone/results/rq4-mfail-ccal-resume
sbatch --account=drl-scheduling --partition=sporc --time=0-01:00:00 --mem=4g --array=0-6 --job-name=mfres-calib \
    --export=ALL,SCRIPT=results/rq4-mfail-ccal-resume/mfail.py,TASKS=$D/tasks_calib.txt slurm/twin_array.sbatch
REMOTE
