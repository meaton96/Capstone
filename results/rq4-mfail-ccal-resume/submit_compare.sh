#!/bin/bash
# rq4-mfail-ccal-resume, stage 2 (10-28; copied from rq4-mfail-ccal 10-14): pick S* from the pulled calibration
# (locally), ship cscale.json, then 12 comparison tasks (none / unif / unif-cal x 4 chunks of 10 test seeds, 7 rules
# each) as a cluster CPU array (~7 min each). Finished outputs are skipped, so resubmitting resumes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=results/rq4-mfail-ccal-resume
.venv/bin/python $H/mfail.py pick > $H/pick.out
.venv/bin/python $H/mfail.py list-compare > $H/tasks_compare.txt
source linux_server/slurm/rit_ssh.sh
rsync -az -e "$RSYNC_SSH" $H/mfail.py $H/cscale.json $H/tasks_compare.txt "$REMOTE:capstone/$H/"
"${SSH[@]}" "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd ~/capstone/linux_server; mkdir -p logs
D=~/capstone/results/rq4-mfail-ccal-resume
sbatch --account=drl-scheduling --partition=sporc --time=0-01:00:00 --mem=4g --array=0-11 --job-name=mfres-compare \
    --export=ALL,SCRIPT=results/rq4-mfail-ccal-resume/mfail.py,TASKS=$D/tasks_compare.txt slurm/twin_array.sbatch
REMOTE
