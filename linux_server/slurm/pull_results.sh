#!/bin/bash
# Copy one experiment's results from the cluster into the local Results folder. Usage: slurm/pull_results.sh <exp>
# Results land in ./Results/<exp>/ (same layout the analysis scripts expect). Safe to re-run: only new files transfer.
# Cluster storage is not backed up, so pulling results is also the backup.
set -euo pipefail
EXP="${1:?usage: pull_results.sh <exp>}"
REMOTE="${RIT_REMOTE:-me3870@sporcsubmit.rc.rit.edu}"
DEST="${RIT_DEST:-capstone/linux_server}"
mkdir -p "Results/$EXP"
rsync -az --info=progress2 "$REMOTE:$DEST/Results/$EXP/" "Results/$EXP/"
find "Results/$EXP" -name results.csv | wc -l | xargs echo "results.csv files:"
