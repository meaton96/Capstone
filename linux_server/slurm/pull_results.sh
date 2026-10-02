#!/bin/bash
# Copy one experiment's results from the cluster into the local Results folder. Usage: slurm/pull_results.sh <exp>
# Results land in ./Results/<exp>/ (same layout the analysis scripts expect). Safe to re-run: only new files transfer.
# Cluster storage is not backed up, so pulling results is also the backup.
set -euo pipefail
EXP="${1:?usage: pull_results.sh <exp>}"
source "$(dirname "$0")/rit_ssh.sh"
mkdir -p "Results/$EXP"
rsync -az -e "$RSYNC_SSH" --info=progress2 "$REMOTE:$DEST/Results/$EXP/" "Results/$EXP/"
find "Results/$EXP" -name results.csv | wc -l | xargs echo "results.csv files:"
