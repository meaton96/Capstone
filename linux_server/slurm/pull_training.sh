#!/bin/bash
# Copy training run folders back from the cluster into the local results/. Run from the repo's linux_server/.
# Usage: slurm/pull_training.sh <RUN_NAME>   (pulls every seed: results/<RUN_NAME>_s*)
# Skips the periodic checkpoint_step*.pt files except the newest one per run, to keep the copy small.
set -euo pipefail
: "${1:?usage: pull_training.sh RUN_NAME}"
source "$(dirname "$0")/rit_ssh.sh"
SRC="$(dirname "$DEST")/results"
mkdir -p ../results
rsync -az -e "$RSYNC_SSH" --info=progress2 --exclude 'checkpoint_step*.pt' \
      "$REMOTE:$SRC/$1_s*" ../results/
echo "pulled $REMOTE:$SRC/$1_s* -> ../results/"
