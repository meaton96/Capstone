#!/bin/bash
# Copy the Python training code (env/) to the cluster, next to linux_server/ (~/capstone/env), so env/train.py
# finds the player and scenario files at the same relative paths as locally. Run from the repo's linux_server/.
# The player, scenarios and slurm scripts go with slurm/sync_to_rit.sh; run both after changing either side.
set -euo pipefail
source "$(dirname "$0")/rit_ssh.sh"
ENV_DEST="$(dirname "$DEST")/env"
"${SSH[@]}" "$REMOTE" "mkdir -p $ENV_DEST $(dirname "$DEST")/results"
rsync -az -e "$RSYNC_SSH" --delete --exclude '__pycache__' --exclude '*.egg-info' --exclude 'results/' \
      ../env/ "$REMOTE:$ENV_DEST/"
echo "synced env/ -> $REMOTE:$ENV_DEST"
