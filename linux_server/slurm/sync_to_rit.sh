#!/bin/bash
# Copy the Unity player, scenario files, runner and slurm scripts to the cluster. Run from the repo's linux_server/.
# Usage: slurm/sync_to_rit.sh [player dir, default ../linux_server2]      Remote: rit-research:~/capstone/linux_server
# The whole player folder (~300 MB) is sent, minus Results/ and debug symbols. Re-runs only send what changed.
set -euo pipefail
PLAYER="${1:-../linux_server2}"
source "$(dirname "$0")/rit_ssh.sh"
"${SSH[@]}" "$REMOTE" "mkdir -p $DEST/BatchConfigs $DEST/slurm $DEST/logs"
rsync -az -e "$RSYNC_SSH" --info=progress2 --exclude 'Results/' --exclude '*BurstDebugInformation*' --exclude '__pycache__' \
      "$PLAYER"/ "$REMOTE:$DEST/"
rsync -az -e "$RSYNC_SSH" --exclude 'Results/' BatchConfigs/ "$REMOTE:$DEST/BatchConfigs/"
rsync -az -e "$RSYNC_SSH" run_experiment_queue.py "$REMOTE:$DEST/"
rsync -az -e "$RSYNC_SSH" slurm/ "$REMOTE:$DEST/slurm/"
"${SSH[@]}" "$REMOTE" "cd $DEST && python3 slurm/patch_glibc.py UnityPlayer.so"   # cluster glibc is 2.34, build wants 2.35
echo "synced -> $REMOTE:$DEST"
