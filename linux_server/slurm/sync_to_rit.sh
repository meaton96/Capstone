#!/bin/bash
# Copy the Unity player, scenario files, runner and slurm scripts to the cluster. Run from the repo's linux_server/.
# Usage: slurm/sync_to_rit.sh [player dir, default ../linux_server2]      Remote: me3870@sporcsubmit.rc.rit.edu:~/capstone/linux_server
# The whole player folder (~300 MB) is sent, minus Results/ and debug symbols. Re-runs only send what changed.
set -euo pipefail
PLAYER="${1:-../linux_server2}"
REMOTE="${RIT_REMOTE:-me3870@sporcsubmit.rc.rit.edu}"
DEST="${RIT_DEST:-capstone/linux_server}"
ssh "$REMOTE" "mkdir -p $DEST/BatchConfigs $DEST/slurm $DEST/logs"
rsync -az --info=progress2 --exclude 'Results/' --exclude '*BurstDebugInformation*' --exclude '__pycache__' \
      "$PLAYER"/ "$REMOTE:$DEST/"
rsync -az --exclude 'Results/' BatchConfigs/ "$REMOTE:$DEST/BatchConfigs/"
rsync -az run_experiment_queue.py "$REMOTE:$DEST/"
rsync -az slurm/ "$REMOTE:$DEST/slurm/"
echo "synced -> $REMOTE:$DEST"
