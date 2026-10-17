#!/bin/bash
# Submit a learning-stack dev run (dev_b2_slot_train_eval.sh) to the RIT cluster as one job (2026-10-07): CPU-only by
# default (DEVICE=cuda for an A100 job). Same
# arguments as the local script: NAME "SEEDS" WORKERS_PER_SEED [extra train.py flags...].
#
# - Code: results/NAME/code/ (the local snapshot) is shipped if it exists; otherwise env/ is synced
#   (linux_server/slurm/sync_env_to_rit.sh) and copied to results/NAME/code/env on the cluster AT
#   SUBMIT TIME, so a later sync never reaches a queued or running job. STAMP records the local git head and env diff.
# - The job runs results/scripts/dev_b2_slot_train_eval_cluster.sh (train all seeds in parallel, then evaluate on B2
#   seeds 0-39) with .venv-gpu. The analysis (dev_b2_slot_analyze.py, needs pandas) runs locally after the pull.
# - Resubmitting resumes (checkpoints, DONE_s<seed>, eval/DONE are skipped as in the local script).
# Lab record: launch this script; pull with results/scripts/pull_dev_cluster.sh NAME; done-glob results/NAME/eval/DONE.
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME="$1"; SEEDS="$2"; WORKERS="$3"; shift 3
source linux_server/slurm/rit_ssh.sh
(cd linux_server && bash slurm/sync_env_to_rit.sh)
"${SSH[@]}" "$REMOTE" "mkdir -p capstone/results/scripts capstone/results/$NAME"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-dev_b2_slot_train_eval_cluster.sh}"   # 10-14: optional variant (e.g. ..._c.sh); default unchanged
MEM="${MEM:-48g}"                # 10-15: per-seed jobs ask for less (MEM=16g)
rsync -az -e "$RSYNC_SSH" "results/scripts/$TRAIN_SCRIPT" "$REMOTE:capstone/results/scripts/"
if [ -d "results/$NAME/code/env" ]; then     # a run queued locally already has its snapshot: ship that one
    rsync -az -e "$RSYNC_SSH" "results/$NAME/code/" "$REMOTE:capstone/results/$NAME/code/"
fi
NS=$(wc -w <<< "$SEEDS")
DEVICE="${DEVICE:-cpu}"          # 10-07: cpu by default; A100 jobs waited 6+ h on priority, and twin training is
                                 # bound by the simulator (~16 agent steps/s), not the network
if [ "$DEVICE" = cuda ]; then
    CPUS=$(( NS * (WORKERS + 2) + 2 )); RES="--gres=gpu:a100:1"; VENV=.venv-gpu
else
    CPUS=$(( NS * (WORKERS + 3) + 2 )); RES=""; VENV=.venv
fi
HEAD="$(git rev-parse --short HEAD)"; DIFF="$(git diff HEAD -- env | sha256sum | cut -c1-12)"
ARGS="$(printf '%q ' "$NAME" "$SEEDS" "$WORKERS" "$@")"
"${SSH[@]}" "$REMOTE" bash -s <<REMOTE
set -euo pipefail
cd ~/capstone
mkdir -p results/scripts linux_server/logs
if [ ! -d results/$NAME/code/env ]; then
    mkdir -p results/$NAME/code
    rsync -a --exclude __pycache__ --exclude tests --exclude '*.egg-info' env/ results/$NAME/code/env/
    { echo "snapshot \$(date -Is) (cluster, at submit)"; echo "git_head $HEAD"; echo "diff_sha $DIFF"; } > results/$NAME/code/STAMP
fi
cd linux_server
sbatch --account=drl-scheduling --partition=sporc --time=1-00:00:00 $RES --cpus-per-task=$CPUS \
    --mem=${MEM:-48g} --job-name=$NAME --output=logs/%x_%j.out --error=logs/%x_%j.err \
    --wrap "cd \$HOME/capstone && DEVICE=$DEVICE PY=\$HOME/capstone/$VENV/bin/python bash results/scripts/$TRAIN_SCRIPT $ARGS"
REMOTE
