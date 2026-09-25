#!/bin/bash -l
# PPO training run on one node (see slurm/submit_train.sh, which sets account/partition/time and the variables).
# One array task = one training seed (TRAIN_SEED_BASE + task id), each with its own run folder
# ~/capstone/results/<RUN_NAME>_s<seed>. Resubmitting the same command resumes every seed from its newest
# checkpoint, so a run can span several jobs.
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=48g
#SBATCH --signal=B:TERM@600
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/capstone/linux_server}"     # the folder holding capstone.x86_64
REPO="$(dirname "$PWD")"

spack env activate default-ml-x86_64-25111401 2>/dev/null || true   # same runtime as run_queue.sbatch for the player
export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
chmod +x capstone.x86_64
source "$REPO/.venv/bin/activate"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$(( ${TRAIN_SEED_BASE:-0} + TASK ))
RUN_ID="${RUN_NAME:?set RUN_NAME}_s${SEED}"
RUN_DIR="$REPO/results/$RUN_ID"

RESUME=()
if [[ -d "$RUN_DIR" ]]; then
    LATEST="$(ls -t "$RUN_DIR"/checkpoint.pt "$RUN_DIR"/checkpoint_step*.pt 2>/dev/null | head -1 || true)"
    [[ -n "$LATEST" ]] && RESUME=(--resume-from "$LATEST")
fi

echo "[train.sbatch] $RUN_ID on $(hostname), job ${SLURM_JOB_ID:-?}, ${SLURM_CPUS_PER_TASK:-?} CPUs, resume: ${RESUME[*]:-none}"
# exec: the batch shell becomes python, so --signal=B:TERM reaches train.py (SIGTERM -> final checkpoint).
exec python "$REPO/env/train.py" \
    --unity --unity-path "$PWD/capstone.x86_64" --no-graphics \
    --scenario-generator randomized --random-warmup --episode-duration-seconds "${EPISODE_SECONDS:-5400}" \
    --reward-spec "$REPO/env/config/rewards/flow_time.json" \
    --train-seed "$SEED" \
    --num-envs "${NUM_ENVS:-6}" --rollout-length 128 --batch-size 64 \
    --total-timesteps "${TOTAL_TIMESTEPS:?set TOTAL_TIMESTEPS}" \
    --ent-coef "${ENT_COEF:-0.01}" --ent-coef-final "${ENT_COEF_FINAL:-0.001}" \
    --torch-threads "${TORCH_THREADS:-4}" \
    --save-every "${SAVE_EVERY:-25}" \
    --base-worker-id $(( 100 + TASK * 20 )) \
    --results-dir "$REPO/results" --run-id "$RUN_ID" \
    "${RESUME[@]}"
