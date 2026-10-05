#!/bin/bash
# PPO training on the event-based twin, one GPU or CPU only (DEVICE=cpu) (2026-10-05; twin-compute: ~380 steps/s at 16 envs on an A100, so
# 3M steps in about 2 h). One array task per training seed. Submit with slurm/submit_train_twin.sh. Run from
# ~/capstone/linux_server_v3 (for logs/); the twin needs no player, only the floor export.
#   RUN_NAME  TOTAL_TIMESTEPS  PARAMS (generator JSON)  WINDOW (agent window, s)  FLOOR (des_floor.json)
#   NUM_ENVS (16)  TWIN_WORKERS (16)  REWARD (tardiness)  TRAIN_SEED_BASE (0)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@600
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/capstone/linux_server_v3}"
REPO="$(dirname "$PWD")"
unset PYTHONPATH
if [[ "${DEVICE:-cuda}" == cuda ]]; then source "$REPO/.venv-gpu/bin/activate"; else source "$REPO/.venv/bin/activate"; fi
TASK="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$(( ${TRAIN_SEED_BASE:-0} + TASK ))
RUN_ID="${RUN_NAME:?}_s${SEED}"
RUN_DIR="$REPO/results/$RUN_ID"
RESUME=()
if [[ -d "$RUN_DIR" ]]; then
    LATEST="$(ls -t "$RUN_DIR"/checkpoint.pt "$RUN_DIR"/checkpoint_step*.pt 2>/dev/null | head -1 || true)"
    [[ -n "$LATEST" ]] && RESUME=(--resume-from "$LATEST")
fi
echo "[train_twin] $RUN_ID on $(hostname), job ${SLURM_JOB_ID:-?}, ${SLURM_CPUS_PER_TASK:-?} CPUs, resume: ${RESUME[*]:-none}"
nvidia-smi --query-gpu=name --format=csv,noheader || true
exec python "$REPO/env/train.py" \
    --twin "${FLOOR:?}" --twin-workers "${TWIN_WORKERS:-16}" \
    --scenario-generator randomized --random-warmup --episode-duration-seconds "${WINDOW:?}" --params "${PARAMS:?}" \
    --reward-spec "$REPO/env/config/rewards/${REWARD:-tardiness}.json" \
    --train-seed "$SEED" \
    --num-envs "${NUM_ENVS:-16}" --rollout-length 128 --batch-size 64 \
    --total-timesteps "${TOTAL_TIMESTEPS:?}" \
    --ent-coef "${ENT_COEF:-0.01}" --ent-coef-final "${ENT_COEF_FINAL:-0.001}" \
    --device "${DEVICE:-cuda}" --torch-threads "${TORCH_THREADS:-4}" \
    --save-every "${SAVE_EVERY:-100}" \
    --stall-minutes "${STALL_MINUTES:-30}" --notify-every-hours 1 \
    --results-dir "$REPO/results" --run-id "$RUN_ID" \
    "${RESUME[@]}"
