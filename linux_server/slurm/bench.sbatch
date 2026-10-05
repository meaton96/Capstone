#!/bin/bash
# twin-compute benchmark (2026-10-04): one short PPO run per job, Unity or the event-based twin, CPU or GPU, with the
# same network, PPO settings and training distribution, for the thesis's compute comparison. train.py writes
# timing.json (env / inference / update seconds, steps per second in the loop, startup) into the run dir.
# Submit with slurm/submit_bench.sh (sets partition, CPUs, GPU and the variables below). Run from the player dir.
#   BACKEND unity|twin  NUM_ENVS  TORCH_THREADS  DEVICE cpu|cuda  TWIN_WORKERS (twin)  CONFIG (run name)
#   UPDATES (default 10): total steps = UPDATES x 128 x NUM_ENVS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/capstone/linux_server_v3}"
REPO="$(dirname "$PWD")"
spack env activate default-ml-x86_64-25111401 2>/dev/null || true
unset PYTHONPATH
export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
chmod +x capstone.x86_64
# GPU runs use .venv-gpu: the same packages as .venv (pip freeze) with torch 2.1.1+cu121 instead of +cpu.
if [[ "${DEVICE:-cpu}" == cuda ]]; then source "$REPO/.venv-gpu/bin/activate"; else source "$REPO/.venv/bin/activate"; fi
: "${BACKEND:?}" "${NUM_ENVS:?}" "${CONFIG:?}"
STEPS=$(( ${UPDATES:-10} * 128 * NUM_ENVS ))
PARAMS='{"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "load_mix": [{"name": "base"}, {"name": "utilhi", "utilization": [1.0, 1.8], "lull_utilization": [0.4, 0.7]}]}'
if [[ "$BACKEND" == unity ]]; then
    SIM=(--unity --unity-path "$PWD/capstone.x86_64" --no-graphics --base-worker-id $(( 100 + (SLURM_JOB_ID % 50) * 40 )))
else
    SIM=(--twin "$REPO/results/twin-compute/des_floor7.json" --twin-workers "${TWIN_WORKERS:-0}")
fi
echo "[bench] $CONFIG on $(hostname): $BACKEND, $NUM_ENVS envs, device ${DEVICE:-cpu}, ${SLURM_CPUS_PER_TASK:-?} CPUs, $STEPS steps"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
exec python "$REPO/env/train.py" "${SIM[@]}" \
    --scenario-generator randomized --random-warmup --episode-duration-seconds 5400 --params "$PARAMS" \
    --reward-spec "$REPO/env/config/rewards/tardiness.json" --train-seed 0 \
    --num-envs "$NUM_ENVS" --rollout-length 128 --batch-size 64 --total-timesteps "$STEPS" \
    --device "${DEVICE:-cpu}" --torch-threads "${TORCH_THREADS:-4}" \
    --stall-minutes 20 --notify-every-hours 0 \
    --results-dir "$REPO/results/twin-compute" --run-id "$CONFIG"
