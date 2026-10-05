#!/bin/bash
# Submit (or resubmit = resume) a twin training run on one A100 per seed (slurm/train_twin.sbatch). Run on the cluster
# from ~/capstone/linux_server_v3. Required: RUN_NAME TOTAL_TIMESTEPS PARAMS WINDOW. Optional: SEEDS (3), FLOOR
# (~/capstone/floors/des_floor7_v3.json), ACCOUNT (drl-scheduling), PARTITION (sporc), TIME (0-08:00:00), CPUS (24),
# NUM_ENVS, TWIN_WORKERS, GPU (1; 0 = a 36-core CPU node, CPU torch with 16 threads: twin-compute twin-cpu-e16,
# ~150 steps/s, used when the A100 queue is backed up), DRY=1.
set -euo pipefail
: "${RUN_NAME:?}" "${TOTAL_TIMESTEPS:?}" "${PARAMS:?}" "${WINDOW:?}"
SEEDS="${SEEDS:-3}"
FLOOR="${FLOOR:-$HOME/capstone/floors/des_floor7_v3.json}"
mkdir -p logs
if [[ "${GPU:-1}" == 1 ]]; then
    DEVICE=cuda; TORCH_THREADS="${TORCH_THREADS:-4}"; RES=(--cpus-per-task="${CPUS:-24}" --gres=gpu:a100:1)
else
    DEVICE=cpu; TORCH_THREADS="${TORCH_THREADS:-16}"; RES=(--cpus-per-task="${CPUS:-36}")
fi
export RUN_NAME TOTAL_TIMESTEPS PARAMS WINDOW FLOOR DEVICE TORCH_THREADS
CMD=(sbatch --account="${ACCOUNT:-drl-scheduling}" --partition="${PARTITION:-sporc}" --time="${TIME:-0-08:00:00}"
     --job-name="$RUN_NAME" --array="${ARRAY:-0-$(( SEEDS - 1 ))}" "${RES[@]}" --mem=64g
     --export=ALL slurm/train_twin.sbatch)   # PARAMS is JSON (commas), so pass everything by environment
if [[ "${DRY:-0}" == 1 ]]; then echo "${CMD[@]}"; else "${CMD[@]}"; fi
