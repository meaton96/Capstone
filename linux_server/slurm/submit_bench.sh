#!/bin/bash
# Submit the twin-compute benchmark matrix (one short job per config; see slurm/bench.sbatch). Run on the cluster from
# ~/capstone/linux_server_v3. ACCOUNT (default drl-scheduling), PARTITION (sporc), TIME (01:30:00), ONLY=<config> to
# submit one config, DRY=1 to print the commands.
#   config        backend envs device cpus threads twin_workers  (same network and PPO settings everywhere)
set -euo pipefail
mkdir -p logs
CONFIGS=(
    "unity-cpu-e6   unity 6  cpu  18 4  0"     # the production setup so far (train.sbatch)
    "unity-cpu-e12  unity 12 cpu  36 8  0"
    "unity-cpu-e24  unity 24 cpu  36 8  0"
    "unity-gpu-e12  unity 12 cuda 24 4  0"
    "unity-gpu-e24  unity 24 cuda 36 4  0"
    "twin-cpu-e16   twin  16 cpu  36 16 16"
    "twin-cpu-e32   twin  32 cpu  36 8  28"
    "twin-gpu-e16   twin  16 cuda 24 4  16"
    "twin-gpu-e32   twin  32 cuda 36 4  30"
)
for line in "${CONFIGS[@]}"; do
    read -r name backend envs device cpus threads workers <<< "$line"
    [[ -n "${ONLY:-}" && "$ONLY" != "$name" ]] && continue
    gres=()
    [[ "$device" == cuda ]] && gres=(--gres=gpu:a100:1)
    CMD=(sbatch --account="${ACCOUNT:-drl-scheduling}" --partition="${PARTITION:-sporc}" --time="${TIME:-01:30:00}"
         --job-name="tc-$name" --cpus-per-task="$cpus" --mem=64g "${gres[@]}"
         --export=ALL,BACKEND="$backend",NUM_ENVS="$envs",DEVICE="$device",TORCH_THREADS="$threads",TWIN_WORKERS="$workers",CONFIG="$name"
         slurm/bench.sbatch)
    if [[ "${DRY:-0}" == 1 ]]; then echo "${CMD[@]}"; else "${CMD[@]}"; fi
done
