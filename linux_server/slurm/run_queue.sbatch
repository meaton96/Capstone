#!/bin/bash -l
# Generic Slurm wrapper around run_experiment_queue.py. Do not sbatch this directly: use slurm/submit.sh, which sets
# the account/partition/time on the command line and passes the experiment through environment variables
# (EXP SCENARIOS LAYOUTS AGVS RULES EXTRA). One node per array task; task i of N runs shard i/N of the grid.
#SBATCH --job-name=capstone
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=120g
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/capstone/linux_server}"     # must be the folder holding run_experiment_queue.py

spack env activate default-ml-x86_64-25111401 2>/dev/null || true   # x86_64 (SPORC) only: the player is x86_64
export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
chmod +x capstone.x86_64

# The player uses ~2 cores each (Unity worker threads), so half as many workers as CPUs.
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-36} / 2 ))
SHARD="${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"

echo "[sbatch] $EXP shard $SHARD on $(hostname), $WORKERS workers, job ${SLURM_JOB_ID}"
python3 run_experiment_queue.py --exp "$EXP" --exe ./capstone.x86_64 --workers "$WORKERS" --shard "$SHARD" \
  --scenarios $SCENARIOS --layouts $LAYOUTS --agv $AGVS --rules $RULES --extra "$EXTRA"
