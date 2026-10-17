#!/bin/bash -l
# rq3-gen-v5-unity-cmix (10-17): cluster array over tasks.txt ("cond group seed" per line), one Unity player per task.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd "$HOME/capstone"
unset PYTHONPATH
source .venv/bin/activate
export OMP_NUM_THREADS=1
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/rq3-gen-v5-unity-cmix/tasks.txt)"
set -- $LINE
PY=python bash results/rq3-gen-v5-unity-cmix/task.sh "$1" "$2" "$3" $((14000 + 4 * SLURM_ARRAY_TASK_ID))
