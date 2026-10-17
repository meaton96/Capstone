#!/bin/bash -l
# rq4-agvfail-v5 (10-14): cluster array over $TASKS ("level group seed" per line), one Unity player per task, run by
# ../rq4-agvfail-v5-rules/task.sh into $ROOT. WID_BASE keeps the two experiments' worker ids (ports) apart on a node.
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
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS:?set TASKS}")"
set -- $LINE
PY=python bash results/rq4-agvfail-v5-rules/task.sh "${ROOT:?set ROOT}" "$1" "$2" "$3" $(( ${WID_BASE:-7000} + 4 * SLURM_ARRAY_TASK_ID ))
