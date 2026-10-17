#!/bin/bash -l
# rq2-oracle-bc-future: one BC model per array task on one A100 (slurm/twin_array.sbatch with the GPU venv): task i runs
#   python $REPO/$SCRIPT <line i+1 of $TASKS> --device cuda
# from ~/capstone/linux_server (logs/). bc_train.py skips a model whose scores exist, so resubmitting resumes.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32g
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
echo "[gpu_array] start on $(hostname), job ${SLURM_JOB_ID:-?}, task ${SLURM_ARRAY_TASK_ID:-0}"
cd "${SLURM_SUBMIT_DIR:-$HOME/capstone/linux_server}"
REPO="$(dirname "$PWD")"
unset PYTHONPATH
source "$REPO/.venv-gpu/bin/activate"
nvidia-smi --query-gpu=name --format=csv,noheader || true
TASK="${SLURM_ARRAY_TASK_ID:-0}"
LINE="$(sed -n "$((TASK + 1))p" "${TASKS:?set TASKS}")"
echo "[gpu_array] ${SCRIPT:?} $LINE"
eval "set -- $LINE"
python "$REPO/$SCRIPT" "$@" --device cuda --threads 8
