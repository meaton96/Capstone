#!/bin/bash -l
# Generic CPU array for twin (DES) experiment scripts under results/ (2026-10-06): task i runs
#   python $REPO/$SCRIPT <line i+1 of $TASKS>
# with one CPU, from the cluster copy of the repo (~/capstone: env/ synced by slurm/sync_env_to_rit.sh, the script and
# its inputs by the experiment's submit.sh). Scripts must skip finished outputs, so resubmitting resumes.
# Example: sbatch --account=drl-scheduling --partition=sporc --time=0-03:00:00 --array=0-199%200 --job-name=oracle-bc \
#            --export=ALL,SCRIPT=results/dev-oracle-bc/collect.py,TASKS=$HOME/capstone/results/dev-oracle-bc/tasks.txt \
#            slurm/twin_array.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4g
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
echo "[twin_array] start on $(hostname), job ${SLURM_JOB_ID:-?}, task ${SLURM_ARRAY_TASK_ID:-0}"   # before anything can fail
cd "${SLURM_SUBMIT_DIR:-$HOME/capstone/linux_server_v3}"
REPO="$(dirname "$PWD")"
unset PYTHONPATH
source "$REPO/.venv/bin/activate"
export OMP_NUM_THREADS=1
TASK="${SLURM_ARRAY_TASK_ID:-0}"
LINE="$(sed -n "$((TASK + 1))p" "${TASKS:?set TASKS}")"
echo "[twin_array] ${SCRIPT:?} $LINE on $(hostname), job ${SLURM_JOB_ID:-?}, task $TASK"
eval "set -- $LINE"
python "$REPO/$SCRIPT" "$@"
