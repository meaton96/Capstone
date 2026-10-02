#!/bin/bash -l
# rq2-oracle-screen on the cluster: one array task = one (regime, seed) switching-oracle job (env/switch_oracle.py).
# Submit from the player folder to use (rq2-oracle-screen: ~/capstone/linux_server_oracle, the 09-30 player;
# rq2-oracle-screen-qfifo: ~/capstone/linux_server_qfifo, the 10-02 strict-FIFO player), synced with
# RIT_DEST=capstone/<folder> slurm/sync_to_rit.sh . TASKS = a file with one "regime seed" per line; task i runs
# line i+1. EXP = results folder under ~/capstone/results (default rq2-oracle-screen); it needs regimes.json.
# Example: sbatch --account=drl-scheduling --partition=sporc --time=0-05:00:00 --array=0-143%60 \
#            --export=ALL,TASKS=$HOME/capstone/results/rq2-oracle-screen/tasks.txt slurm/oracle.sbatch
#SBATCH --job-name=oracle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/capstone/linux_server_oracle}"
REPO="$(dirname "$PWD")"
OUT="$REPO/results/${EXP:-rq2-oracle-screen}"

spack env activate default-ml-x86_64-25111401 2>/dev/null || true
unset PYTHONPATH
export LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}"
chmod +x capstone.x86_64
source "$REPO/.venv/bin/activate"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
read -r REGIME SEED < <(sed -n "$((TASK + 1))p" "${TASKS:?set TASKS}")
DIR="$OUT/$REGIME/s$SEED"
if [[ -f "$DIR/result.json" ]]; then echo "[oracle] $REGIME s$SEED already done"; exit 0; fi
mkdir -p "$DIR"
mapfile -t EXTRA < <(python -c "import json;[print(a) for a in json.load(open('$OUT/regimes.json'))['$REGIME']['args']]")

echo "[oracle] $REGIME s$SEED on $(hostname), job ${SLURM_JOB_ID:-?}, task $TASK"
python "$REPO/env/switch_oracle.py" --unity-path "$PWD/capstone.x86_64" --seed "$SEED" \
    --scenario-generator randomized --random-warmup --episode-duration-seconds 5400 \
    --segment-seconds "${SEGMENT:-1800}" --reward-spec "$REPO/env/config/rewards/flow_time.json" \
    --base-worker-id $(( 100 + (TASK % 400) * 2 )) --out "$DIR" "${EXTRA[@]}" > "$DIR/oracle.log" 2>&1
tail -n 1 "$DIR/oracle.log"
