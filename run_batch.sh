#!/usr/bin/env bash
##############################################################################
#  run_batch_parallel.sh
#
#  Launches one Unity process per PDR rule simultaneously, waits for all to
#  finish, then merges their CSVs into a single results.csv.
#
#  Usage:
#    chmod +x run_batch_parallel.sh
#    ./run_batch_parallel.sh \
#        --exe        "./capstone.x86_64" \
#        --config     "./BatchConfigs/BatchConfigs.json" \
#        --results    "./Results" \
#        --repeats    3 \
#        --timescale  100 \
#        --loglevel   Low
#
#  macOS app bundle:
#    --exe "./capstone.app/Contents/MacOS/capstone"
##############################################################################

    

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────
EXE=""
BATCH_CONFIG=""
RESULTS_DIR="./Results"
REPEATS=1
TIMESCALE=100
LOG_LEVEL="Low"

RULES=(
    "SPT_SMPT"
    "SPT_SRWT"
    "LPT_MMUR"
    "LPT_SMPT"
    "SRT_SRWT"
    "SRT_SMPT"
    "LRT_MMUR"
    "SDT_SRWT"
)

# ── Parse args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --exe)        EXE="$2";          shift 2 ;;
        --config)     BATCH_CONFIG="$2"; shift 2 ;;
        --results)    RESULTS_DIR="$2";  shift 2 ;;
        --repeats)    REPEATS="$2";      shift 2 ;;
        --timescale)  TIMESCALE="$2";    shift 2 ;;
        --loglevel)   LOG_LEVEL="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXE" || -z "$BATCH_CONFIG" ]]; then
    echo "Usage: $0 --exe <path> --config <path> [--results ./Results] [--repeats 1] [--timescale 100] [--loglevel Low]"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "[Launcher] Starting ${#RULES[@]} workers simultaneously..."
echo "[Launcher] Exe:     $EXE"
echo "[Launcher] Config:  $BATCH_CONFIG"
echo "[Launcher] Results: $RESULTS_DIR"
echo ""

PIDS=()

for RULE in "${RULES[@]}"; do
    SUFFIX="_${RULE}"
    LOG_FILE="${RESULTS_DIR}/worker_${RULE}.log"

    "$EXE" \
        -batchmode -nographics \
        -batchconfig "$BATCH_CONFIG" \
        -rules        "$RULE" \
        -outputsuffix "$SUFFIX" \
        -repeats      "$REPEATS" \
        -timescale    "$TIMESCALE" \
        -loglevel     "$LOG_LEVEL" \
        -logFile      "$LOG_FILE" \
        > /dev/null 2>&1 &

    PID=$!
    PIDS+=($PID)
    echo "[Launcher] Spawned worker $RULE  (PID $PID)"
done

echo ""
echo "[Launcher] All ${#PIDS[@]} workers launched. Waiting for completion..."

START_TIME=$SECONDS
LAST_REPORT=$SECONDS

# ── Poll progress ─────────────────────────────────────────────────────────
while true; do
    RUNNING=0
    for PID in "${PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            RUNNING=$((RUNNING + 1))
        fi
    done

    NOW=$SECONDS
    if (( NOW - LAST_REPORT >= 30 )); then
        ELAPSED=$(( (NOW - START_TIME) / 60 ))
        echo "[Launcher] $((${#PIDS[@]} - RUNNING))/${#PIDS[@]} done  (${ELAPSED} min elapsed)"
        LAST_REPORT=$NOW
    fi

    [[ $RUNNING -eq 0 ]] && break
    sleep 5
done

TOTAL_MIN=$(( (SECONDS - START_TIME) / 60 ))
echo ""
echo "[Launcher] All workers finished in ~${TOTAL_MIN} min."

# ── Merge CSVs ────────────────────────────────────────────────────────────
echo ""
echo "[Launcher] Merging CSV files..."

MERGED="${RESULTS_DIR}/results.csv"
HEADER_WRITTEN=0
ROWS_TOTAL=0

for RULE in "${RULES[@]}"; do
    CSV="${RESULTS_DIR}/results_${RULE}.csv"
    if [[ ! -f "$CSV" ]]; then
        echo "  [WARN] Missing: $CSV"
        continue
    fi

    LINE_COUNT=$(wc -l < "$CSV")
    if (( LINE_COUNT < 2 )); then
        echo "  [WARN] No data rows in: $CSV"
        continue
    fi

    if [[ $HEADER_WRITTEN -eq 0 ]]; then
        head -1 "$CSV" > "$MERGED"
        HEADER_WRITTEN=1
    fi

    # Append data rows (skip header)
    tail -n +2 "$CSV" >> "$MERGED"
    DATA_ROWS=$(( LINE_COUNT - 1 ))
    ROWS_TOTAL=$(( ROWS_TOTAL + DATA_ROWS ))
    echo "  Merged ${DATA_ROWS} rows from results_${RULE}.csv"
done

echo ""
echo "[Launcher] Done. ${ROWS_TOTAL} total rows written to:"
echo "  ${MERGED}"