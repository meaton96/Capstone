#!/usr/bin/env bash
##############################################################################
#  run_batch_parallel.sh
#
#  Brandimarte-only parallel runner.
#  Launches ALL workers across ALL disruption levels simultaneously
#  (9 rules × N disruptions), waits for all to finish, then merges
#  the four CSV types per disruption level.
#
#  Mirrors RunBatchParallel.ps1 — same arguments, same output layout.
#
#  Usage:
#    chmod +x run_batch_parallel.sh
#
#    # Two disruption levels, 18 workers at once:
#    ./run_batch_parallel.sh \
#        --exe         ./capstone.x86_64 \
#        --benchmarks  ./BatchConfigs/Benchmarks \
#        --results     ./Results \
#        --repeats     3 \
#        --timescale   100 \
#        --loglevel    Low \
#        --disruption  none,low
#
#    # Three levels, 27 workers at once:
#    ./run_batch_parallel.sh ... --disruption none,low,high
#
#  Output layout:
#    Results/brandimarte/<disruption>/results_bm_<rule>.csv        (per-worker)
#    Results/brandimarte/<disruption>/merged_results.csv           (merged)
#    Results/brandimarte/<disruption>/merged_machine_utilization.csv
#    Results/brandimarte/<disruption>/merged_agv_performance.csv
#    Results/brandimarte/<disruption>/merged_segment_congestion.csv
##############################################################################

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
EXE=""
BENCHMARK_DIR=""
RESULTS_DIR="./Results"
REPEATS=3
TIMESCALE=100
LOG_LEVEL="Low"
DISRUPTION_LIST="none"          # comma-separated: none,low,high

RULES=(
    "SPT_SMPT"
    "SPT_SRWT"
    "LPT_MMUR"
    "LPT_SMPT"
    "SRT_SRWT"
    "SRT_SMPT"
    "LRT_MMUR"
    "SDT_SRWT"
    "random"
)

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --exe)         EXE="$2";              shift 2 ;;
        --benchmarks)  BENCHMARK_DIR="$2";    shift 2 ;;
        --results)     RESULTS_DIR="$2";      shift 2 ;;
        --repeats)     REPEATS="$2";          shift 2 ;;
        --timescale)   TIMESCALE="$2";        shift 2 ;;
        --loglevel)    LOG_LEVEL="$2";        shift 2 ;;
        --disruption)  DISRUPTION_LIST="$2";  shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ -z "$EXE" ]]; then
    echo "[ERROR] --exe is required."
    exit 1
fi
if [[ ! -f "$EXE" ]]; then
    echo "[ERROR] Executable not found: $EXE"
    exit 1
fi
if [[ -z "$BENCHMARK_DIR" ]]; then
    echo "[ERROR] --benchmarks is required."
    exit 1
fi
if [[ ! -d "$BENCHMARK_DIR" ]]; then
    echo "[ERROR] Benchmark directory not found: $BENCHMARK_DIR"
    exit 1
fi

# Convert comma-separated disruption list to array
IFS=',' read -ra DISRUPTIONS <<< "$DISRUPTION_LIST"

# ── Merge helper ──────────────────────────────────────────────────────────────
# merge_csvs <output_file> <label> <input_file1> [input_file2 ...]
merge_csvs() {
    local out_file="$1"
    local label="$2"
    shift 2
    local inputs=("$@")

    echo ""
    echo "[Merge] Merging $label → $(basename "$out_file")"

    local header_written=0
    local rows_total=0

    for csv in "${inputs[@]}"; do
        if [[ ! -f "$csv" ]]; then
            echo "  [WARN] Missing: $(basename "$csv")"
            continue
        fi

        local line_count
        line_count=$(wc -l < "$csv")

        if (( line_count < 2 )); then
            echo "  [WARN] No data rows in: $(basename "$csv")"
            continue
        fi

        if [[ $header_written -eq 0 ]]; then
            head -1 "$csv" > "$out_file"
            header_written=1
        fi

        # Append data rows (skip header)
        tail -n +2 "$csv" >> "$out_file"
        local data_rows=$(( line_count - 1 ))
        rows_total=$(( rows_total + data_rows ))
        echo "  Merged $data_rows rows from $(basename "$csv")"
    done

    if [[ $header_written -eq 1 ]]; then
        echo "  ✅ $rows_total total rows → $(basename "$out_file")"
    else
        echo "  ❌ No files found or merged for: $label"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

SCRIPT_START=$SECONDS

echo "========================================================================"
echo "[Launcher] Brandimarte parallel batch runner"
echo "[Launcher] Exe:          $EXE"
echo "[Launcher] Benchmarks:   $BENCHMARK_DIR"
echo "[Launcher] Results:      $RESULTS_DIR"
echo "[Launcher] Rules:        ${RULES[*]}"
echo "[Launcher] Disruptions:  ${DISRUPTIONS[*]}"
echo "[Launcher] Workers:      $(( ${#RULES[@]} * ${#DISRUPTIONS[@]} )) total (${#RULES[@]} rules × ${#DISRUPTIONS[@]} disruption levels)"
echo "[Launcher] Repeats:      $REPEATS  |  Timescale: ${TIMESCALE}x  |  LogLevel: $LOG_LEVEL"
echo "========================================================================"

# ── Spawn all workers across all disruption levels simultaneously ─────────────
declare -a ALL_PIDS=()
declare -a ALL_RULES=()
declare -a ALL_DISRUPTIONS=()

for DISRUPTION in "${DISRUPTIONS[@]}"; do
    DISRUPTION="${DISRUPTION// /}"
    BM_OUT_DIR="${RESULTS_DIR}/brandimarte/${DISRUPTION}"
    mkdir -p "$BM_OUT_DIR"

    echo ""
    echo "[Launcher] Spawning workers for disruption=$DISRUPTION → $BM_OUT_DIR"

    for RULE in "${RULES[@]}"; do
        SUFFIX="_bm_${RULE}"
        LOG_FILE="${BM_OUT_DIR}/worker_bm_${RULE}.log"

        "$EXE" \
            -batchmode -nographics \
            -benchmarkdir   "$BENCHMARK_DIR" \
            -rules          "$RULE" \
            -outputsuffix   "$SUFFIX" \
            -outputdir      "brandimarte/${DISRUPTION}" \
            -repeats        "$REPEATS" \
            -timescale      "$TIMESCALE" \
            -loglevel       "$LOG_LEVEL" \
            -disruption     "$DISRUPTION" \
            -logFile        "$LOG_FILE" \
            > /dev/null 2>&1 &

        PID=$!
        ALL_PIDS+=("$PID")
        ALL_RULES+=("$RULE")
        ALL_DISRUPTIONS+=("$DISRUPTION")
        echo "[Launcher]   Spawned [$DISRUPTION] $RULE  (PID $PID)"
    done
done

TOTAL_WORKERS=${#ALL_PIDS[@]}
echo ""
echo "[Launcher] All $TOTAL_WORKERS workers launched. Waiting for completion..."

# ── Poll until every worker is done ──────────────────────────────────────────
LAST_REPORT=$SECONDS

while true; do
    RUNNING=0
    STILL_RUNNING=()

    for i in "${!ALL_PIDS[@]}"; do
        if kill -0 "${ALL_PIDS[$i]}" 2>/dev/null; then
            RUNNING=$(( RUNNING + 1 ))
            STILL_RUNNING+=("[${ALL_DISRUPTIONS[$i]}] ${ALL_RULES[$i]}")
        fi
    done

    NOW=$SECONDS
    if (( NOW - LAST_REPORT >= 30 )); then
        DONE=$(( TOTAL_WORKERS - RUNNING ))
        ELAPSED_MIN=$(( (NOW - SCRIPT_START) / 60 ))
        echo "[Launcher] $DONE/$TOTAL_WORKERS done  (${ELAPSED_MIN} min elapsed)"
        if [[ ${#STILL_RUNNING[@]} -gt 0 ]]; then
            echo "[Launcher]   Still running: ${STILL_RUNNING[*]}"
        fi
        LAST_REPORT=$NOW
    fi

    [[ $RUNNING -eq 0 ]] && break
    sleep 5
done

TOTAL_MIN=$(( (SECONDS - SCRIPT_START) / 60 ))
echo ""
echo "[Launcher] All $TOTAL_WORKERS workers finished in ~${TOTAL_MIN} min."

# ── Merge phase — one set of 4 CSVs per disruption level ─────────────────────
echo ""
echo "[Launcher] Merging CSVs..."

for DISRUPTION in "${DISRUPTIONS[@]}"; do
    DISRUPTION="${DISRUPTION// /}"
    BM_OUT_DIR="${RESULTS_DIR}/brandimarte/${DISRUPTION}"

    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "[Launcher] Merging disruption=$DISRUPTION"
    echo "────────────────────────────────────────────────────────────────────"

    declare -a RESULTS_CSVS=()
    declare -a MACHINE_CSVS=()
    declare -a AGV_CSVS=()
    declare -a SEGMENT_CSVS=()

    for RULE in "${RULES[@]}"; do
        RESULTS_CSVS+=( "${BM_OUT_DIR}/results_bm_${RULE}.csv" )
        MACHINE_CSVS+=( "${BM_OUT_DIR}/machine_utilization_bm_${RULE}.csv" )
        AGV_CSVS+=(     "${BM_OUT_DIR}/agv_performance_bm_${RULE}.csv" )
        SEGMENT_CSVS+=( "${BM_OUT_DIR}/segment_congestion_bm_${RULE}.csv" )
    done

    merge_csvs "${BM_OUT_DIR}/merged_results.csv" \
        "results ($DISRUPTION)" \
        "${RESULTS_CSVS[@]}"

    merge_csvs "${BM_OUT_DIR}/merged_machine_utilization.csv" \
        "machine_utilization ($DISRUPTION)" \
        "${MACHINE_CSVS[@]}"

    merge_csvs "${BM_OUT_DIR}/merged_agv_performance.csv" \
        "agv_performance ($DISRUPTION)" \
        "${AGV_CSVS[@]}"

    merge_csvs "${BM_OUT_DIR}/merged_segment_congestion.csv" \
        "segment_congestion ($DISRUPTION)" \
        "${SEGMENT_CSVS[@]}"

    echo ""
    echo "[Launcher] Output files for disruption=$DISRUPTION:"
    echo "  ${BM_OUT_DIR}/merged_results.csv"
    echo "  ${BM_OUT_DIR}/merged_machine_utilization.csv"
    echo "  ${BM_OUT_DIR}/merged_agv_performance.csv"
    echo "  ${BM_OUT_DIR}/merged_segment_congestion.csv"

    unset RESULTS_CSVS MACHINE_CSVS AGV_CSVS SEGMENT_CSVS
done

echo ""
echo "========================================================================"
echo "🎉 All done in ~${TOTAL_MIN} min."
echo ""
echo "Merged output locations:"
for DISRUPTION in "${DISRUPTIONS[@]}"; do
    DISRUPTION="${DISRUPTION// /}"
    echo "  ${RESULTS_DIR}/brandimarte/${DISRUPTION}/merged_*.csv"
done
echo "========================================================================"
