#!/bin/bash
##############################################################################
#  run_batch.sh
#
#  Launches one Unity process per PDR rule simultaneously for:
#     - Generated job data   (optional, via --batch-config)
#     - Brandimarte benchmarks (optional, via --benchmark-dir)
#
#  EXAMPLE USAGE:
#     # Benchmark-only:
#     ./run_batch.sh --exe-path ./capstone.x86_64 \
#         --benchmark-dir ./BatchConfigs/Benchmarks \
#         --repeats 3 --timescale 100 --disruption low
#
#     # Generated data only:
#     ./run_batch.sh --exe-path ./capstone.x86_64 \
#         --batch-config ./BatchConfigs/BatchConfigs.json \
#         --repeats 1 --timescale 100 --disruption none
##############################################################################

# Default parameters
EXE_PATH="./capstone.x86_64"
BATCH_CONFIG=""
BENCHMARK_DIR=""
RESULTS_DIR="./Results"
REPEATS=1
TIMESCALE=100
LOG_LEVEL="Low"
DISRUPTION="none"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --exe-path) EXE_PATH="$2"; shift ;;
        --batch-config) BATCH_CONFIG="$2"; shift ;;
        --benchmark-dir) BENCHMARK_DIR="$2"; shift ;;
        --results-dir) RESULTS_DIR="$2"; shift ;;
        --repeats) REPEATS="$2"; shift ;;
        --timescale) TIMESCALE="$2"; shift ;;
        --log-level) LOG_LEVEL="$2"; shift ;;
        --disruption) DISRUPTION="$2"; shift ;;
        *) echo -e "\e[31m[Launcher] ERROR: Unknown parameter passed: $1\e[0m"; exit 1 ;;
    esac
    shift
done

# Validate: at least one data source must be provided
if [[ -z "$BATCH_CONFIG" && -z "$BENCHMARK_DIR" ]]; then
    echo -e "\e[31m[Launcher] ERROR: Provide at least one of --batch-config or --benchmark-dir.\e[0m"
    exit 1
fi

RULES=("SPT_SMPT" "SPT_SRWT" "LPT_MMUR" "LPT_SMPT" "SRT_SRWT" "SRT_SMPT" "LRT_MMUR" "SDT_SRWT" "random")

# PIDs array for cleanup
PIDS=()

# Cleanup function to prevent orphaned processes
cleanup() {
    echo -e "\n\e[31m[Cleanup] Interrupt detected. Terminating running workers...\e[0m"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
            echo -e "\e[31m  - Killed PID: $pid\e[0m"
        fi
    done
    exit 1
}
trap cleanup SIGINT SIGTERM

# Merge CSVs function
merge_csvs() {
    local OUT_PATH=$1
    local LABEL=$2
    shift 2
    local CSV_PATHS=("$@")

    echo -e "\n\e[36m[Launcher] Merging $LABEL CSVs...\e[0m"
    local HEADER_WRITTEN=0
    local ROWS_TOTAL=0

    for csv in "${CSV_PATHS[@]}"; do
        if [[ ! -f "$csv" ]]; then
            echo -e "\e[33m  [WARN] Missing: $csv\e[0m"
            continue
        fi

        local LINES=$(wc -l < "$csv")
        if [[ "$LINES" -lt 2 ]]; then continue; fi

        if [[ "$HEADER_WRITTEN" -eq 0 ]]; then
            head -n 1 "$csv" > "$OUT_PATH"
            HEADER_WRITTEN=1
        fi

        tail -n +2 "$csv" >> "$OUT_PATH"
        local DATA_LINES=$((LINES - 1))
        ROWS_TOTAL=$((ROWS_TOTAL + DATA_LINES))
        echo "  Merged $DATA_LINES rows from $(basename "$csv")"
    done

    if [[ "$HEADER_WRITTEN" -eq 1 ]]; then
        echo -e "\e[32m[Launcher] $ROWS_TOTAL total rows -> $OUT_PATH\e[0m"
    else
        echo -e "\e[33m[Launcher] No data rows found for $LABEL.\e[0m"
    fi
}

# Resolve paths
EXE_PATH=$(realpath "$EXE_PATH")
mkdir -p "$RESULTS_DIR"
RESULTS_DIR=$(realpath "$RESULTS_DIR")

RUN_GENERATED=0
if [[ -n "$BATCH_CONFIG" ]]; then
    if [[ ! -f "$BATCH_CONFIG" ]]; then
        echo -e "\e[33m[Launcher] WARNING: --batch-config '$BATCH_CONFIG' not found -- skipping.\e[0m"
    else
        BATCH_CONFIG=$(realpath "$BATCH_CONFIG")
        RUN_GENERATED=1
    fi
fi

RUN_BENCHMARKS=0
BM_RESULTS_DIR=""
if [[ -n "$BENCHMARK_DIR" ]]; then
    if [[ ! -d "$BENCHMARK_DIR" ]]; then
        echo -e "\e[33m[Launcher] WARNING: --benchmark-dir '$BENCHMARK_DIR' not found -- skipping.\e[0m"
    else
        BENCHMARK_DIR=$(realpath "$BENCHMARK_DIR")
        RUN_BENCHMARKS=1
        BM_RESULTS_DIR="$RESULTS_DIR/brandimarte"
        mkdir -p "$BM_RESULTS_DIR"
    fi
fi

if [[ $RUN_GENERATED -eq 0 && $RUN_BENCHMARKS -eq 0 ]]; then
    echo -e "\e[31m[Launcher] ERROR: No valid data sources found. Exiting.\e[0m"
    exit 1
fi

TOTAL_WORKERS=$(( ${#RULES[@]} * (RUN_GENERATED + RUN_BENCHMARKS) ))
echo -e "\e[36m[Launcher] Starting $TOTAL_WORKERS workers simultaneously...\e[0m"
echo "Exe:         $EXE_PATH"
[[ $RUN_GENERATED -eq 1 ]] && echo "BatchConfig: $BATCH_CONFIG"
[[ $RUN_BENCHMARKS -eq 1 ]] && echo "Benchmarks:  $BENCHMARK_DIR"
echo "Results:     $RESULTS_DIR"
echo "Params:      Repeats: $REPEATS | Timescale: ${TIMESCALE}x | Disruption: $DISRUPTION"
echo ""

START_TIME=$(date +%s)

# Wave 1: Generated Data
if [[ $RUN_GENERATED -eq 1 ]]; then
    echo -e "\e[36m[Launcher] Wave 1: Generated job-data workers\e[0m"
    for rule in "${RULES[@]}"; do
        LOG_FILE="$RESULTS_DIR/worker_${rule}.log"
        echo "  Spawning generated worker: $rule"
        
        "$EXE_PATH" -batchmode -nographics \
            -batchconfig "$BATCH_CONFIG" \
            -rules "$rule" \
            -outputsuffix "_${rule}" \
            -repeats "$REPEATS" \
            -timescale "$TIMESCALE" \
            -loglevel "$LOG_LEVEL" \
            -disruption "$DISRUPTION" \
            -logFile "$LOG_FILE" &
            
        PIDS+=($!)
    done
fi

# Wave 2: Benchmarks
if [[ $RUN_BENCHMARKS -eq 1 ]]; then
    echo ""
    echo -e "\e[36m[Launcher] Wave 2: Brandimarte benchmark workers\e[0m"
    for rule in "${RULES[@]}"; do
        LOG_FILE="$RESULTS_DIR/worker_bm_${rule}.log"
        echo "  Spawning benchmark worker: $rule"
        
        "$EXE_PATH" -batchmode -nographics \
            -benchmarkdir "$BENCHMARK_DIR" \
            -rules "$rule" \
            -outputsuffix "_bm_${rule}" \
            -outputdir "brandimarte" \
            -repeats "$REPEATS" \
            -timescale "$TIMESCALE" \
            -loglevel "$LOG_LEVEL" \
            -disruption "$DISRUPTION" \
            -logFile "$LOG_FILE" &
            
        PIDS+=($!)
    done
fi

echo -e "\n\e[33m[Launcher] All workers launched. Waiting for completion...\e[0m"

# Polling and progress reporting
LAST_REPORT=$START_TIME
while true; do
    RUNNING_COUNT=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            RUNNING_COUNT=$((RUNNING_COUNT + 1))
        fi
    done

    if [[ $RUNNING_COUNT -eq 0 ]]; then
        break
    fi

    CURRENT_TIME=$(date +%s)
    if (( CURRENT_TIME - LAST_REPORT >= 30 )); then
        ELAPSED_MIN=$(echo "scale=1; ($CURRENT_TIME - $START_TIME) / 60" | bc)
        echo -e "\e[36m[Launcher] $RUNNING_COUNT workers still running (${ELAPSED_MIN} min elapsed)\e[0m"
        LAST_REPORT=$CURRENT_TIME
    fi
    sleep 5
done

TOTAL_MIN=$(echo "scale=1; ($(date +%s) - $START_TIME) / 60" | bc)
echo -e "\n\e[32m[Launcher] All workers finished in ${TOTAL_MIN} min.\e[0m"

# MERGE PHASE
if [[ $RUN_GENERATED -eq 1 ]]; then
    GEN_EPISODES=()
    GEN_MACHINES=()
    for rule in "${RULES[@]}"; do
        GEN_EPISODES+=("$RESULTS_DIR/baseline_results_${rule}.csv")
        GEN_MACHINES+=("$RESULTS_DIR/machine_utilization_${rule}.csv")
    done
    merge_csvs "$RESULTS_DIR/results.csv" "generated (episodes)" "${GEN_EPISODES[@]}"
    merge_csvs "$RESULTS_DIR/machine_utilization.csv" "generated (machine utilization)" "${GEN_MACHINES[@]}"
fi

if [[ $RUN_BENCHMARKS -eq 1 ]]; then
    BM_EPISODES=()
    BM_MACHINES=()
    for rule in "${RULES[@]}"; do
        BM_EPISODES+=("$BM_RESULTS_DIR/baseline_results_bm_${rule}.csv")
        BM_MACHINES+=("$BM_RESULTS_DIR/machine_utilization_bm_${rule}.csv")
    done
    merge_csvs "$BM_RESULTS_DIR/results.csv" "brandimarte (episodes)" "${BM_EPISODES[@]}"
    merge_csvs "$BM_RESULTS_DIR/machine_utilization.csv" "brandimarte (machine utilization)" "${BM_MACHINES[@]}"
fi

echo -e "\n\e[36m[Launcher] Processing Complete.\e[0m"