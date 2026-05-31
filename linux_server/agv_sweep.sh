#!/usr/bin/env bash
##############################################################################
#  run_agv_sweep.sh
#
#  Brandimarte AGV fleet-size sensitivity sweep with a capped worker pool.
#  Runs at most --maxworkers processes simultaneously; queues the remainder
#  and feeds them in as slots open up.
#
#  Usage:
#    chmod +x run_agv_sweep.sh
#
#    # Deterministic sweep, fleet sizes 5-30 in steps of 5, 18 workers max:
#    ./run_agv_sweep.sh \
#        --exe         ./capstone.x86_64 \
#        --benchmarks  ./BatchConfigs/Benchmarks \
#        --results     ./Results \
#        --repeats     3 \
#        --timescale   100 \
#        --loglevel    Low \
#        --disruption  none \
#        --agvcounts   5,10,15,20,25,30 \
#        --maxworkers  18
#
#    # Deterministic + stochastic_low (108 total jobs, 18 at a time = 6 waves):
#    ./run_agv_sweep.sh ... --disruption none,low --agvcounts 5,10,15,20,25,30
#
#  Output layout:
#    Results/agv_sweep/agv<N>/<disruption>/results_bm_<rule>.csv   (per-worker)
#    Results/agv_sweep/agv<N>/<disruption>/merged_results.csv
#    Results/agv_sweep/agv<N>/<disruption>/merged_machine_utilization.csv
#    Results/agv_sweep/agv<N>/<disruption>/merged_agv_performance.csv
#    Results/agv_sweep/agv<N>/<disruption>/merged_segment_congestion.csv
##############################################################################

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
EXE=""
BENCHMARK_DIR=""
RESULTS_DIR="./Results"
REPEATS=3
TIMESCALE=100
LOG_LEVEL="Low"
DISRUPTION_LIST="none"
AGV_COUNT_LIST="5,10,15,20,25,30"
MAX_WORKERS=18

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
        --agvcounts)   AGV_COUNT_LIST="$2";   shift 2 ;;
        --maxworkers)  MAX_WORKERS="$2";      shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ -z "$EXE" ]]; then
    echo "[ERROR] --exe is required."; exit 1
fi
if [[ ! -f "$EXE" ]]; then
    echo "[ERROR] Executable not found: $EXE"; exit 1
fi
if [[ -z "$BENCHMARK_DIR" ]]; then
    echo "[ERROR] --benchmarks is required."; exit 1
fi
if [[ ! -d "$BENCHMARK_DIR" ]]; then
    echo "[ERROR] Benchmark directory not found: $BENCHMARK_DIR"; exit 1
fi

IFS=',' read -ra DISRUPTIONS <<< "$DISRUPTION_LIST"
IFS=',' read -ra AGV_COUNTS  <<< "$AGV_COUNT_LIST"

# ── Build job queue ───────────────────────────────────────────────────────────
# Each job is stored as a set of parallel indexed arrays.
# Order: agv_count → disruption → rule  (matches merge phase loops).
declare -a Q_AGV=()
declare -a Q_DISRUPTION=()
declare -a Q_RULE=()
declare -a Q_OUTDIR=()
declare -a Q_SUFFIX=()
declare -a Q_LOGFILE=()
declare -a Q_LABEL=()

for AGV_COUNT in "${AGV_COUNTS[@]}"; do
    AGV_COUNT="${AGV_COUNT// /}"
    for DISRUPTION in "${DISRUPTIONS[@]}"; do
        DISRUPTION="${DISRUPTION// /}"
        OUT_DIR="${RESULTS_DIR}/agv_sweep/agv${AGV_COUNT}/${DISRUPTION}"
        for RULE in "${RULES[@]}"; do
            Q_AGV+=("$AGV_COUNT")
            Q_DISRUPTION+=("$DISRUPTION")
            Q_RULE+=("$RULE")
            Q_OUTDIR+=("$OUT_DIR")
            Q_SUFFIX+=("_bm_${RULE}")
            Q_LOGFILE+=("${OUT_DIR}/worker_${RULE}.log")
            Q_LABEL+=("agv${AGV_COUNT}/${DISRUPTION}/${RULE}")
        done
    done
done

TOTAL_JOBS=${#Q_AGV[@]}

echo "========================================================================"
echo "[Launcher] AGV fleet-size sensitivity sweep (capped worker pool)"
echo "[Launcher] Exe:          $EXE"
echo "[Launcher] Benchmarks:   $BENCHMARK_DIR"
echo "[Launcher] Results:      $RESULTS_DIR"
echo "[Launcher] AGV counts:   ${AGV_COUNTS[*]}"
echo "[Launcher] Disruptions:  ${DISRUPTIONS[*]}"
echo "[Launcher] Rules:        ${RULES[*]}"
echo "[Launcher] Total jobs:   $TOTAL_JOBS"
echo "[Launcher] Max workers:  $MAX_WORKERS  (~$(( (TOTAL_JOBS + MAX_WORKERS - 1) / MAX_WORKERS )) sequential waves)"
echo "[Launcher] Repeats:      $REPEATS  |  Timescale: ${TIMESCALE}x  |  LogLevel: $LOG_LEVEL"
echo "========================================================================"

mkdir -p "$RESULTS_DIR"

# ── Worker pool ───────────────────────────────────────────────────────────────
declare -a POOL_PIDS=()
declare -a POOL_LABELS=()
NEXT_JOB=0
COMPLETED=0
SCRIPT_START=$SECONDS
LAST_REPORT=$SECONDS

spawn_worker() {
    local i=$1
    mkdir -p "${Q_OUTDIR[$i]}"

    "$EXE" \
        -batchmode -nographics \
        -benchmarkdir   "$BENCHMARK_DIR" \
        -rules          "${Q_RULE[$i]}" \
        -outputsuffix   "${Q_SUFFIX[$i]}" \
        -outputdir      "agv_sweep/agv${Q_AGV[$i]}/${Q_DISRUPTION[$i]}" \
        -repeats        "$REPEATS" \
        -timescale      "$TIMESCALE" \
        -loglevel       "$LOG_LEVEL" \
        -disruption     "${Q_DISRUPTION[$i]}" \
        -agvcount       "${Q_AGV[$i]}" \
        -logFile        "${Q_LOGFILE[$i]}" \
        > /dev/null 2>&1 &

    echo $!
}

# Seed the pool with the first wave
while [[ $NEXT_JOB -lt $TOTAL_JOBS && ${#POOL_PIDS[@]} -lt $MAX_WORKERS ]]; do
    PID=$(spawn_worker "$NEXT_JOB")
    POOL_PIDS+=("$PID")
    POOL_LABELS+=("${Q_LABEL[$NEXT_JOB]}")
    echo "[Pool] → Spawned ${Q_LABEL[$NEXT_JOB]}  (PID $PID)"
    NEXT_JOB=$(( NEXT_JOB + 1 ))
done

echo ""
echo "[Pool] Initial wave launched (${#POOL_PIDS[@]} workers). Entering run loop..."
echo ""

# Main run loop — reap finished, fill empty slots, report progress
while [[ ${#POOL_PIDS[@]} -gt 0 ]]; do

    # ── Reap any finished workers ─────────────────────────────────────────
    declare -a LIVE_PIDS=()
    declare -a LIVE_LABELS=()
    for i in "${!POOL_PIDS[@]}"; do
        if kill -0 "${POOL_PIDS[$i]}" 2>/dev/null; then
            LIVE_PIDS+=("${POOL_PIDS[$i]}")
            LIVE_LABELS+=("${POOL_LABELS[$i]}")
        else
            COMPLETED=$(( COMPLETED + 1 ))
            echo "[Pool] ✓ ${POOL_LABELS[$i]}  ($COMPLETED/$TOTAL_JOBS done)"
        fi
    done

    # Reassign — handle empty array safely
    if [[ ${#LIVE_PIDS[@]} -gt 0 ]]; then
        POOL_PIDS=("${LIVE_PIDS[@]}")
        POOL_LABELS=("${LIVE_LABELS[@]}")
    else
        POOL_PIDS=()
        POOL_LABELS=()
    fi

    # ── Fill empty slots ──────────────────────────────────────────────────
    while [[ $NEXT_JOB -lt $TOTAL_JOBS && ${#POOL_PIDS[@]} -lt $MAX_WORKERS ]]; do
        PID=$(spawn_worker "$NEXT_JOB")
        POOL_PIDS+=("$PID")
        POOL_LABELS+=("${Q_LABEL[$NEXT_JOB]}")
        echo "[Pool] → Spawned ${Q_LABEL[$NEXT_JOB]}  (PID $PID, slot ${#POOL_PIDS[@]}/${MAX_WORKERS})"
        NEXT_JOB=$(( NEXT_JOB + 1 ))
    done

    # ── Periodic summary every 60s ────────────────────────────────────────
    NOW=$SECONDS
    if (( NOW - LAST_REPORT >= 60 )); then
        ELAPSED_MIN=$(( (NOW - SCRIPT_START) / 60 ))
        QUEUED=$(( TOTAL_JOBS - NEXT_JOB ))
        echo ""
        echo "[Pool] ── Status: $COMPLETED done / ${#POOL_PIDS[@]} running / $QUEUED queued  (${ELAPSED_MIN} min elapsed)"
        LAST_REPORT=$NOW
        echo ""
    fi

    sleep 5
done

TOTAL_MIN=$(( (SECONDS - SCRIPT_START) / 60 ))
echo ""
echo "[Pool] All $TOTAL_JOBS jobs finished in ~${TOTAL_MIN} min."

# ── Merge phase ───────────────────────────────────────────────────────────────
merge_csvs() {
    local out_file="$1"
    local label="$2"
    shift 2
    local inputs=("$@")

    local header_written=0
    local rows_total=0

    for csv in "${inputs[@]}"; do
        if [[ ! -f "$csv" ]]; then
            echo "  [WARN] Missing: $(basename "$csv")"; continue
        fi

        local line_count
        line_count=$(wc -l < "$csv")
        if (( line_count < 2 )); then
            echo "  [WARN] No data rows: $(basename "$csv")"; continue
        fi

        if [[ $header_written -eq 0 ]]; then
            head -1 "$csv" > "$out_file"
            header_written=1
        fi

        tail -n +2 "$csv" >> "$out_file"
        local data_rows=$(( line_count - 1 ))
        rows_total=$(( rows_total + data_rows ))
        echo "  Merged $data_rows rows from $(basename "$csv")"
    done

    if [[ $header_written -eq 1 ]]; then
        echo "  ✅ $rows_total total rows → $(basename "$out_file")"
    else
        echo "  ❌ No files merged for: $label"
    fi
}

echo ""
echo "[Launcher] Merging CSVs..."

for AGV_COUNT in "${AGV_COUNTS[@]}"; do
    AGV_COUNT="${AGV_COUNT// /}"
    for DISRUPTION in "${DISRUPTIONS[@]}"; do
        DISRUPTION="${DISRUPTION// /}"
        OUT_DIR="${RESULTS_DIR}/agv_sweep/agv${AGV_COUNT}/${DISRUPTION}"

        echo ""
        echo "────────────────────────────────────────────────────────────────────"
        echo "[Launcher] Merging agv=${AGV_COUNT}  disruption=${DISRUPTION}"
        echo "────────────────────────────────────────────────────────────────────"

        declare -a RESULTS_CSVS=()
        declare -a MACHINE_CSVS=()
        declare -a AGV_CSVS=()
        declare -a SEGMENT_CSVS=()

        for RULE in "${RULES[@]}"; do
            RESULTS_CSVS+=( "${OUT_DIR}/results_bm_${RULE}.csv" )
            MACHINE_CSVS+=( "${OUT_DIR}/machine_utilization_bm_${RULE}.csv" )
            AGV_CSVS+=(     "${OUT_DIR}/agv_performance_bm_${RULE}.csv" )
            SEGMENT_CSVS+=( "${OUT_DIR}/segment_congestion_bm_${RULE}.csv" )
        done

        merge_csvs "${OUT_DIR}/merged_results.csv" \
            "results" "${RESULTS_CSVS[@]}"

        merge_csvs "${OUT_DIR}/merged_machine_utilization.csv" \
            "machine_utilization" "${MACHINE_CSVS[@]}"

        merge_csvs "${OUT_DIR}/merged_agv_performance.csv" \
            "agv_performance" "${AGV_CSVS[@]}"

        merge_csvs "${OUT_DIR}/merged_segment_congestion.csv" \
            "segment_congestion" "${SEGMENT_CSVS[@]}"

        unset RESULTS_CSVS MACHINE_CSVS AGV_CSVS SEGMENT_CSVS
    done
done

echo ""
echo "========================================================================"
echo "🎉 AGV sweep complete in ~${TOTAL_MIN} min."
echo ""
echo "Output layout:"
for AGV_COUNT in "${AGV_COUNTS[@]}"; do
    AGV_COUNT="${AGV_COUNT// /}"
    for DISRUPTION in "${DISRUPTIONS[@]}"; do
        DISRUPTION="${DISRUPTION// /}"
        echo "  ${RESULTS_DIR}/agv_sweep/agv${AGV_COUNT}/${DISRUPTION}/merged_*.csv"
    done
done
echo ""
echo "Suggested plot command after sweep:"
echo "  python plot_agv_sweep.py ${RESULTS_DIR}/agv_sweep --out plots/agv_sweep"
echo "========================================================================"