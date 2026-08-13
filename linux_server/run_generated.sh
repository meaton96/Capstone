#!/usr/bin/env bash
set -euo pipefail
EXE=""
BATCH_CONFIG=""
RESULTS_DIR="./Results"
REPEATS=3
TIMESCALE=100
LOG_LEVEL="Low"
BASELINE_DRAIN=0          # off by default; --baselinedrain enables per-frame heuristic drain
RULES=(
"SPT_SMPT" "SPT_SRWT" "LPT_MMUR" "LPT_SMPT"
"SRT_SRWT" "SRT_SMPT" "LRT_MMUR" "SDT_SRWT" "random"
)
while [[ $# -gt 0 ]]; do
case $1 in
--exe)          EXE="$2";           shift 2 ;;
--batchconfig)  BATCH_CONFIG="$2";  shift 2 ;;
--results)      RESULTS_DIR="$2";   shift 2 ;;
--repeats)      REPEATS="$2";       shift 2 ;;
--timescale)    TIMESCALE="$2";     shift 2 ;;
--loglevel)     LOG_LEVEL="$2";     shift 2 ;;
--baselinedrain) BASELINE_DRAIN=1;  shift 1 ;;   # presence flag, no value
*) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
esac
done
[[ -z "$EXE" ]]          && echo "[ERROR] --exe required"         && exit 1
[[ ! -f "$EXE" ]]        && echo "[ERROR] Exe not found: $EXE"    && exit 1
[[ -z "$BATCH_CONFIG" ]] && echo "[ERROR] --batchconfig required" && exit 1
[[ ! -f "$BATCH_CONFIG" ]] && echo "[ERROR] Config not found: $BATCH_CONFIG" && exit 1

# Build the optional baseline-drain arg as an array so it expands to nothing when off.
DRAIN_ARGS=()
(( BASELINE_DRAIN == 1 )) && DRAIN_ARGS=( -baselinedrain true )

OUT_DIR="${RESULTS_DIR}/generated"
mkdir -p "$OUT_DIR"
SCRIPT_START=$SECONDS
declare -a ALL_PIDS=()
echo "========================================================================"
echo "[Launcher] Generated jobs parallel runner"
echo "[Launcher] Config:        $BATCH_CONFIG"
echo "[Launcher] Output:        $OUT_DIR"
echo "[Launcher] Workers:       ${#RULES[@]} (one per rule)"
echo "[Launcher] Timescale:     ${TIMESCALE}x"
echo "[Launcher] BaselineDrain: $( (( BASELINE_DRAIN == 1 )) && echo ENABLED || echo off )"
echo "========================================================================"
for RULE in "${RULES[@]}"; do
LOG_FILE="${OUT_DIR}/worker_${RULE}.log"
"$EXE" \
-batchmode -nographics \
-batchconfig  "$BATCH_CONFIG" \
-rules        "$RULE" \
-outputsuffix "_${RULE}" \
-outputdir    "generated" \
-repeats      "$REPEATS" \
-timescale    "$TIMESCALE" \
-loglevel     "$LOG_LEVEL" \
-logFile      "$LOG_FILE" \
"${DRAIN_ARGS[@]}" \
> /dev/null 2>&1 &
PID=$!
ALL_PIDS+=("$PID")
echo "[Launcher] Spawned $RULE (PID $PID)"
done
echo ""
echo "[Launcher] All ${#ALL_PIDS[@]} workers launched. Waiting..."
LAST_REPORT=$SECONDS
while true; do
RUNNING=0
for PID in "${ALL_PIDS[@]}"; do
kill -0 "$PID" 2>/dev/null && RUNNING=$(( RUNNING + 1 ))
done
NOW=$SECONDS
if (( NOW - LAST_REPORT >= 30 )); then
DONE=$(( ${#ALL_PIDS[@]} - RUNNING ))
echo "[Launcher] $DONE/${#ALL_PIDS[@]} done  ($(( (NOW - SCRIPT_START) / 60 )) min elapsed)"
LAST_REPORT=$NOW
fi
    [[ $RUNNING -eq 0 ]] && break
sleep 5
done
echo ""
echo "[Launcher] All workers finished. Merging CSVs..."
merge_csvs() {
local out_file="$1"; local label="$2"; shift 2
local header_written=0; local rows_total=0
for csv in "$@"; do
        [[ ! -f "$csv" ]] && echo "  [WARN] Missing: $(basename "$csv")" && continue
        (( $(wc -l < "$csv") < 2 )) && continue
        [[ $header_written -eq 0 ]] && head -1 "$csv" > "$out_file" && header_written=1
tail -n +2 "$csv" >> "$out_file"
rows_total=$(( rows_total + $(wc -l < "$csv") - 1 ))
done
echo "  $label → $rows_total rows"
}
declare -a R=() M=() A=() S=() T=()
for RULE in "${RULES[@]}"; do
R+=( "${OUT_DIR}/results_${RULE}.csv" )
M+=( "${OUT_DIR}/machine_utilization_${RULE}.csv" )
A+=( "${OUT_DIR}/agv_performance_${RULE}.csv" )
S+=( "${OUT_DIR}/segment_congestion_${RULE}.csv" )
T+=( "${OUT_DIR}/throughput_${RULE}.csv" )
done
merge_csvs "${OUT_DIR}/merged_results.csv"              "results"              "${R[@]}"
merge_csvs "${OUT_DIR}/merged_machine_utilization.csv"  "machine_utilization"  "${M[@]}"
merge_csvs "${OUT_DIR}/merged_agv_performance.csv"      "agv_performance"      "${A[@]}"
merge_csvs "${OUT_DIR}/merged_segment_congestion.csv"   "segment_congestion"   "${S[@]}"
merge_csvs "${OUT_DIR}/merged_throughput.csv"           "throughput"           "${T[@]}"
echo ""
echo "Done in ~$(( (SECONDS - SCRIPT_START) / 60 )) min."
echo "Output: ${OUT_DIR}/merged_*.csv"