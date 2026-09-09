#!/usr/bin/env bash
##############################################################################
#  run_scenarios.sh
#
#  Runs every hand-crafted scenario in BatchConfigs/Scenarios/ (ScenarioLoader,
#  -scenariodir) across all 9 dispatching rules, one worker process per rule
#  in parallel -- mirrors run_generated.sh's launch pattern exactly, just
#  pointed at -scenariodir instead of -batchconfig.
#
#  Each worker's -scenariodir run walks every .json file in the directory in
#  one process (RunMultiScenarioCoroutine), so a single results_<rule>.csv
#  ends up with one row per (scenario, repeat) for that rule -- the `instance`
#  column carries the scenario name, same convention as the generated l008
#  matrix. Repeats default to 3: scenario job sets are fully fixed (explicit
#  arrival times, durations, machine assignments), so with determinism fixed,
#  every rule except `random` should come out identical across repeats --
#  that's a free, cheap re-check of the determinism fix on top of the actual
#  scenario sweep.
#
#  Usage:
#    ./run_scenarios.sh \
#        --exe        ./capstone.x86_64 \
#        --scenarios  ./BatchConfigs/Scenarios \
#        --results    ./Results \
#        --repeats    3 \
#        --timescale  100 \
#        --loglevel   Low
##############################################################################

set -euo pipefail

EXE=""
SCENARIO_DIR="./BatchConfigs/Scenarios"
RESULTS_DIR="./Results"
REPEATS=3
TIMESCALE=100
LOG_LEVEL="Low"

RULES=(
"SPT_SMPT" "SPT_SRWT" "LPT_MMUR" "LPT_SMPT"
"SRT_SRWT" "SRT_SMPT" "LRT_MMUR" "FIFO_SRWT" "random"
)

while [[ $# -gt 0 ]]; do
case $1 in
--exe)         EXE="$2";           shift 2 ;;
--scenarios)   SCENARIO_DIR="$2";  shift 2 ;;
--results)     RESULTS_DIR="$2";   shift 2 ;;
--repeats)     REPEATS="$2";       shift 2 ;;
--timescale)   TIMESCALE="$2";     shift 2 ;;
--loglevel)    LOG_LEVEL="$2";     shift 2 ;;
*) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
esac
done

[[ -z "$EXE" ]]              && echo "[ERROR] --exe required"                       && exit 1
[[ ! -f "$EXE" ]]            && echo "[ERROR] Exe not found: $EXE"                  && exit 1
[[ ! -d "$SCENARIO_DIR" ]]   && echo "[ERROR] Scenario dir not found: $SCENARIO_DIR" && exit 1

N_SCENARIOS=$(find "$SCENARIO_DIR" -maxdepth 1 -name '*.json' | wc -l)
[[ "$N_SCENARIOS" -eq 0 ]] && echo "[ERROR] No .json scenario files in $SCENARIO_DIR" && exit 1

OUT_DIR="${RESULTS_DIR}/scenarios"
mkdir -p "$OUT_DIR"
SCRIPT_START=$SECONDS
declare -a ALL_PIDS=()

echo "========================================================================"
echo "[Launcher] Hand-crafted scenario sweep"
echo "[Launcher] Scenarios:     $N_SCENARIOS files in $SCENARIO_DIR"
echo "[Launcher] Output:        $OUT_DIR"
echo "[Launcher] Workers:       ${#RULES[@]} (one per rule)"
echo "[Launcher] Repeats/rule:  $REPEATS  |  Timescale: ${TIMESCALE}x  |  LogLevel: $LOG_LEVEL"
echo "========================================================================"

for RULE in "${RULES[@]}"; do
LOG_FILE="${OUT_DIR}/worker_${RULE}.log"
"$EXE" \
-batchmode -nographics \
-scenariodir  "$SCENARIO_DIR" \
-rules        "$RULE" \
-outputsuffix "_${RULE}" \
-outputdir    "scenarios" \
-repeats      "$REPEATS" \
-timescale    "$TIMESCALE" \
-loglevel     "$LOG_LEVEL" \
-logFile      "$LOG_FILE" \
> /dev/null 2>&1 &
PID=$!
ALL_PIDS+=("$PID")
echo "[Launcher]   Spawned $RULE  (PID $PID)"
done

echo ""
echo "[Launcher] All ${#ALL_PIDS[@]} workers launched. Waiting for completion..."

LAST_REPORT=$SECONDS
while true; do
RUNNING=0
for PID in "${ALL_PIDS[@]}"; do
kill -0 "$PID" 2>/dev/null && RUNNING=$((RUNNING + 1))
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
( cd "$RESULTS_DIR" && python3 merge.py -dir scenarios -out scenarios_merged )

echo ""
echo "Done in ~$(( (SECONDS - SCRIPT_START) / 60 )) min."
echo "Per-rule output:  ${OUT_DIR}/results_<rule>.csv (+ machine_utilization/agv_performance/etc.)"
echo "Merged output:    ${RESULTS_DIR}/scenarios_merged/results.csv (+ merged siblings)"
