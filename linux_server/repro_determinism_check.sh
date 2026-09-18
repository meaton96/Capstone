#!/usr/bin/env bash
##############################################################################
#  repro_determinism_check.sh
#
#  Verifies the SimTime determinism fix (Time.time -> fixed-tick clock in
#  FactoryOrchestrator/PhysicalMachine). Runs the SAME (config, rule, seed)
#  multiple times and checks whether the outcome is identical every time.
#
#  Two modes, both run by default:
#    sequential — each rule run N times back-to-back, one process at a time.
#                 Isolates "is this rule/config deterministic on its own."
#    parallel   — N rounds, each round launching every rule simultaneously in
#                 the background (mirrors run_generated.sh's real launch
#                 pattern, which is what originally exposed the bug: 9
#                 processes contending for the same CPU cores). This is the
#                 more realistic test — sequential can pass while parallel
#                 still fails if there's residual contention-sensitivity.
#
#  Each run uses -repeats 1, so every invocation of a given rule uses the
#  SAME base seed (42) from the scenario config — no seed drift between runs.
#
#  Usage:
#    ./repro_determinism_check.sh \
#        --exe ./capstone.x86_64 \
#        --config ./BatchConfigs/scenario_matrix_l008.json \
#        --configname l008_mf_high \
#        --rules SPT_SMPT,SPT_SRWT,FIFO_SRWT,SRT_SMPT,random \
#        --runs 3
#
#  Defaults target l008_mf_high, since that scenario showed the worst
#  same-seed swings (up to 88 percentage points of completion rate) in the
#  0908a vs 0908b comparison that first surfaced this bug.
##############################################################################

set -euo pipefail

EXE="./capstone.x86_64"
CONFIG="./BatchConfigs/scenario_matrix_l008.json"
CONFIGNAME="l008_mf_high"
RULES_CSV="SPT_SMPT,SPT_SRWT,FIFO_SRWT,SRT_SMPT,random"
RUNS=3
TIMESCALE=100
LOG_LEVEL="Low"
RESULTS_ROOT="./Results/repro_check"
MODE="both"   # sequential | parallel | both

while [[ $# -gt 0 ]]; do
case $1 in
--exe)         EXE="$2";          shift 2 ;;
--config)      CONFIG="$2";       shift 2 ;;
--configname)  CONFIGNAME="$2";   shift 2 ;;
--rules)       RULES_CSV="$2";    shift 2 ;;
--runs)        RUNS="$2";         shift 2 ;;
--timescale)   TIMESCALE="$2";    shift 2 ;;
--loglevel)    LOG_LEVEL="$2";    shift 2 ;;
--results)     RESULTS_ROOT="$2"; shift 2 ;;
--mode)        MODE="$2";         shift 2 ;;
*) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
esac
done

[[ ! -f "$EXE" ]]    && echo "[ERROR] Exe not found: $EXE" && exit 1
[[ ! -f "$CONFIG" ]] && echo "[ERROR] Config not found: $CONFIG" && exit 1

IFS=',' read -ra RULES <<< "$RULES_CSV"

echo "========================================================================"
echo "[Repro] Determinism check"
echo "[Repro] Exe:        $EXE"
echo "[Repro] Config:     $CONFIG (scenario=$CONFIGNAME)"
echo "[Repro] Rules:      ${RULES[*]}"
echo "[Repro] Runs/rule:  $RUNS  (each -repeats 1, same base seed every time)"
echo "[Repro] Mode:       $MODE"
echo "========================================================================"

run_one() {
    local rule="$1" run_idx="$2" out_subdir="$3"
    local suffix="_${rule}_run${run_idx}"
    local log_file="${RESULTS_ROOT}/${out_subdir}/worker${suffix}.log"
    mkdir -p "${RESULTS_ROOT}/${out_subdir}"
    "$EXE" \
        -batchmode -nographics \
        -batchconfig  "$CONFIG" \
        -configname   "$CONFIGNAME" \
        -rules        "$rule" \
        -outputsuffix "$suffix" \
        -outputdir    "repro_check/${out_subdir}" \
        -repeats      1 \
        -timescale    "$TIMESCALE" \
        -loglevel     "$LOG_LEVEL" \
        -logFile      "$log_file" \
        > /dev/null 2>&1 || true
}

# ── Sequential: one process at a time, N times per rule ────────────────────
if [[ "$MODE" == "sequential" || "$MODE" == "both" ]]; then
    echo ""
    echo "[Repro] --- Sequential runs (no contention) ---"
    for rule in "${RULES[@]}"; do
        for ((i=1; i<=RUNS; i++)); do
            echo "[Repro]   sequential: $rule run $i/$RUNS"
            run_one "$rule" "$i" "sequential"
        done
    done
fi

# ── Parallel: N rounds, every rule launched simultaneously each round ──────
if [[ "$MODE" == "parallel" || "$MODE" == "both" ]]; then
    echo ""
    echo "[Repro] --- Parallel runs (contention, mirrors run_generated.sh) ---"
    for ((i=1; i<=RUNS; i++)); do
        echo "[Repro]   parallel round $i/$RUNS: launching ${#RULES[@]} rules at once"
        declare -a PIDS=()
        for rule in "${RULES[@]}"; do
            run_one "$rule" "$i" "parallel" &
            PIDS+=("$!")
        done
        for pid in "${PIDS[@]}"; do wait "$pid" || true; done
        unset PIDS
    done
fi

echo ""
echo "[Repro] All runs complete. Comparing outcomes..."
echo ""

python3 "$(dirname "$0")/repro_compare.py" \
    --results-root "$RESULTS_ROOT" \
    --rules "$RULES_CSV" \
    --runs "$RUNS" \
    --mode "$MODE"
