#!/usr/bin/env bash
##############################################################################
#  run_gridlock_sweep.sh
#
#  AGV-count sweep on ONE fixed scenario, to find where AGV traffic gridlocks
#  under the current single-lane (one-way) layout. One worker process per
#  (agvCount, rule) cell, capped worker pool, each cell running -repeats
#  seeds (seed drift = failure timing, so repeats only matter when the
#  scenario has machine failures on; a failures-off scenario is deterministic
#  and needs --repeats 1).
#
#  Output: Results/gridlock_sweep/<tag>/agv<N>_<RULE>/{results,agv_performance,...}.csv
#          Results/gridlock_sweep/<tag>/agv<N>_<RULE>/sim.log
#  Analyse with: python results/scripts/analyze_gridlock.py Results/gridlock_sweep/<tag>
#
#  Use --extra "-legacyorphanpredispatch" to disable the orphaned-pre-dispatch
#  reaper (FlagHarvester.ReleaseOrphanedPreDispatches) and reproduce the
#  pre-fix behaviour on the same build.
#
#  Usage:
#    ./run_gridlock_sweep.sh --exe ./capstone.x86_64 \
#        --scenario BatchConfigs/Scenarios/_mfsweep_control.json --tag nofail \
#        --agv 3,5,7,9,10,11,12,14 --rules SPT_SRWT,SPT_SMPT,LRT_MMUR,LPT_MMUR \
#        --repeats 1 --workers 20
##############################################################################

set -euo pipefail

EXE=""; SCENARIO=""; TAG=""; AGV_LIST="3,5,7,9,10,11,12,14"
RULES="SPT_SRWT,SPT_SMPT,LRT_MMUR,LPT_MMUR"
REPEATS=1; WORKERS=20; TIMESCALE=100; LOG_LEVEL="Low"; EXTRA=""; RESULTS_DIR="./Results"

while [[ $# -gt 0 ]]; do
case $1 in
--exe)       EXE="$2";       shift 2 ;;
--scenario)  SCENARIO="$2";  shift 2 ;;
--tag)       TAG="$2";       shift 2 ;;
--agv)       AGV_LIST="$2";  shift 2 ;;
--rules)     RULES="$2";     shift 2 ;;
--repeats)   REPEATS="$2";   shift 2 ;;
--workers)   WORKERS="$2";   shift 2 ;;
--timescale) TIMESCALE="$2"; shift 2 ;;
--loglevel)  LOG_LEVEL="$2"; shift 2 ;;
--extra)     EXTRA="$2";     shift 2 ;;
--results)   RESULTS_DIR="$2"; shift 2 ;;
*) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
esac
done

[[ -z "$EXE" || ! -f "$EXE" ]]           && { echo "[ERROR] --exe missing or not found"; exit 1; }
[[ -z "$SCENARIO" || ! -f "$SCENARIO" ]] && { echo "[ERROR] --scenario missing or not found"; exit 1; }
[[ -z "$TAG" ]]                          && { echo "[ERROR] --tag required"; exit 1; }

IFS=',' read -ra AGVS  <<< "$AGV_LIST"
IFS=',' read -ra RULEA <<< "$RULES"

OUT_ROOT="gridlock_sweep/${TAG}"
mkdir -p "${RESULTS_DIR}/${OUT_ROOT}"

echo "[Sweep] scenario=$SCENARIO tag=$TAG agv=${AGVS[*]} rules=${RULEA[*]} repeats=$REPEATS workers=$WORKERS extra='$EXTRA'"

run_cell() {
    local agv="$1" rule="$2"
    local cell="agv${agv}_${rule}"
    local out="${RESULTS_DIR}/${OUT_ROOT}/${cell}"
    mkdir -p "$out"
    # shellcheck disable=SC2086
    "$EXE" -batchmode -nographics \
        -scenario "$SCENARIO" \
        -agvcount "$agv" \
        -rules "$rule" \
        -repeats "$REPEATS" \
        -timescale "$TIMESCALE" \
        -loglevel "$LOG_LEVEL" \
        -outputdir "${OUT_ROOT}/${cell}" \
        -logFile "${out}/sim.log" \
        $EXTRA > /dev/null 2>&1
    echo "[Sweep] done ${cell}"
}
export -f run_cell
export EXE SCENARIO REPEATS TIMESCALE LOG_LEVEL EXTRA RESULTS_DIR OUT_ROOT

for agv in "${AGVS[@]}"; do
    for rule in "${RULEA[@]}"; do
        echo "$agv $rule"
    done
done | xargs -P "$WORKERS" -L 1 bash -c 'run_cell "$0" "$1"'

echo "[Sweep] all cells finished -> ${RESULTS_DIR}/${OUT_ROOT}"
