#!/usr/bin/env bash
# AGV physical study rerun on the randomized training family (2026-09-29).
# Evaluation instances rnd_load_s0-8 (= generator seeds 0-8), current rules, player linux_server_dev (09-28 build).
#   AGVR       releasePrevious: fleet sweep on D + all 15 layouts at 7 / 12 AGVs, SPT_ECT and SRT_TECT,
#              LRT_MMUR as a stress rule at 12 / 15 AGVs only.
#   AGVR_hold  holdPrevious on D, same fleet sizes, for the protocol comparison (separate tree: paths omit protocol).
# Resumable: rerunning skips finished cells. Results: linux_server_dev/Results/{AGVR,AGVR_hold}/.
set -euo pipefail
cd "$(dirname "$0")"

EXE=../linux_server_dev/capstone.x86_64
WORKERS=${WORKERS:-16}
SCEN=$(for i in 0 1 2 3 4 5 6 7 8; do printf "rnd_load_s%d:1 " "$i"; done)
REL='-reservation releasePrevious -parking lane'
HOLD='-reservation holdPrevious -parking lane'
LAYOUTS="A B C D E F G H I J K L M N O"

q() { python3 run_experiment_queue.py --exe "$EXE" --workers "$WORKERS" --scenarios $SCEN "$@"; }

# 1. Fleet-size sweep on D (release)
q --exp AGVR --layouts D --agv 3 5 7 9 12 15 --rules SPT_ECT SRT_TECT --extra "$REL"
q --exp AGVR --layouts D --agv 12 15 --rules LRT_MMUR --extra "$REL"
# 2. Layout comparison (release); D cells at 7 / 12 are already done by step 1
q --exp AGVR --layouts $LAYOUTS --agv 7 12 --rules SPT_ECT SRT_TECT --extra "$REL"
q --exp AGVR --layouts $LAYOUTS --agv 12 --rules LRT_MMUR --extra "$REL"
# 3. Protocol comparison on D (hold); release side is step 1
q --exp AGVR_hold --layouts D --agv 3 5 7 9 12 15 --rules SPT_ECT SRT_TECT --extra "$HOLD"
q --exp AGVR_hold --layouts D --agv 12 15 --rules LRT_MMUR --extra "$HOLD"
echo "[AGVR] all sweeps done"
