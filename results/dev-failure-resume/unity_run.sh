#!/bin/bash
# dev-failure-resume (10-28, copied from dev-csharp-failures 10-09): one Unity batch cell, as results/dev-twin-failures/unity_run.sh (layout D, release, lane
# parking, -baselinedrain -destrace, -timescale 100).
#   unity_run.sh <player dir> <scenario json> <rule> <agvs> <out dir> [extra player flags...]
# A cell whose out dir has results.csv is skipped, so a rerun resumes. LAYOUT (env, default D) sets -layout.
#   cells.sh-style use: a file of lines "<player> <scenario> <rule> <agvs> <out> [flags]" through
#   xargs -P N -L1 bash unity_run.sh
set -u
P=$1; SCEN=$2; RULE=$3; AGVS=$4; OUT=$5; shift 5
[ -f "$OUT/results.csv" ] && exit 0
rm -rf "$OUT"; mkdir -p "$OUT"
t0=$(date +%s)
( cd "$P" && timeout 5400 ./capstone.x86_64 -batchmode -nographics -scenario "$SCEN" \
    -agvcount "$AGVS" -rules "$RULE" -repeats 1 -timescale 100 -loglevel Low -layout "${LAYOUT:-D}" -outputdir "$OUT" \
    -logFile "$OUT/sim.log" -baselinedrain -destrace -reservation releasePrevious -parking lane "$@" \
    > "$OUT/player.out" 2>&1 )
rc=$?
echo "[$(date '+%H:%M:%S')] $(basename "$OUT") rc=$rc wall=$(( $(date +%s) - t0 ))s $( [ -f "$OUT/results.csv" ] && echo ok || echo MISSING )"
