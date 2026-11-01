#!/bin/bash
# Export one floor (des_floor.json) from the current player linux_server/ (5e6e0f87), flags as
# ../agv-ratio-scale/export_floor.sh. usage: export_floor.sh M [timeout_s]   -> floors/mM/
set -u
M=$1; TO=${2:-1800}
HERE=$(cd "$(dirname "$0")" && pwd)
P=$(cd "$HERE/../../linux_server" && pwd)
out=$HERE/floors/m$M
[ -f $out/des_floor.json ] && { echo "m$M exists"; exit 0; }
mkdir -p $out
cd $P
start=$(date +%s)
timeout $TO ./capstone.x86_64 -batchmode -nographics -scenario $HERE/floors/floor_m$M.json -agvcount $M -rules MDD_PTECT \
  -repeats 1 -timescale 100 -loglevel Low -layout D -outputdir $out -logFile $out/sim.log \
  -baselinedrain -destrace -reservation releasePrevious -parking lane > $out/player.out 2>&1
rc=$?
echo "m$M rc=$rc wall=$(( $(date +%s) - start ))s des_floor=$( [ -f $out/des_floor.json ] && echo yes || echo NO )"
