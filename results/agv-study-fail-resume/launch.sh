#!/bin/bash
# agv-study-fail-resume (10-28, user): the 30 failures-on cells of agv-study-rerun (agv-v5-fail-rel / -fail-hold)
# rerun on the preempt-resume player, release cells first:
#   agv-study-fail-resume-rel / -hold   agv5_b2tf_s0-s2 (machine failures on), D x 5 7 10 15 20 x MDD_PTECT (15 + 15)
# Settings and scenarios exactly as launch_layouts_fail.sh of agv-study-rerun. run_experiment_queue.py skips runs with
# results.csv, so a rerun resumes. Refuses to start unless linux_server/ holds the resume player. env: W (default 6)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=results/agv-study-fail-resume
W=${W:-6}
WANT=22891261e82b72d3bb3dc0fbc3764e38
GOT=$(md5sum linux_server/capstone_Data/Managed/Simulation.dll | cut -d' ' -f1)
[ "$GOT" = "$WANT" ] || { echo "linux_server/ Simulation.dll is $GOT, not the resume player $WANT" >&2; exit 1; }
Q() { (cd linux_server && python3 run_experiment_queue.py "$@"); }
C="-baselinedrain -parking lane -allowunsafefleet"
{
    echo "== start $(date -Is)"; echo "git HEAD $(git rev-parse HEAD)"; echo "Simulation.dll $GOT"
    echo "scenarios $(cat linux_server/BatchConfigs/Scenarios/agv5_b2tf_*.json | md5sum | cut -d' ' -f1)"
} >> $H/PROVENANCE.txt
{
    Q --exp agv-study-fail-resume-rel --workers "$W" --timeout-hours 20 --scenarios agv5_b2tf_s0 agv5_b2tf_s1 agv5_b2tf_s2 \
      --layouts D --agv 5 7 10 15 20 --rules MDD_PTECT --extra "$C -destrace -reservation releasePrevious"
    Q --exp agv-study-fail-resume-hold --workers "$W" --timeout-hours 20 --scenarios agv5_b2tf_s0 agv5_b2tf_s1 agv5_b2tf_s2 \
      --layouts D --agv 5 7 10 15 20 --rules MDD_PTECT --extra "$C -destrace -reservation holdPrevious"
} >> $H/launch.out 2>&1
n=0
for X in agv-study-fail-resume-rel agv-study-fail-resume-hold; do
    k=$(ls linux_server/Results/$X/*/*/agv*/results.csv 2>/dev/null | wc -l); echo "$X $k"; n=$((n + k))
done > $H/DONE.tmp
echo "total $n/30 $(date -Is)" >> $H/DONE.tmp
mv $H/DONE.tmp $H/DONE
