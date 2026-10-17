#!/bin/bash
# agv-study-layouts-fail (10-15, user): the 15-layout check and the failures-on cells of agv-study-rerun, split out of its
# 105-machine stream. That stream (TW workers) spent 30 h on four linked 105-machine cells that hit --timeout-hours
# 30 without a result, so these 15-machine cells never started; it was stopped (10-15 22:40) and the 105-machine cells
# wait on dev-agv105-probe. Cells, settings and output folders are exactly launch.sh's:
#   agv-v5-layouts   releasePrevious, A B E F H I K L M N O x agv5_b2m_s0, agv5_b2t_s0 x 7 / 15 AGVs x MDD_PTECT (44)
#   agv-v5-fail-rel / agv-v5-fail-hold   agv5_b2tf_s0-s2 (machine failures on), D x 5 7 10 15 20 x MDD_PTECT (15 + 15)
# then the twin pairing of agv-v5-layouts (launch.sh repeats it at its end; existing pairs are reused). run_experiment_queue.py
# skips runs with results.csv, so a rerun resumes.   env: W (default 6)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=results/agv-study-rerun
W=${W:-6}
Q() { (cd linux_server && python3 run_experiment_queue.py "$@"); }
C="-baselinedrain -parking lane -allowunsafefleet"
{
    echo "== layouts + failures-on start $(date -Is) (launch_layouts_fail.sh)"; echo "git HEAD $(git rev-parse HEAD)"
    echo "Simulation.dll $(md5sum linux_server/capstone_Data/Managed/Simulation.dll | cut -d' ' -f1)"
    echo "scenarios $(cat linux_server/BatchConfigs/Scenarios/agv5_*.json | md5sum | cut -d' ' -f1)"
} >> $H/PROVENANCE.txt
{
    Q --exp agv-v5-layouts --workers "$W" --timeout-hours 20 --scenarios agv5_b2m_s0 agv5_b2t_s0 \
      --layouts A B E F H I K L M N O --agv 7 15 --rules MDD_PTECT --extra "$C -destrace -reservation releasePrevious"
    Q --exp agv-v5-fail-rel --workers "$W" --timeout-hours 20 --scenarios agv5_b2tf_s0 agv5_b2tf_s1 agv5_b2tf_s2 \
      --layouts D --agv 5 7 10 15 20 --rules MDD_PTECT --extra "$C -destrace -reservation releasePrevious"
    Q --exp agv-v5-fail-hold --workers "$W" --timeout-hours 20 --scenarios agv5_b2tf_s0 agv5_b2tf_s1 agv5_b2tf_s2 \
      --layouts D --agv 5 7 10 15 20 --rules MDD_PTECT --extra "$C -destrace -reservation holdPrevious"
} >> $H/launch_layouts_fail.out 2>&1
.venv/bin/python results/scripts/des_twin_gap.py linux_server/Results/agv-v5-layouts --out $H/twin --tag agv-v5-layouts \
    --twin-out twin_agv-v5-layouts --workers "$W" > $H/twin_agv-v5-layouts.out 2>&1 \
    || echo "twin pairing failed for agv-v5-layouts (see $H/twin_agv-v5-layouts.out)"
n=0
for X in agv-v5-layouts agv-v5-fail-rel agv-v5-fail-hold; do
    k=$(ls linux_server/Results/$X/*/*/agv*/results.csv 2>/dev/null | wc -l); echo "$X $k"; n=$((n + k))
done > $H/DONE_layouts_fail.tmp
echo "total $n/74 $(date -Is)" >> $H/DONE_layouts_fail.tmp
mv $H/DONE_layouts_fail.tmp $H/DONE_layouts_fail
