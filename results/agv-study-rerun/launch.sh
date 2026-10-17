#!/bin/bash
# agv-study-rerun (v5 design, 10-14; user: "the point is to use the new rules"): the AGV physical study (RQ1 / N4) on
# the current player (linux_server/, Simulation.dll 5e6e0f87, without -pyrules) with the thesis's v5 rules and
# objective. Instances: due-dated B2-family batch scenarios (agv5_scenarios.py: b2m machine-bound, b2t transport-bound
# with 90-150 s mean ops, b2tf = b2t + machine failures, 105-machine tiled / linked versions), run until every job has
# finished, so total tardiness is uncensored. Rules: MDD_PTECT (the thesis's best pair) and MDD_PECT (the same without
# the travel term). All with -baselinedrain -parking lane -allowunsafefleet (only lifts the protocol's fleet bound).
#   agv-v5-rel       releasePrevious, D J C G x agv5_b2m_s0/s1, agv5_b2t_s0/s1 x 2 3 5 7 10 15 20 25 AGVs x 2 rules (256)
#   agv-v5-hold      holdPrevious, D J x the same scenarios, fleets, rules (128)
#   agv-v5-layouts   releasePrevious, A B E F H I K L M N O x agv5_b2m_s0, agv5_b2t_s0 x 7 / 15 AGVs x MDD_PTECT (44)
#   agv-v5-fail-rel / agv-v5-fail-hold   agv5_b2tf_s0-s2 (machine failures on), D x 5 7 10 15 20 x MDD_PTECT (15 + 15)
#   agv-v5-tiles     105 machines, 7 isolated tiles (tiling block in the scenario, no -tiles flag), MDD_PTECT:
#                    b2m at 21 / 35 / 49 AGVs (3 / 5 / 7 per tile), b2t at 49 / 70 / 84 (7 / 10 / 12 per tile;
#                    15 per tile is refused on a linked floor: lane exits run into the seam bridges) (6)
#   agv-v5-linked    the same floors linked (job scope open, pooled AGVs), MDD_PTECT and MDD_PECT (12)
# Single-tile sweeps add -destrace; the twin pairing (DES-0 / 1g / 1k, results/scripts/des_twin_gap.py; failures-off sweeps only) runs at the
# end. Outputs linux_server/Results/<sweep>/<scenario>/<layout>/agv<N>_<RULE>_s<seed>/. run_experiment_queue.py skips
# runs with results.csv, so a rerun resumes. Two streams of about equal length (~28 h): release + hold on W workers;
# the 105-machine cells, then the 15-layout check and the failures-on cells, on TW workers. Batch mode paces frames by
# wall clock (no capture frame rate), about 10 sim-s per wall-s on 15 machines, so a 6 h B2-family cell takes ~40 min
# and b2t at 2-3 AGVs (long drain) 1-2 h.   env: W (default 12), TW (default 4)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=results/agv-study-rerun
W=${W:-12}; TW=${TW:-4}
.venv/bin/python $H/agv5_scenarios.py > $H/scenarios.out
Q() { (cd linux_server && python3 run_experiment_queue.py "$@"); }
C="-baselinedrain -parking lane -allowunsafefleet"
F="2 3 5 7 10 15 20 25"
S4="agv5_b2m_s0 agv5_b2m_s1 agv5_b2t_s0 agv5_b2t_s1"
R2="MDD_PTECT MDD_PECT"
{
    echo "== start $(date -Is)"; echo "git HEAD $(git rev-parse HEAD)"
    echo "Simulation.dll $(md5sum linux_server/capstone_Data/Managed/Simulation.dll | cut -d' ' -f1)"
    echo "des_twin $(cat env/des_twin/*.py | md5sum | cut -d' ' -f1)"
    echo "scenarios $(cat linux_server/BatchConfigs/Scenarios/agv5_*.json | md5sum | cut -d' ' -f1)"
} >> $H/PROVENANCE.txt
(
    Q --exp agv-v5-linked --workers "$TW" --timeout-hours 30 --scenarios agv5_b2t_t7open_s0 --layouts D --agv 49 70 84 \
      --rules $R2 --extra "$C -reservation releasePrevious"
    Q --exp agv-v5-tiles --workers "$TW" --timeout-hours 30 --scenarios agv5_b2t_t7iso_s0 --layouts D --agv 49 70 84 \
      --rules MDD_PTECT --extra "$C -reservation releasePrevious"
    Q --exp agv-v5-linked --workers "$TW" --timeout-hours 30 --scenarios agv5_b2m_t7open_s0 --layouts D --agv 21 35 49 \
      --rules $R2 --extra "$C -reservation releasePrevious"
    Q --exp agv-v5-tiles --workers "$TW" --timeout-hours 30 --scenarios agv5_b2m_t7iso_s0 --layouts D --agv 21 35 49 \
      --rules MDD_PTECT --extra "$C -reservation releasePrevious"
    Q --exp agv-v5-layouts --workers "$TW" --timeout-hours 20 --scenarios agv5_b2m_s0 agv5_b2t_s0 \
      --layouts A B E F H I K L M N O --agv 7 15 --rules MDD_PTECT --extra "$C -destrace -reservation releasePrevious"
    Q --exp agv-v5-fail-rel --workers "$TW" --timeout-hours 20 --scenarios agv5_b2tf_s0 agv5_b2tf_s1 agv5_b2tf_s2 \
      --layouts D --agv 5 7 10 15 20 --rules MDD_PTECT --extra "$C -destrace -reservation releasePrevious"
    Q --exp agv-v5-fail-hold --workers "$TW" --timeout-hours 20 --scenarios agv5_b2tf_s0 agv5_b2tf_s1 agv5_b2tf_s2 \
      --layouts D --agv 5 7 10 15 20 --rules MDD_PTECT --extra "$C -destrace -reservation holdPrevious"
) >> $H/launch_tiles.out 2>&1 &
TILES=$!
{
    Q --exp agv-v5-rel --workers "$W" --timeout-hours 20 --scenarios $S4 --layouts D J C G --agv $F --rules $R2 \
      --extra "$C -destrace -reservation releasePrevious"
    Q --exp agv-v5-hold --workers "$W" --timeout-hours 20 --scenarios $S4 --layouts D J --agv $F --rules $R2 \
      --extra "$C -destrace -reservation holdPrevious"
} >> $H/launch.out 2>&1
wait $TILES
echo "== end $(date -Is) Simulation.dll $(md5sum linux_server/capstone_Data/Managed/Simulation.dll | cut -d' ' -f1)" >> $H/PROVENANCE.txt
for X in agv-v5-rel agv-v5-hold agv-v5-layouts; do   # not the failures-on sweeps: the twin does not replay batch-mode failures (smoke)
    .venv/bin/python results/scripts/des_twin_gap.py linux_server/Results/$X --out $H/twin --tag $X --twin-out twin_$X \
        --workers 12 > $H/twin_$X.out 2>&1 || echo "twin pairing failed for $X (see $H/twin_$X.out)"
done
echo "unity runs: $(ls linux_server/Results/agv-v5-*/*/*/agv*/results.csv 2>/dev/null | wc -l)/476"
date -Is > $H/DONE
