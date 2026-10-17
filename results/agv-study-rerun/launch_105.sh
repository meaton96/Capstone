#!/bin/bash
# agv-study-105 (10-16, user): the 105-machine cells of agv-study-rerun after dev-agv105-probe. The full b2t-105 cells
# (5,893 jobs) run at 0.73 sim-s per wall-s and did not finish in 30 h, so b2t runs as regime b2tb = its first 1.5 h
# load block (1,685 jobs, make_b2tb.py), run to empty. b2m-105 runs as designed (3.7 sim-s per wall-s, a few hours).
# Same flags, fleets, rules and output folders as launch.sh, plus --loglevel Medium (sim.log stamps every job exit,
# so progress is visible):
#   agv-v5-linked   agv5_b2tb_t7open_s0 at 49 70 84 AGVs x MDD_PTECT MDD_PECT (6), agv5_b2m_t7open_s0 at 21 35 49 (6)
#   agv-v5-tiles    agv5_b2tb_t7iso_s0 at 49 70 84 x MDD_PTECT (3), agv5_b2m_t7iso_s0 at 21 35 49 (3)
# run_experiment_queue.py skips runs with results.csv, so a rerun resumes. 10-15/16: first run with W=2 (b2tb linked
# finished); restarted 10-16 18:50 on 7 slots (user). The b2tb-linked call runs first (it only resumes); the other
# three run in parallel so all slots are used: b2tb isolated on WA, b2m linked on WB, b2m isolated on WC.
#   env: W (b2tb linked, default 2), WA (default 3), WB (default 3), WC (default 1)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=results/agv-study-rerun
W=${W:-2}; WA=${WA:-3}; WB=${WB:-3}; WC=${WC:-1}
.venv/bin/python $H/make_b2tb.py > $H/make_b2tb.out
Q() { (cd linux_server && python3 run_experiment_queue.py "$@"); }
C="-baselinedrain -parking lane -allowunsafefleet -reservation releasePrevious"
{
    echo "== 105-machine start $(date -Is) (launch_105.sh, W=$W)"; echo "git HEAD $(git rev-parse HEAD)"
    echo "Simulation.dll $(md5sum linux_server/capstone_Data/Managed/Simulation.dll | cut -d' ' -f1)"
    echo "scenarios $(cat linux_server/BatchConfigs/Scenarios/agv5_b2tb_t7*_s0.json linux_server/BatchConfigs/Scenarios/agv5_b2m_t7*_s0.json | md5sum | cut -d' ' -f1)"
} >> $H/PROVENANCE.txt
{
    Q --exp agv-v5-linked --workers "$W" --timeout-hours 16 --loglevel Medium --scenarios agv5_b2tb_t7open_s0 \
      --layouts D --agv 49 70 84 --rules MDD_PTECT MDD_PECT --extra "$C"
} >> $H/launch_105.out 2>&1
Q --exp agv-v5-tiles --workers "$WA" --timeout-hours 16 --loglevel Medium --scenarios agv5_b2tb_t7iso_s0 \
  --layouts D --agv 49 70 84 --rules MDD_PTECT --extra "$C" >> $H/launch_105_b2tb_iso.out 2>&1 &
Q --exp agv-v5-linked --workers "$WB" --timeout-hours 16 --loglevel Medium --scenarios agv5_b2m_t7open_s0 \
  --layouts D --agv 21 35 49 --rules MDD_PTECT MDD_PECT --extra "$C" >> $H/launch_105_b2m_open.out 2>&1 &
Q --exp agv-v5-tiles --workers "$WC" --timeout-hours 16 --loglevel Medium --scenarios agv5_b2m_t7iso_s0 \
  --layouts D --agv 21 35 49 --rules MDD_PTECT --extra "$C" >> $H/launch_105_b2m_iso.out 2>&1 &
wait
n=0
for S in agv-v5-linked/agv5_b2tb_t7open_s0 agv-v5-tiles/agv5_b2tb_t7iso_s0 agv-v5-linked/agv5_b2m_t7open_s0 \
         agv-v5-tiles/agv5_b2m_t7iso_s0; do
    k=$(ls linux_server/Results/$S/*/agv*/results.csv 2>/dev/null | wc -l); echo "$S $k"; n=$((n + k))
done > $H/DONE_105.tmp
echo "total $n/18 $(date -Is)" >> $H/DONE_105.tmp
mv $H/DONE_105.tmp $H/DONE_105
