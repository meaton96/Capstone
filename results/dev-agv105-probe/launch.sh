#!/bin/bash
# dev-agv105-probe (10-15, user): how fast does the player run a 105-machine floor? agv-study-rerun's linked b2t cells
# ran 30 h without finishing (killed by --timeout-hours 30, no output). Two short cells, run to empty, with
# --loglevel Medium so sim.log stamps every job exit with wall time (progress even if a cell times out):
#   agv5_b2t_t7open_arr1h_s0 at 84 AGVs (linked, pooled; the heaviest killed cell, first hour = 1,186 jobs incl. the burst)
#   agv5_b2m_t7iso_arr1h_s0  at 21 AGVs (7 isolated tiles; the lightest 105-machine cell, 402 jobs)
# Same flags as launch.sh's 105-machine cells. 3 h timeout each, one worker each, in parallel.
set -uo pipefail
cd "$(dirname "$0")/../.."
H=results/dev-agv105-probe
.venv/bin/python $H/make_scenarios.py > $H/scenarios.out
Q() { (cd linux_server && python3 run_experiment_queue.py "$@"); }
C="-baselinedrain -parking lane -allowunsafefleet -reservation releasePrevious"
echo "== start $(date -Is) Simulation.dll $(md5sum linux_server/capstone_Data/Managed/Simulation.dll | cut -d' ' -f1)" >> $H/PROVENANCE.txt
Q --exp dev-agv105-probe --workers 1 --timeout-hours 3 --loglevel Medium --scenarios agv5_b2t_t7open_arr1h_s0 \
  --layouts D --agv 84 --rules MDD_PTECT --extra "$C" >> $H/launch.out 2>&1 &
Q --exp dev-agv105-probe --workers 1 --timeout-hours 3 --loglevel Medium --scenarios agv5_b2m_t7iso_arr1h_s0 \
  --layouts D --agv 21 --rules MDD_PTECT --extra "$C" >> $H/launch.out 2>&1 &
wait
date -Is > $H/DONE
