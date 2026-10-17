#!/bin/bash
# agv-ratio-unity (v5 design, 10-14): Unity check of the twin's free-flow fleet knee on the v5 rules
# (../agv-ratio-twin-v5) at 50 / 100 machines: the twin's knee5 under MDD-PTECT and the twin fleet nearest 1.5x it,
# util 0.85, mean op 240 / 480 s, MDD_PTECT, instance seeds 0 / 1 = 16 runs on the twin sweep's own instances
# (make_scenarios.py writes the scenarios and cells.txt from knees.csv). Player linux_server/ (5e6e0f87),
# releasePrevious, lane parking, layout D, -destrace so every run is also paired with DES-1k on its own export
# (des_twin_gap.py). One run per worker, longest first; resumes.   env: WORKERS (default 8)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=results/agv-ratio-unity
.venv/bin/python $H/make_scenarios.py > $H/make_scenarios.out || { cat $H/make_scenarios.out; exit 1; }
echo "== start $(date -Is) git $(git rev-parse --short HEAD) Simulation.dll $(md5sum linux_server/capstone_Data/Managed/Simulation.dll | cut -d' ' -f1)" >> $H/PROVENANCE.txt
cell() {  # M U OP SEED AGVS
    (cd linux_server && python3 run_experiment_queue.py --exp agv-ratio-unity --workers 1 --timeout-hours 30 \
        --scenarios "agvratio5_m$1_u$2_op$3_s$4" --layouts D --agv "$5" --rules MDD_PTECT \
        --extra "-baselinedrain -destrace -parking lane -allowunsafefleet -reservation releasePrevious")
}
export -f cell
xargs -P "${WORKERS:-8}" -L1 bash -c 'cell $0 $1 $2 $3 $4' < $H/cells.txt >> $H/launch.out 2>&1
echo "== end $(date -Is)" >> $H/PROVENANCE.txt
.venv/bin/python results/scripts/des_twin_gap.py linux_server/Results/agv-ratio-unity --out $H --tag agv-ratio-unity \
    --twin-out twin_agv-ratio-unity --workers 8 > $H/twin.out 2>&1
echo "unity runs: $(ls linux_server/Results/agv-ratio-unity/*/D/agv*/results.csv 2>/dev/null | wc -l)/16"
date -Is > $H/DONE
