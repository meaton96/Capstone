#!/bin/bash
# agv-ratio-twin-v5 (10-14): ../agv-ratio-twin on the v5 rules. The free-flow fleet knee in the twin (a lower bound)
# on due-dated instances (TWK, c ~ U[1.75, 2.5]) under MDD-PTECT and its travel-blind twin MDD-PECT, instead of
# SRT_TECT / SRT_ECT without due dates. Same grid as agv-ratio-twin + its 10-08 extension: M = 15 / 50 / 100 (layout D,
# the same floor exports) x util 0.7 / 0.85 / 1.0 x mean op 240 / 480 s x 10 fleets (m100 at 240 s: 13) x seeds 0-2 x
# 2 rules = 1,134 runs, about 10-12 CPU-h. One CSV per (M, load); sweep.py resumes. analyze.py: knees (on mean time in
# system and on tardiness), the fit, knees.csv (read by ../agv-ratio-unity/launch.sh).
#   env: WORKERS (default 8)
set -uo pipefail
cd "$(dirname "$0")/../.."
S=results/agv-ratio-scale
D=results/agv-ratio-twin-v5
P=${WORKERS:-8}
floor() { case $1 in 15) echo linux_server/Results/outdated/G1/_mfsweep_control/D/agv25_FIFO_ECT_s42/des_floor.json;;
                     *) echo $S/floors/m$1/des_floor.json;; esac; }
fleets() { case $1_$2 in 15_*) echo 2 3 4 5 6 7 8 9 10 12;;
                         50_*) echo 5 8 10 12 14 16 18 21 25 30;;
                         100_240) echo 10 15 20 25 30 35 40 45 50 60 70 80 90;;
                         100_*) echo 10 15 20 25 30 35 40 45 50 60;; esac; }
for M in 100 50 15; do for U in 0.7 0.85 1.0; do for O in 240 480; do
    .venv/bin/python $D/sweep.py $D/m${M}_u${U}_op${O}.csv --m $M --floor "$(floor $M)" --agvs $(fleets $M $O) \
        --seeds 0 1 2 --rules MDD_PTECT MDD_PECT --workers $P --extra "--util $U $U --opmean $O $O" >> $D/sweep.out 2>&1
done; done; done
.venv/bin/python $D/analyze.py
