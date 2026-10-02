#!/bin/bash
# Re-split the unfinished FLEX groups into small array tasks so the backfill scheduler can start them (the original
# full-node 36-CPU / 120g shard-1 tasks sat in (Priority) for days). Run from ~/capstone/linux_server on the cluster,
# after cancelling the old arrays so no two jobs work on the same group:
#   scancel 21787418 21787419 21787420 21787421 21787422
#   ACCOUNT=drl-scheduling slurm/resubmit_flex.sh            # DRY=1 prints the sbatch lines only
# Finished cells are skipped. Shard membership is i::N over the stable longest-first grid, and the old 2-shard run
# finished the even indices, so NODES must be odd (N=9 gives every task 6 of the remaining 54 cells; N=6 would give
# half the tasks nothing to do). Groups with all 108 results.csv present are skipped.
# The cell flags reproduce the original run (checked against FLEX_p0.15_m1.25 sim.log): rnd_flex_s0-8, D, 7 AGVs,
# 12 head rules, releasePrevious, lane parking, -flex p -flexmult m.
set -euo pipefail
: "${ACCOUNT:?}" "${NODES:=9}" "${CPUS:=8}" "${MEM:=24g}" "${TIME:=0-04:00:00}" "${PARTITION:=sporc}"
(( NODES % 2 == 1 )) || { echo "NODES must be odd (the old run finished the even grid indices)"; exit 1; }
SCENARIOS="$(printf 'rnd_flex_s%d:1 ' {0..8})"
RULES="SPT_ECT SPT_TECT SPT_SRWT SRT_ECT SRT_TECT SRT_SRWT PTWINQ_ECT PTWINQ_TECT PTWINQ_SRWT FIFO_ECT FIFO_TECT FIFO_SRWT"
for g in 0.15:1.0 0.15:1.25 0.3:1.0 0.3:1.25 0.5:1.0 0.5:1.25; do
  p=${g%:*}; m=${g#*:}; exp="FLEX_p${p}_m${m}"
  n=$(find "Results/$exp" -name results.csv 2>/dev/null | wc -l)
  if (( n >= 108 )); then echo "$exp: 108/108, skipped"; continue; fi
  echo "$exp: $n/108 done, $NODES tasks x $CPUS CPUs"
  EXP="$exp" SCENARIOS="$SCENARIOS" LAYOUTS=D AGVS=7 RULES="$RULES" \
    EXTRA="-reservation releasePrevious -parking lane -flex $p -flexmult $m" \
    ACCOUNT="$ACCOUNT" NODES="$NODES" CPUS="$CPUS" MEM="$MEM" TIME="$TIME" PARTITION="$PARTITION" slurm/submit.sh
done
