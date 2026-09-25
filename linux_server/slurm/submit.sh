#!/bin/bash
# Submit one experiment as a Slurm job array (one node per shard). Run from ~/capstone/linux_server on the cluster:
#   ACCOUNT=<slurm account> EXP=E4_pilot SCENARIOS="compound_scenario:1" LAYOUTS="D" AGVS="7" RULES="SPT_SRWT" \
#   EXTRA="-reservation releasePrevious -parking lane" NODES=1 TIME=0-02:00:00 PARTITION=debug slurm/submit.sh
# NODES > 1 splits the grid across that many nodes. Re-running the same command resumes: finished cells are skipped.
# Use PARTITION=debug only for short test runs (1-day max); real sweeps go on sporc. Add DRY=1 to print the sbatch line.
# CPUS/MEM default to the sbatch file (36 cores, 120g); small tests start much sooner with e.g. CPUS=4 MEM=16g.
set -euo pipefail
: "${ACCOUNT:?}" "${EXP:?}" "${SCENARIOS:?}" "${LAYOUTS:?}" "${AGVS:?}" "${RULES:?}"
: "${PARTITION:=debug}" "${TIME:=0-01:00:00}" "${NODES:=1}" "${EXTRA:=-reservation holdPrevious -parking lane}"
export EXP SCENARIOS LAYOUTS AGVS RULES EXTRA             # sbatch passes the submit environment through (--export=ALL default)
mkdir -p logs                                             # Slurm will not create the --output folder
CMD=(sbatch --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME" --job-name="$EXP"
     --array="0-$((NODES - 1))"
     ${CPUS:+--cpus-per-task="$CPUS"} ${MEM:+--mem="$MEM"}
     slurm/run_queue.sbatch)
if [ "${DRY:-0}" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; else "${CMD[@]}"; fi
