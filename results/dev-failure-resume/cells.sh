#!/bin/bash
# dev-failure-resume (2026-10-28): the new player (preempt-resume, CapstoneBuild-resume/linux_server_dev) on
# dev-csharp-failures' scenarios. Output runs/<set>/<cell>/; finished cells are skipped (resumable). WORKERS players.
#   exact    1 AGV, injected failureSchedule, 8 scenarios x {SPT_ECT, MDD_TECT, PTWINQ_SRWT}, default (resume): the twin
#            must match event for event (failparity.py)
#   exact    3 cells with -failurerestart: must equal dev-csharp-failures' v5 runs of the same cell (old model) and the
#            twin with failure_restart
#   regress  7 AGVs, native failure samplers: fail_s0 / fail_s1 with -failurerestart and nofail_s0 default must equal
#            the current player's run of the same cell; fail_s0 / fail_s1 default show the change on a native-sampler run
#   base     the current player (linux_server/, Simulation.dll 5e6e0f87, the old model) on the restart / nofail cells,
#            the reference for the two "must equal" checks above
HERE=$(cd "$(dirname "$0")" && pwd)
P=/home/ubuntupc/Documents/Github/CapstoneBuild-resume/linux_server_dev
B=/home/ubuntupc/Documents/Github/Capstone/linux_server
OLD=/home/ubuntupc/Documents/Github/Capstone/results/dev-csharp-failures/scenarios
WORKERS="${1:-${WORKERS:-8}}"
R=$HERE/runs
{
  for f in "$OLD"/exact/*.json; do
    n=$(basename "$f" .json)
    for r in SPT_ECT MDD_TECT PTWINQ_SRWT; do echo "$P $f $r 1 $R/exact/${n}_$r"; done
  done
  for n in x1_s0_uni x1_s2_het x1_s4_wkx; do
    echo "$P $OLD/exact/$n.json SPT_ECT 1 $R/exact/${n}_SPT_ECT_restart -failurerestart"
    echo "$B $OLD/exact/$n.json SPT_ECT 1 $R/base/${n}_SPT_ECT"
  done
  echo "$B $OLD/regress/fail_s0.json SPT_ECT 7 $R/base/fail_s0_SPT_ECT"
  echo "$B $OLD/regress/fail_s1.json MDD_TECT 7 $R/base/fail_s1_MDD_TECT"
  echo "$B $OLD/regress/nofail_s0.json SPT_ECT 7 $R/base/nofail_s0_SPT_ECT"
  echo "$P $OLD/regress/fail_s0.json SPT_ECT 7 $R/regress/restart_fail_s0_SPT_ECT -failurerestart"
  echo "$P $OLD/regress/fail_s1.json MDD_TECT 7 $R/regress/restart_fail_s1_MDD_TECT -failurerestart"
  echo "$P $OLD/regress/nofail_s0.json SPT_ECT 7 $R/regress/resume_nofail_s0_SPT_ECT"
  echo "$P $OLD/regress/fail_s0.json SPT_ECT 7 $R/regress/resume_fail_s0_SPT_ECT"
  echo "$P $OLD/regress/fail_s1.json MDD_TECT 7 $R/regress/resume_fail_s1_MDD_TECT"
} | xargs -P "$WORKERS" -L1 bash "$HERE/unity_run.sh"
echo "[$(date '+%H:%M:%S')] done: $(ls "$R"/*/*/results.csv 2>/dev/null | wc -l) / 38 cells with results.csv"
