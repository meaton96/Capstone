#!/bin/bash
# dev-failure-resume (2026-10-28) analysis: Unity vs twin parity on the exact cells (failparity.py), the regression
# cells against the current player (compare_regress.py), and the resume vs restart effect on the native-sampler cells.
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home/ubuntupc/Documents/Github/Capstone/.venv/bin/python
R=$HERE/runs
cd "$HERE" || exit 1
echo "== Unity vs twin, injected schedule, 1 AGV (resume cells: twin default; *_restart: twin failure_restart)"
$PY failparity.py "$R"/exact/* --csv parity_exact.csv
echo
echo "== -failurerestart / failures off on the new player vs the current player (must be identical)"
for n in x1_s0_uni x1_s2_het x1_s4_wkx; do
  echo "-- exact/${n}_SPT_ECT_restart vs base/${n}_SPT_ECT"; $PY compare_regress.py "$R/exact/${n}_SPT_ECT_restart" "$R/base/${n}_SPT_ECT"
done
for c in fail_s0_SPT_ECT fail_s1_MDD_TECT; do
  echo "-- regress/restart_$c vs base/$c"; $PY compare_regress.py "$R/regress/restart_$c" "$R/base/$c"
done
echo "-- regress/resume_nofail_s0_SPT_ECT vs base/nofail_s0_SPT_ECT"
$PY compare_regress.py "$R/regress/resume_nofail_s0_SPT_ECT" "$R/base/nofail_s0_SPT_ECT"
echo
echo "== resume vs restart (current player) with Unity's native failure samplers, 7 AGVs"
$PY - "$R" <<'PYEOF'
import csv, sys, os
R = sys.argv[1]
def row(d):
    with open(os.path.join(d, "results.csv")) as f:
        return list(csv.DictReader(f))[0]
for c in ("fail_s0_SPT_ECT", "fail_s1_MDD_TECT"):
    a, b = row(f"{R}/base/{c}"), row(f"{R}/regress/resume_{c}")
    out = []
    for k in ("makespan", "mean_flow_time", "mean_tardiness", "jobs_completed", "episode_failures"):
        if k in a and k in b:
            out.append(f"{k} {a[k]} -> {b[k]}")
    print(c + ": " + "; ".join(out))
PYEOF
