#!/usr/bin/env bash
# rq2-replan-axes, 7b: the horizon arm (README §19, approved by the user 11-10; lab entry rq2-replan-axes-horizon, local).
# 1. The 720 base tasks (run7.py list horizon), costliest first, on <workers> processes.
# 2. Once all exist: analyze7.py extension names the floors where H 3,600 s fails H1, and their H 7,200 s tasks run
#    (80 per floor; the pre-declared extension rule). Nothing more if none fails.
# 3. Writes <root>/out7/DONE (the lab's done check) when every task the rule calls for exists.
# Every step skips finished outputs, so a rerun resumes; exits 1 while any task is missing. Writes only under
# <root>/out7/ (run.py's out/ is read, never written). ROOT7 (default: this folder) is for tests in a scratch folder.
# Usage: bash results/rq2-replan-axes/launch7.sh [workers]
set -euo pipefail
cd "$(dirname "$0")/../.."
W="${1:-20}"
PY=.venv/bin/python
R7=results/rq2-replan-axes/run7.py
A7=results/rq2-replan-axes/analyze7.py
ROOT="${ROOT7:-results/rq2-replan-axes}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one core per task (serial forks)
mkdir -p "$ROOT/out7"
rm -f "$ROOT/out7/DONE"
"$PY" "$R7" list horizon --root "$ROOT" > "$ROOT/tasks_horizon.txt"
"$PY" "$R7" list horizon --todo --root "$ROOT" | xargs -r -P "$W" -L 1 "$PY" "$R7" || true
left=$("$PY" "$R7" list horizon --todo --root "$ROOT" | wc -l)
if [ "$left" -ne 0 ]; then
    echo "horizon: $left of 720 base tasks unfinished" >&2
    exit 1
fi
ext=$("$PY" "$A7" extension --root "$ROOT")
echo "horizon: 720 base tasks done; H 7,200 s extension at: ${ext:-none}"
if [ -n "$ext" ]; then
    # shellcheck disable=SC2086  # ext is a space-separated list of floor names
    "$PY" "$R7" list horizon-ext $ext --root "$ROOT" > "$ROOT/tasks_horizon_ext.txt"
    "$PY" "$R7" list horizon-ext $ext --todo --root "$ROOT" | xargs -r -P "$W" -L 1 "$PY" "$R7" || true
    left=$("$PY" "$R7" list horizon-ext $ext --todo --root "$ROOT" | wc -l)
    if [ "$left" -ne 0 ]; then
        echo "horizon: $left extension tasks unfinished" >&2
        exit 1
    fi
fi
n=$(find "$ROOT/out7" -name 's*.json' | wc -l)
printf 'done %s\ntask outputs %s\nextension %s\n' "$(date '+%Y-%m-%d %H:%M')" "$n" "${ext:-none}" > "$ROOT/out7/DONE"
echo "horizon: all $n task outputs written; extension: ${ext:-none}"
