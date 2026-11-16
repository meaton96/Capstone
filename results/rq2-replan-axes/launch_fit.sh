#!/usr/bin/env bash
# rq2-replan-axes, §8's fit on axes 1, 2 and 6 (README §17; lab entry rq2-replan-axes-fit, local, analysis only).
# Builds the per-seed table, then runs the point fit and the bootstrap chunks on <workers> processes. Every step skips
# the outputs already in fit_out/, so a rerun resumes. Exits non-zero while any task is unfinished.
# Usage: bash results/rq2-replan-axes/launch_fit.sh [workers]
set -euo pipefail
cd "$(dirname "$0")/../.."
W="${1:-4}"
PY=.venv/bin/python
FIT=results/rq2-replan-axes/fit.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one core per worker
"$PY" "$FIT" table
"$PY" "$FIT" tasks | xargs -r -P "$W" -n 1 "$PY" "$FIT" run
left=$("$PY" "$FIT" tasks | wc -l)
if [ "$left" -ne 0 ]; then
    echo "fit: $left tasks unfinished" >&2
    exit 1
fi
echo "fit: all $(ls "${FIT_OUT:-results/rq2-replan-axes/fit_out}"/*.json | wc -l) task outputs written"
