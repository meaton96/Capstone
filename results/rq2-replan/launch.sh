#!/bin/bash
# rq2-replan (10-21, user): local lab run, W tasks at a time (one CPU each), costliest variants first (cost ~ number of
# re-plans: ~4.7 s each on a loaded machine). Finished (variant, seed) outputs are skipped, so relaunching resumes.
set -uo pipefail
W="${1:-16}"
cd "$(dirname "$0")/../.."
H=results/rq2-replan
mkdir -p $H/logs
.venv/bin/python - > $H/tasks.txt <<'PY'
import importlib.util
s = importlib.util.spec_from_file_location("r", "results/rq2-replan/run.py"); r = importlib.util.module_from_spec(s); s.loader.exec_module(r)
cost = lambda v: (21600.0 / r.parse(v)[1]) * (0.6 if r.parse(v)[2] else 1.0)
for v in sorted(r.variants(), key=cost, reverse=True):
    for seed in r.SEEDS:
        print(v, seed)
PY
echo "$(wc -l < $H/tasks.txt) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '.venv/bin/python results/rq2-replan/run.py "$@" > "results/rq2-replan/logs/$1_s$2.log" 2>&1; echo "$1 s$2 rc=$?"' _ < $H/tasks.txt
echo "all tasks ended ($(date -Is))"
