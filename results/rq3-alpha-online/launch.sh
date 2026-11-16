#!/bin/bash
# rq3-alpha-online (10-29, user: online alpha, h from the on-duty fleet per regime, plus the spread law): stage 1,
# local lab run, W tasks at a time (one CPU each): the 49-pair surfaces (run.py list, longest first), then the
# held-out cells' floor statistics. Finished tasks are skipped, so relaunching resumes.
set -uo pipefail
W="${1:-14}"
cd "$(dirname "$0")/../.."
H=results/rq3-alpha-online
mkdir -p $H/logs
.venv/bin/python $H/run.py list > $H/tasks.txt
echo "$(wc -l < $H/tasks.txt) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '
.venv/bin/python results/rq3-alpha-online/run.py "$1" "$2" "$3" > "results/rq3-alpha-online/logs/$1_s$2.log" 2>&1
echo "$* rc=$?"' _ < $H/tasks.txt
echo "all tasks ended ($(date -Is))"
