#!/bin/bash
# eval-alpha-transport-size (10-28, user: size / transport term in the alpha law, pair heads restrained above ~50
# machines): stage 2, local lab run, W tasks at a time (one CPU each). Runs only if stage 1's gate F1 passed
# (../rq3-alpha-transport/law.json). The policy tasks (task.py list, longest first) come first, then the simulator's
# law runs (MDD-PTECT at the law's alpha, ../rq3-alpha-transport/run.py list law). Finished tasks are skipped, so
# relaunching resumes.
set -uo pipefail
W="${1:-20}"
cd "$(dirname "$0")/../.."
H=results/eval-alpha-transport-size
T=results/rq3-alpha-transport
if ! .venv/bin/python -c "import json, sys; sys.exit(json.load(open('$T/law.json'))['F1'] != 'PASS')"; then
    echo "$T/law.json: gate F1 did not pass (or no law.json); stage 2 does not run"
    exit 3
fi
mkdir -p $H/logs
{ .venv/bin/python $H/task.py list | sed 's/^/eval /'; .venv/bin/python $T/run.py list law | sed 's/^/law /'; } > $H/tasks.txt
echo "$(grep -c '^eval' $H/tasks.txt) policy tasks + $(grep -c '^law' $H/tasks.txt) law runs, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '
if [ "$1" = eval ]; then
    .venv/bin/python results/eval-alpha-transport-size/task.py "$2" "$3" "$4" > "results/eval-alpha-transport-size/logs/$2_$3_s$4.log" 2>&1
else
    .venv/bin/python results/rq3-alpha-transport/run.py "$2" "$3" law > "results/eval-alpha-transport-size/logs/law_$2_s$3.log" 2>&1
fi
echo "$* rc=$?"' _ < $H/tasks.txt
echo "all tasks ended ($(date -Is))"
