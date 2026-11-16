#!/bin/bash
# eval-alpha-online (10-29, user: online alpha, h from the on-duty fleet per regime, plus the spread law): stage 2,
# local lab run, W tasks at a time (one CPU each). Runs only if stage 1's gate G1 passed
# (../rq3-alpha-online/law_online.json). The policy tasks (task.py list: the cells where the online law changes the
# pair, longest first) come first, then the simulator's law runs (static / online / map law alone,
# ../rq3-alpha-online/run.py list law). Finished tasks are skipped, so relaunching resumes. ALL_DONE is written only
# when every task has its output (the lab's done check; the task count depends on the selected law).
set -uo pipefail
W="${1:-20}"
cd "$(dirname "$0")/../.."
H=results/eval-alpha-online
T=results/rq3-alpha-online
if ! .venv/bin/python -c "import json, sys; sys.exit(json.load(open('$T/law_online.json'))['G1'] != 'PASS')"; then
    echo "$T/law_online.json: gate G1 did not pass (or no law_online.json); stage 2 does not run"
    exit 3
fi
mkdir -p $H/logs
rm -f $H/ALL_DONE
{ .venv/bin/python $H/task.py list | sed 's/^/eval /'; .venv/bin/python $T/run.py list law | sed 's/^/law /'; } > $H/tasks.txt
echo "$(grep -c '^eval' $H/tasks.txt) policy tasks + $(grep -c '^law' $H/tasks.txt) law runs, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '
if [ "$1" = eval ]; then
    .venv/bin/python results/eval-alpha-online/task.py "$2" "$3" > "results/eval-alpha-online/logs/$2_s$3.log" 2>&1
else
    .venv/bin/python results/rq3-alpha-online/run.py "$2" "$3" law > "results/eval-alpha-online/logs/law_$2_s$3.log" 2>&1
fi
echo "$* rc=$?"' _ < $H/tasks.txt
missing=0
while read -r kind cell s _; do    # law lines carry a 4th field (run.py list law: cell seed law)
    if [ "$kind" = eval ]; then f="$H/out/$cell/og_s$s/DONE"; else f="$T/out/$cell/law/s$s.json"; fi
    [ -f "$f" ] || { echo "missing: $kind $cell $s"; missing=$((missing + 1)); }
done < $H/tasks.txt
if [ "$missing" -eq 0 ]; then touch $H/ALL_DONE; echo "all tasks done ($(date -Is))"; else echo "$missing tasks missing"; exit 1; fi
