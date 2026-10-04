#!/bin/bash
# base-due-c2 rerun of the 16 seeds that hit the old 120k s player budget (removed 10-03 after dev-clock-check).
cd "$(dirname "$0")/../.."
for s in 0 1 2 3 4 5 7 10 11 12 13 14 15 16 17 18; do
    results/base-due-c2/eval_task.sh results/base-due-c2 $s $((1700 + 4*s)) '{"due_date_allowance": 2.0}' &
done
wait
.venv/bin/python results/base-due-c2/analyze.py > results/base-due-c2/analysis.out 2>&1
echo "[$(date '+%m-%d %H:%M')] all done"
