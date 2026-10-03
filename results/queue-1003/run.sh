#!/bin/bash
# 10-03 afternoon queue (planned with the user 10-03 14:00): tasks.txt in priority order, WORKERS at a time.
#   oracle  rq2-oracle-due seeds 8-19, then the regime screen rq2-oracle-due-agv4 / -l1500 / -utilhi (seeds 0-3)
#   base    base-due-c2 seeds 0-19 (21 head pairs, tardiness)
# Every task skips itself when its output exists, so rerunning resumes. Player: linux_server_due/ (frozen copy).
# Usage: nohup results/queue-1003/run.sh > results/queue-1003/run.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
WORKERS="${WORKERS:-20}"
run_task() {  # line number in tasks.txt
    local line kind out seed wid params extra
    line=$(sed -n "${1}p" results/queue-1003/tasks.txt)
    IFS='|' read -r kind out seed wid params extra <<< "$line"
    local -a ex=(); [[ -n "$extra" ]] && read -ra ex <<< "$extra"
    case "$kind" in
        oracle) results/rq2-oracle-due/oracle_task.sh "$out" "$seed" "$wid" "$params" "${ex[@]}" ;;
        base)   results/base-due-c2/eval_task.sh "$out" "$seed" "$wid" "$params" ;;
    esac
}
export -f run_task
echo "[$(date '+%m-%d %H:%M')] starting $(wc -l < results/queue-1003/tasks.txt) tasks, $WORKERS workers"
seq 1 "$(wc -l < results/queue-1003/tasks.txt)" | xargs -P "$WORKERS" -I{} bash -c 'run_task {}'
echo "[$(date '+%m-%d %H:%M')] all done"
for d in results/rq2-oracle-due results/rq2-oracle-due-agv4 results/rq2-oracle-due-l1500 results/rq2-oracle-due-utilhi; do
    echo "== $d"; .venv/bin/python results/rq2-oracle-due/analyze.py "$d" > "$d/analysis.out" 2>&1; head -40 "$d/analysis.out"
done
.venv/bin/python results/base-due-c2/analyze.py > results/base-due-c2/analysis.out 2>&1; cat results/base-due-c2/analysis.out
