#!/bin/bash
# rq3-size-grid (10-20): run one part locally (lab run, W parallel tasks, one CPU each), largest floors first.
#   ref   pdr + cmix on m20 m30 m75 m150 m200 (56 tasks; the local machine was idle while the cluster queue held the
#         train-size-* jobs). arms needs the train-size-* checkpoints locally; the default for arms is submit.sh.
# Same task lines and outputs as submit.sh; task.py skips finished chunks, so relaunching resumes.
set -uo pipefail
PART="${1:?usage: launch_local.sh ref|arms W}"; W="${2:-16}"
cd "$(dirname "$0")/../.."
D=results/rq3-size-grid
if [[ $PART == ref ]]; then GRPS="pdr cmix"; SMALL="m20 m30 m75"; RUNS="train-v5-cmix"
else GRPS="size50 size100 sizemix"; SMALL="m15 m20 m30 m50 m75 m100"; RUNS="train-size-m50 train-size-m100 train-size-mix"; fi
for r in $RUNS; do for k in 0 1 2 3 4; do
    [[ -f results/${r}_s$k/checkpoint.pt ]] || { echo "missing results/${r}_s$k/checkpoint.pt"; exit 1; }
done; done
list() {
    for c in $1; do for g in $GRPS; do for k in $2; do
        [[ -f $D/$c/${g}_s$k/DONE ]] || echo "--cond $c --group $g --chunk $k"
    done; done; done
}
T=$D/tasks_${PART}_local.txt
{ list "m200 m150" "0-4 5-9 10-14 15-19 20-24 25-29 30-34 35-39"
  list "$(tr ' ' '\n' <<< "$SMALL" | tac | tr '\n' ' ')" "0-9 10-19 20-29 30-39"; } > $T
echo "$(wc -l < $T) tasks, $W at a time ($(date -Is))"
mkdir -p $D/logs
xargs -P "$W" -L 1 bash -c 'echo "start $*"; .venv/bin/python results/rq3-size-grid/task.py "$@" \
    > "results/rq3-size-grid/logs/$2_$4_$6.log" 2>&1; echo "end $* rc=$?"' _ < $T
echo "all tasks ended ($(date -Is))"
