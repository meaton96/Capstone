#!/bin/bash
# eval-alpha-unity (10-30, user: plan §3.2, the Unity check of the static law + gate through the Python rule mode on B2
# and one large floor): local lab run, W Unity tasks at a time. task.py lists the tasks (cell, policy, seeds), the
# 100-machine ones first; each runs one player on its own port. Finished tasks (DONE) are skipped, so relaunching
# resumes. Refuses to start unless linux_server/ holds the player the check was set up on (task.py checks it per task).
# ALL_DONE is written only when every task has its output (the lab's done check).
set -uo pipefail
W="${1:-8}"
cd "$(dirname "$0")/../.."
H=results/eval-alpha-unity
mkdir -p $H/logs
rm -f $H/ALL_DONE
{
    echo "== start $(date -Is) W=$W"; echo "git HEAD $(git rev-parse HEAD)"
    echo "Simulation.dll $(md5sum linux_server/capstone_Data/Managed/Simulation.dll | cut -d' ' -f1)"
    echo "code $(cd $H/code && find env -name '*.py' -not -path '*__pycache__*' | sort | xargs md5sum | md5sum | cut -d' ' -f1)"
} >> $H/PROVENANCE.txt
.venv/bin/python $H/task.py list > $H/tasks.txt
echo "$(wc -l < $H/tasks.txt) tasks, $W at a time ($(date -Is))"
xargs -P "$W" -L 1 bash -c '
.venv/bin/python results/eval-alpha-unity/task.py "$1" "$2" "$3" > "results/eval-alpha-unity/logs/$1_$2_s$3.log" 2>&1
echo "$* rc=$?"' _ < $H/tasks.txt
missing=0
while read -r cell pol s _; do
    [ -f "$H/out/$cell/${pol}_s$s/DONE" ] || { echo "missing: $cell $pol $s"; missing=$((missing + 1)); }
done < $H/tasks.txt
if [ "$missing" -eq 0 ]; then touch $H/ALL_DONE; echo "all tasks done ($(date -Is))"; else echo "$missing tasks missing"; exit 1; fi
