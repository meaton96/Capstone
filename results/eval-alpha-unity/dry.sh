#!/bin/bash
# eval-alpha-unity dry run on the 100-machine cell (lab dev-alpha-unity-dry, 10-30), seed 0, into dry/:
#   pdr (C# MDD-PTECT), law (MDD-PTECT@x1 in Python = the law alone), ck0 (gated policy), ck0 with the offset forced to
#   x1 (must equal law exactly: the gate and the alpha mapping in Unity), all at once. Times one m100 episode per policy.
#   Finished tasks are skipped. dry/DONE when all four have their output; dry_check.out compares ck0-forced with law.
set -uo pipefail
cd "$(dirname "$0")/../.."
H=results/eval-alpha-unity
D=$H/dry
mkdir -p $D
run() { .venv/bin/python $H/task.py m100_r052 "$@" --out-root $D > "$D/log_$(echo "$*" | tr ' ' '_').log" 2>&1; echo "$* rc=$?"; }
run pdr 0-0 &
sleep 5; run law 0-0 &
sleep 5; run ck0 0-0 &
sleep 5; mkdir -p $D/forced && .venv/bin/python $H/task.py m100_r052 ck0 0-0 --force-offset 0 --out-root $D/forced \
    > $D/log_forced.log 2>&1; echo "forced rc=$?" &
wait
n=$(ls $D/m100_r052/*/DONE $D/forced/m100_r052/*/DONE 2>/dev/null | wc -l)
.venv/bin/python - > $D/dry_check.out 2>&1 <<'EOF'
import csv, json
from pathlib import Path
D = Path("results/eval-alpha-unity/dry")
rows = {}
for name, d in (("pdr", D / "m100_r052/pdr_s0-0"), ("law", D / "m100_r052/law_s0-0"), ("ck0", D / "m100_r052/ck0_s0-0"),
                ("forced", D / "forced/m100_r052/ck0_s0-0")):
    if (d / "DONE").exists():
        r = list(csv.DictReader(open(d / "episodes.csv")))[0]
        rows[name] = r
        print(f"{name:7s} tardiness {float(r['window_tardiness']) / 1e3:10.1f} k  jobs {r['jobs_exited']}  decisions "
              f"{r['decisions']}  makespan {r['makespan']}  instance {r['instance_hash']}  wall {(d / 'DONE').read_text().strip()} s  "
              f"picks {sum(json.loads((d / 'cond.json').read_text()).get('python_picks', {}).values())}")
if "forced" in rows and "law" in rows:
    f = ("window_tardiness", "jobs_exited", "decisions", "makespan", "instance_hash")
    same = all(rows["forced"][k] == rows["law"][k] for k in f)
    print(f"ck0 with offset forced to x1 == the law alone: {same}")
EOF
cat $D/dry_check.out
[ "$n" -eq 4 ] && date -Is > $D/DONE
