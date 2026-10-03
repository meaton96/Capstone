"""dev-due-reg part A: every results.csv / job_completions.csv field of the due-date build (linux_server_dev/) against
the current build (linux_server/), cell by cell, on scenarios without due dates. Expected: identical except the
timestamp and the columns the due-date build adds."""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OLD, NEW = REPO / "linux_server/Results/dev-due-reg", REPO / "linux_server_dev/Results/dev-due-reg"
NEW_RESULTS = {"jobs_with_due_date", "mean_tardiness", "pct_tardy", "max_tardiness"}
NEW_JOBS = {"due_date", "tardiness"}
cells = sorted(p.parent.relative_to(NEW) for p in NEW.glob("*/*/*/results.csv"))
bad = 0
for cell in cells:
    for name, extra in (("results.csv", NEW_RESULTS), ("job_completions.csv", NEW_JOBS)):
        o, n = pd.read_csv(OLD / cell / name), pd.read_csv(NEW / cell / name)
        added = set(n.columns) - set(o.columns)
        shared = [c for c in o.columns if c != "timestamp"]
        same = o[shared].equals(n[shared])
        if not same or added != extra:
            bad += 1
            diff = [c for c in shared if not o[c].equals(n[c])]
            print(f"DIFF {cell} {name}: columns {diff[:8]} added {sorted(added)}")
    r = pd.read_csv(NEW / cell / "results.csv").iloc[0]
    print(f"ok   {cell}: makespan {r.makespan:.1f} mean flow {r.mean_flow_time:.1f} due-date jobs {r.jobs_with_due_date}"
          if not bad else f"     {cell}")
print(f"{len(cells)} cells compared, {bad} file differences")
sys.exit(1 if bad else 0)
