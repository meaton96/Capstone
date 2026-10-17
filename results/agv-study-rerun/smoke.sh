#!/bin/bash
# agv-study-rerun (v5 design) smoke, 10-14: three Unity batch cells on the due-dated B2-family scenarios
# (agv5_scenarios.py), then the twin on each run's -destrace export. Checks: due dates load, results.csv carries
# tardiness, MDD_PTECT / MDD_PECT run in batch mode, hold and machine failures apply, and the twin pairs.
set -uo pipefail
cd "$(dirname "$0")/../.."
Q() { (cd linux_server && python3 run_experiment_queue.py --exp dev-agv5-smoke --workers 1 --timeout-hours 2 "$@"); }
C="-baselinedrain -destrace -parking lane -allowunsafefleet"
Q --scenarios agv5_b2m_s0 --layouts D --agv 7 --rules MDD_PTECT --extra "$C -reservation releasePrevious" &
Q --scenarios agv5_b2t_s0 --layouts D --agv 10 --rules MDD_PECT --extra "$C -reservation holdPrevious" &
Q --scenarios agv5_b2tf_s0 --layouts J --agv 7 --rules MDD_PTECT --extra "$C -reservation releasePrevious" &
wait
for r in linux_server/Results/dev-agv5-smoke/*/*/agv*; do
    [ -f "$r/results.csv" ] || { echo "$r: no results.csv"; continue; }
    [ -f "$r/twin_smoke/twin_results.csv" ] || (cd env && ../.venv/bin/python -m des_twin.run "../$r" --out twin_smoke > "../$r/twin.out" 2>&1)
done
.venv/bin/python - <<'PY'
import glob, pandas as pd
for f in sorted(glob.glob("linux_server/Results/dev-agv5-smoke/*/*/agv*/results.csv")):
    r = pd.read_csv(f, dtype=str).iloc[-1]
    print(f.split("dev-agv5-smoke/")[1].rsplit("/", 1)[0])
    print("  unity:", {k: r.get(k) for k in ("rule", "agvCount", "reservation_protocol", "layout_id", "episode_failures",
          "mean_flow_time", "jobs_censored", "deadlock_detected", "jobs_with_due_date", "mean_tardiness", "pct_tardy",
          "agv_collision_events")})
    t = glob.glob(f.replace("results.csv", "twin_smoke/twin_results.csv"))
    if t:
        print("  twin: ", pd.read_csv(t[0]).iloc[-1].to_dict())
PY
date -Is > linux_server/Results/dev-agv5-smoke/DONE
