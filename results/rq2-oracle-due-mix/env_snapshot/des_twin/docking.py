"""Docking check: does the twin reproduce a Unity run where AGVs cannot interfere (one AGV)?

  python -m des_twin.docking RUN_DIR [RUN_DIR ...] [--csv out.csv]

For each Unity run ("-destrace -baselinedrain") the kinematic twin is run on the same floor and jobs, then:

  episode   makespan, mean / p95 flow time and mean transport wait, twin vs Unity;
  events    AGV milestones (dispatch, pickup, dropoff, ...) compared in order: how many match exactly (time,
            AGV, event, job) before the first difference, and when that is;
  decisions routing / dispatch decisions (decision_log.csv) compared the same way;
  jobs      per-job exit times within Unity's logging precision (float32, written to 0.1 s);
  legs      share of uncontended Unity legs the free-flow model times to the tick (legs.compare_legs).

With one AGV everything should match; a first difference marks where the two models part (a tie broken the
other way, float drift in a timer), after which the runs are different samples of the same system.
"""
import argparse
import csv
import json
import os

import numpy as np

from .engine import TwinConfig, run_twin
from .floor import Floor
from .legs import compare_legs
from .run import unity_row


def _first_diff(a, b, same):
    n = min(len(a), len(b))
    for i in range(n):
        if not same(a[i], b[i]):
            return i
    return n if len(a) == len(b) else n


def dock(run_dir, transport="kinematic"):
    u = unity_row(run_dir)
    floor = Floor.load(os.path.join(run_dir, "des_floor.json"))
    with open(os.path.join(run_dir, "des_jobs.json")) as f:
        jobs = json.load(f)
    tw = run_twin(floor, jobs, TwinConfig(u["rule"], transport, agv_count=int(u["agvCount"])))
    s = tw.summary()

    with open(os.path.join(run_dir, "agv_events.csv")) as f:
        ue = [(float(r["sim_time"]), int(r["agv_id"]), r["event"].split("_")[0] if "dispatch" in r["event"]
               else r["event"], int(r["job_id"])) for r in csv.DictReader(f) if r["event"] not in ("return", "park")]
    # Same-tick milestones (an AGV dispatched while already at the pickup dock) are logged in either order.
    ue = sorted(ue, key=lambda x: (round(x[0], 2), x[1:]))
    te = sorted(((round(t, 2), a, e, j) for t, a, e, j in tw.trace if e != "park(planned)"),
                key=lambda x: (x[0], x[1:]))
    same_ev = lambda x, y: x[1:] == y[1:] and abs(x[0] - y[0]) < 0.011
    i_ev = _first_diff(ue, te, same_ev)

    ud = []
    if os.path.exists(os.path.join(run_dir, "decision_log.csv")):
        with open(os.path.join(run_dir, "decision_log.csv")) as f:
            ud = [(float(r["sim_time"]), r["decision_type"].lower(), int(r["subject_id"]), int(r["chosen_id"]))
                  for r in csv.DictReader(f)]
    td = [(t, k, a, b) for t, k, a, b, _, _ in tw.decision_rows]
    # decision_log.csv times are rounded to 0.1 s and drift from SimTime by up to ~0.1 s; AGV events check timing
    i_dec = _first_diff(ud, td, lambda x, y: x[1:] == y[1:] and abs(x[0] - y[0]) < 0.15)

    with open(os.path.join(run_dir, "job_completions.csv")) as f:
        exits = {int(r["job_id"]): float(r["exit_time"]) for r in csv.DictReader(f) if r["completed"] == "1"}
    diffs, exit_ok = [], []
    for r in tw.job_rows():
        if r["exit_time"] is None or r["job_id"] not in exits:
            continue
        d = abs(r["exit_time"] - exits[r["job_id"]])
        diffs.append(d)
        # Unity stores ExitTime as float32 and writes it to 0.1 s: allow the rounding plus one float32 spacing
        exit_ok.append(d <= 0.05 + float(np.spacing(np.float32(r["exit_time"]))) + 1e-9)

    legs = compare_legs(run_dir, transport)
    free = [r for r in legs if r["zone_wait_s"] == 0]

    def rel(a, b):
        return (a - b) / b if b else 0.0

    return {
        "run": os.path.relpath(run_dir), "instance": u["instance"], "rule": u["rule"], "agvs": int(u["agvCount"]),
        "jobs": s["jobs"], "unity_makespan": float(u["makespan"]), "twin_makespan": round(s["makespan"], 2),
        "makespan_rel": round(rel(s["makespan"], float(u["makespan"])), 6),
        "unity_mean_flow": float(u["mean_flow_time"]), "twin_mean_flow": round(s["mean_flow_time"], 2),
        "mean_flow_rel": round(rel(s["mean_flow_time"], float(u["mean_flow_time"])), 6),
        "p95_flow_rel": round(rel(s["p95_flow_time"], float(u["p95_flow_time"])), 6),
        "transport_wait_rel": round(rel(s["mean_transport_wait"], float(u["mean_transport_wait"])), 6),
        "events": len(ue), "events_matched": i_ev, "events_first_diff_t": ue[i_ev][0] if i_ev < len(ue) else None,
        "decisions": len(ud), "decisions_matched": i_dec,
        "decisions_first_diff_t": ud[i_dec][0] if i_dec < len(ud) else None,
        "jobs_exit_max_abs_s": round(max(diffs), 3) if diffs else None,
        "jobs_exit_within_rounding": round(sum(ok for ok in exit_ok) / len(exit_ok), 4) if exit_ok else None,
        "legs_uncontended": len(free),
        "legs_exact": round(sum(abs(r["delay_s"]) < 0.011 for r in free) / len(free), 4) if free else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--csv")
    a = ap.parse_args(argv)
    rows = []
    for d in a.run_dirs:
        r = dock(d)
        rows.append(r)
        print(f"{r['instance']:14s} {r['rule']:12s} makespan {r['makespan_rel']:+.4%}  flow {r['mean_flow_rel']:+.4%}  "
              f"events {r['events_matched']}/{r['events']}  decisions {r['decisions_matched']}/{r['decisions']}  "
              f"exits {r['jobs_exit_within_rounding']}  legs exact {r['legs_exact']}")
    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
