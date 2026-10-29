"""
@file failparity.py
@brief dev-failure-resume (2026-10-28, copied from dev-csharp-failures 10-09): Unity (the preempt-resume player,
       injected failureSchedule) against the kinematic twin on the same schedule (FailureTrace.from_schedule), run by
       run. Scenarios are dev-csharp-failures'.

For each run directory (unity_run.sh output: results.csv, des_floor.json, des_jobs.json, agv_events.csv,
decision_log.csv, machine_failures.csv, machine_utilization.csv) the scenario is found by its instance name under
scenarios/*/, its "stochastic.failureSchedule" is read back into a FailureTrace, and des_twin.docking.dock(...,
failures=trace) replays the run. Reported per run:
  exact columns   AGV events, decisions and machine_failures.csv rows matched before the first difference (all of them
                  when the runs are identical), makespan / mean flow / p95 flow / mean tardiness relative gap
  failure process failures and per-machine failure counts and repair time (machine_utilization.csv vs the twin's
                  machine_rows), equal when the failure process is exogenous and the runs reach the same end
  handlers        Unity's machine_failures.csv column sums vs the twin's counters (interrupted = rows with active_job
                  >= 0 whose job was processing; Unity logs the active job id even when idle -> -1)
A run directory whose name contains "legacy" is replayed with legacy_failure_quirks (player flag
-legacyfailurequirks), one whose name contains "restart" with failure_restart (player flag -failurerestart); otherwise
the twin runs its default, preempt-resume.
@par Usage
@code{.sh}
/home/ubuntupc/Documents/Github/Capstone/.venv/bin/python failparity.py runs/exact/* --csv parity_exact.csv
@endcode
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "env"))

import des_twin  # noqa: E402
from des_twin.docking import dock  # noqa: E402
from des_twin.engine import TwinConfig, run_twin  # noqa: E402
from des_twin.failures import FailureTrace  # noqa: E402
from des_twin.floor import Floor  # noqa: E402
from des_twin.run import unity_row  # noqa: E402

assert str(HERE.parents[1]) in des_twin.__file__, des_twin.__file__

HANDLERS = (("queued_rerouted", "queued_rerouted"), ("pickups_cancelled", "pickups_cancelled"),
            ("redirected", "redirects"), ("held", "held_to_down"), ("aborted", None),
            ("predispatch_cleared", "predispatch_cleared"))


def scenario_for(instance):
    hits = list((HERE.parent / "dev-csharp-failures").glob(f"scenarios/*/{instance}.json"))
    if not hits:   # run from the worktree: the scenarios are in the main checkout
        hits = list(Path("/home/ubuntupc/Documents/Github/Capstone/results/dev-csharp-failures").glob(
            f"scenarios/*/{instance}.json"))
    if len(hits) != 1:
        raise FileNotFoundError(f"scenario {instance}: {hits}")
    return json.loads(hits[0].read_text())


def check(run_dir):
    run_dir = str(run_dir)
    u = unity_row(run_dir)
    sc = scenario_for(u["instance"])
    trace = FailureTrace.from_schedule(sc["stochastic"]["failureSchedule"])
    legacy = "legacy" in os.path.basename(os.path.normpath(run_dir))
    restart = "restart" in os.path.basename(os.path.normpath(run_dir))
    r = dock(run_dir, failures=trace, legacy_failure_quirks=legacy, failure_restart=restart)
    # the twin again for the per-machine rows and counters (dock() returns summaries only)
    floor = Floor.load(os.path.join(run_dir, "des_floor.json"))
    with open(os.path.join(run_dir, "des_jobs.json")) as f:
        jobs = json.load(f)
    tw = run_twin(floor, jobs, TwinConfig(u["rule"], "kinematic", agv_count=int(u["agvCount"]), failures=trace,
                                          legacy_failure_quirks=legacy, failure_restart=restart))
    with open(os.path.join(run_dir, "machine_utilization.csv")) as f:
        um = {int(x["machine_id"]): x for x in csv.DictReader(f)}
    tm = {x["machine_id"]: x for x in tw.machine_rows()}
    same_count = all(int(um[m]["failure_count"]) == tm[m]["failure_count"] for m in um)
    rep_gap = max(abs(float(um[m]["total_repair_time"]) - tm[m]["total_repair_time"]) for m in um)
    with open(os.path.join(run_dir, "machine_failures.csv")) as f:
        uf = list(csv.DictReader(f))
    fails = [x for x in uf if x["event"] == "fail"]
    # the failure process itself (exogenous): every failure / repair event (tick, machine, event, repair) up to the
    # earlier of the two episode ends must be the same set in both simulators, whatever the runs did in between
    t_end = min(float(u["makespan"]), tw.makespan) - 0.011
    ev_u = sorted((int(x["tick"]), int(x["machine_id"]), x["event"], float(np.float32(float(x["repair_duration"]))))
                  for x in uf if float(x["sim_time"]) <= t_end)
    ev_t = sorted((x["tick"], x["machine_id"], x["event"], float(np.float32(x["repair_duration"]))) for x in tw.fail_rows
                  if x["sim_time"] <= t_end)
    row = {k: v for k, v in r.items()}
    row["failure_events_to_common_end_equal"] = ev_u == ev_t
    row["failure_events_to_common_end"] = len(ev_u)
    row.update({"legacy": legacy, "restart": restart, "per_machine_counts_equal": same_count, "per_machine_repair_max_abs_s": round(rep_gap, 3),
                "unity_active_job_at_fail": sum(int(x["active_job"]) >= 0 for x in fails),
                "twin_jobs_interrupted": tw.fstats["jobs_interrupted"]})
    for ucol, tkey in HANDLERS:
        row[f"unity_{ucol}"] = sum(int(x[ucol]) for x in fails)
        row[f"twin_{ucol}"] = (tw.fstats[tkey] if tkey else tw.fstats["aborts"] + tw.fstats["redirect_no_route"])
    row["unity_deferred_max"] = max([int(x["deferred_jobs"]) for x in uf if x["event"] == "repair_done"] or [0])
    row["twin_deferred_job_seconds"] = round(tw.summary()["deferred_job_seconds"], 2)
    row["twin_delivered_to_down"] = tw.fstats["delivered_to_down"]
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--csv")
    a = ap.parse_args(argv)
    rows = []
    for d in sorted(a.run_dirs):
        if not os.path.exists(os.path.join(d, "results.csv")):
            print(f"{d}: no results.csv, skipped")
            continue
        r = check(d)
        rows.append(r)
        # results.csv writes means to 0.01 s: equal within that rounding, and every job's exit within Unity's
        # float32 / 0.1 s logging precision (docking's jobs_exit_within_rounding)
        exact = (r["events_matched"] == r["events"] and r["decisions_matched"] == r["decisions"]
                 and r["fail_rows_matched"] == r["fail_rows"]
                 and abs(r["twin_makespan"] - r["unity_makespan"]) <= 0.011
                 and abs(r["twin_mean_flow"] - r["unity_mean_flow"]) <= 0.011
                 and abs(r["twin_mean_tardiness"] - r["unity_mean_tardiness"]) <= 0.011
                 and r["jobs_exit_within_rounding"] == 1.0)
        r["exact"] = exact
        print(f"{os.path.basename(os.path.normpath(d)):34s} {'EXACT' if exact else 'DIFF '} "
              f"events {r['events_matched']}/{r['events']}  decisions {r['decisions_matched']}/{r['decisions']}  "
              f"fail rows {r['fail_rows_matched']}/{r['fail_rows']}  makespan {r['makespan_rel']:+.4%}  "
              f"flow {r['mean_flow_rel']:+.4%}  tard {r['mean_tardiness_rel']:+.4%}  failures {r['unity_failures']}/"
              f"{r['twin_failures']}  machines equal {r['per_machine_counts_equal']}  "
              f"failure process to common end equal {r['failure_events_to_common_end_equal']} "
              f"({r['failure_events_to_common_end']} events)  "
              f"handlers U/T q{r['unity_queued_rerouted']}/{r['twin_queued_rerouted']} "
              f"c{r['unity_pickups_cancelled']}/{r['twin_pickups_cancelled']} "
              f"r{r['unity_redirected']}/{r['twin_redirected']} h{r['unity_held']}/{r['twin_held']} "
              f"a{r['unity_aborted']}/{r['twin_aborted']} p{r['unity_predispatch_cleared']}/"
              f"{r['twin_predispatch_cleared']} i{r['unity_active_job_at_fail']}/{r['twin_jobs_interrupted']} "
              f"defer {r['unity_deferred_max']} b3 {r['twin_delivered_to_down']}")
    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
