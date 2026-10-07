"""Run the event-based twin against Unity run directories exported with "-destrace".

  python -m des_twin.run RUN_DIR [RUN_DIR ...] [--transport instant geometric kinematic] [--out twin]

Each RUN_DIR holds a Unity run (results.csv, des_floor.json, des_jobs.json). The rule and fleet size are taken
from its results.csv unless --rule / --agvs are given. Writes RUN_DIR/<out>/twin_results.csv (one row per
transport model, plus the Unity row's key metrics for pairing) and per-job / per-op CSVs for each model.
"""
import argparse
import csv
import json
import os
import sys
import time

from .engine import TwinConfig, run_twin
from .floor import Floor


def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def unity_row(run_dir):
    with open(os.path.join(run_dir, "results.csv")) as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"{run_dir}: expected one episode in results.csv, found {len(rows)}")
    return rows[0]


def run_dir(path, transports, rule=None, agvs=None, out="twin"):
    u = unity_row(path)
    floor = Floor.load(os.path.join(path, "des_floor.json"))
    with open(os.path.join(path, "des_jobs.json")) as f:
        jobs = json.load(f)
    rule = rule or u["rule"]
    n_agv = agvs if agvs is not None else int(u["agvCount"])
    if int(u.get("episode_failures") or 0) > 0:
        print(f"[twin] WARNING {path}: Unity run had machine failures; the twin does not model them", file=sys.stderr)
    out_dir = os.path.join(path, out)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for tr in transports:
        t0 = time.time()
        twin = run_twin(floor, jobs, TwinConfig(rule=rule, transport=tr, agv_count=n_agv))
        s = twin.summary()
        s.update({
            "instance": u["instance"], "seed": u["seed"], "layout": u.get("layout_id", ""),
            "reservation_protocol": u.get("reservation_protocol", ""), "wall_s": round(time.time() - t0, 2),
            "unity_makespan": float(u["makespan"]), "unity_mean_flow_time": float(u["mean_flow_time"]),
            "unity_p95_flow_time": float(u["p95_flow_time"]), "unity_mean_transport_wait": float(u["mean_transport_wait"]),
            "unity_jobs_censored": int(u["jobs_censored"]), "unity_deadlock": int(u["deadlock_detected"]),
            "unity_decisions": int(u["decisions"]),
        })
        rows.append(s)
        _write(os.path.join(out_dir, f"twin_jobs_{tr}.csv"), twin.job_rows())
        _write(os.path.join(out_dir, f"twin_ops_{tr}.csv"), twin.op_rows())
    _write(os.path.join(out_dir, "twin_results.csv"), rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--transport", nargs="+", default=["instant", "geometric", "kinematic"])
    ap.add_argument("--rule")
    ap.add_argument("--agvs", type=int)
    ap.add_argument("--out", default="twin")
    a = ap.parse_args(argv)
    for d in a.run_dirs:
        for r in run_dir(d, a.transport, a.rule, a.agvs, a.out):
            print(f"{d}  {r['transport']:9s} agv={r['agv_count']:2d}  flow={r['mean_flow_time']:9.2f} "
                  f"(unity {r['unity_mean_flow_time']:9.2f})  makespan={r['makespan']:9.2f} "
                  f"(unity {r['unity_makespan']:9.2f})  wall={r['wall_s']}s")


if __name__ == "__main__":
    main()
