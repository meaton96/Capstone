"""
@file compare_regress.py
@brief dev-failure-resume (copied from dev-csharp-failures 10-09): compares two Unity run directories of the same cell (old vs new player)
       file by file: results.csv (every column but the wall-clock ones), decision_log.csv, agv_events.csv,
       job_completions.csv, job_operations.csv, machine_utilization.csv, agv_performance.csv, machine_handoffs.csv,
       des_jobs.json and des_floor.json. Prints IDENTICAL or the differing columns / first differing row per file.
@par Usage
@code{.sh}
/home/ubuntupc/Documents/Github/Capstone/.venv/bin/python compare_regress.py runs/regress/v4_X runs/regress/v5legacy_X
@endcode
"""
import csv
import json
import os
import sys

WALL = {"timestamp", "wall_seconds", "wall_s", "sim_speed", "real_time_factor", "wall_time", "episode_wall_seconds"}
FILES = ("results.csv", "decision_log.csv", "agv_events.csv", "job_completions.csv", "job_operations.csv",
         "machine_utilization.csv", "agv_performance.csv", "machine_handoffs.csv")


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def compare(a, b, quiet=False):
    """Returns {file: None (identical) or a description}."""
    out = {}
    for name in FILES:
        pa, pb = os.path.join(a, name), os.path.join(b, name)
        if not os.path.exists(pa) and not os.path.exists(pb):
            continue
        if not (os.path.exists(pa) and os.path.exists(pb)):
            out[name] = "missing in one run"
            continue
        ra, rb = read(pa), read(pb)
        cols = [c for c in (ra[0].keys() if ra else []) if c not in WALL and not c.startswith("wall")]
        extra = sorted(set(rb[0].keys() if rb else []) - set(cols) - WALL)
        diff_cols = set()
        first = None
        for i, (x, y) in enumerate(zip(ra, rb)):
            d = [c for c in cols if x.get(c) != y.get(c)]
            if d:
                diff_cols.update(d)
                if first is None:
                    first = (i, {c: (x.get(c), y.get(c)) for c in d[:6]})
        if len(ra) != len(rb):
            out[name] = f"rows {len(ra)} vs {len(rb)}; columns differing in the common rows: {sorted(diff_cols)}"
        elif diff_cols:
            out[name] = f"{len(ra)} rows; differing columns {sorted(diff_cols)}; first at row {first[0]}: {first[1]}"
        else:
            out[name] = None if not extra else f"identical, new columns {extra}"
    for name in ("des_jobs.json", "des_floor.json"):
        pa, pb = os.path.join(a, name), os.path.join(b, name)
        if os.path.exists(pa) and os.path.exists(pb):
            with open(pa) as f1, open(pb) as f2:
                out[name] = None if json.load(f1) == json.load(f2) else "differs"
    if not quiet:
        for k, v in out.items():
            print(f"  {k:24s} {'IDENTICAL' if v is None else v}")
    return out


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    print(f"{a}\n  vs {b}")
    res = compare(a, b)
    print("ALL IDENTICAL" if all(v is None for v in res.values()) else "DIFFERENCES")
