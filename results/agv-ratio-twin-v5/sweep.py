"""agv-ratio-twin-v5 copy of ../agv-ratio-scale/sweep.py that runs twin_point_v5.py (due dates, tardiness).
Twin fleet sweep: for each (M, fleet, seed) run twin_point_v5.py, collect JSON rows into a CSV.

usage: sweep.py OUT.csv --m 50 --floor floors/m50/des_floor.json --agvs 5 8 10 ... --seeds 0 1 [--workers 12]
       [--extra "--util 0.9 0.9"]
Resumable: rows already in OUT.csv are skipped.
"""
import argparse, csv, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "../../.venv/bin/python")
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--m", type=int, required=True); ap.add_argument("--floor", required=True)
ap.add_argument("--agvs", type=int, nargs="+", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--rules", nargs="+", default=["SRT_TECT"]); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--extra", default="", help="extra twin_point args, one string")
a = ap.parse_args()
done = set()
if os.path.exists(a.out):
    for r in csv.DictReader(open(a.out)):
        done.add((int(r["m"]), int(r["agv_count"]), int(r["seed"]), r["rule"]))
cells = [(n, s, r) for r in a.rules for s in a.seeds for n in a.agvs if (a.m, n, s, r) not in done]

def run(c):
    n, s, r = c
    cmd = [PY, os.path.join(HERE, "twin_point_v5.py"), a.floor, str(a.m), str(n), "--seed", str(s), "--rule", r,
           "--json", *a.extra.split()]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode:
        print("FAIL", c, out.stderr[-500:], flush=True); return None
    return json.loads(out.stdout.strip().splitlines()[-1])

rows = []
with ThreadPoolExecutor(a.workers) as ex:
    for row in ex.map(run, cells):
        if row:
            rows.append(row)
            print(f"m={row['m']} agv={row['agv_count']:3d} s={row['seed']} {row['rule']} tis={row['mean_tis']:.0f} "
                  f"agv_busy={row['agv_busy_fraction']:.2f} mach={row['machine_util_mean']:.2f} wall={row['wall_s']}s", flush=True)
if rows:
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new: w.writeheader()
        w.writerows(rows)
