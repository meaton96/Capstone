"""
@file analyze.py
@brief dev-agv105-probe (2026-10-15): simulation speed of the player on 105-machine floors. Per probe cell: jobs exited
       vs wall time (sim.log, --loglevel Medium stamps every exit), finished or not, sim makespan / wall time, and a
       projection for the full 6 h agv-study-rerun cell. A 15-machine reference cell of agv-study-rerun gives the
       baseline speed. -> analysis.out
"""
import csv
import io
import re
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = REPO / "linux_server" / "Results"
TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)")
FULL = {"agv5_b2t_t7open_arr1h_s0": ("agv5_b2t_t7open_s0", 5893, 21390.0),
        "agv5_b2m_t7iso_arr1h_s0": ("agv5_b2m_t7iso_s0", 2021, 20854.0)}
REF = RES / "agv-v5-rel" / "agv5_b2t_s0" / "D" / "agv7_MDD_PTECT_s0"


def stamp(line):
    m = TS.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f") if m else None


def cell(d):
    start, jobs, exits, last = None, None, [], None
    for line in (d / "sim.log").read_text(errors="replace").splitlines():
        t = stamp(line)
        if t is None:
            continue
        last = t
        if "Episode started:" in line and start is None:
            start = t
            jobs = int(re.search(r"Episode started: (\d+) jobs", line).group(1))
        elif "exited the system" in line:
            exits.append(t)
    res = None
    if (d / "results.csv").exists():
        res = next(csv.DictReader(open(d / "results.csv")))
    return start, jobs, exits, last, res


def wall_h(a, b):
    return (b - a).total_seconds() / 3600.0


def main():
    if (REF / "results.csv").exists():
        s, _, _, last, r = cell(REF)
        end = datetime.strptime(r["timestamp"][:19].replace("T", " "), "%Y-%m-%d %H:%M:%S") if r.get("timestamp") \
            else last
        w = (end - s).total_seconds()
        print(f"reference 15-machine cell {REF.relative_to(RES)}: makespan {float(r['makespan']):.0f} sim-s in "
              f"{w / 60:.0f} wall-min = {float(r['makespan']) / w:.1f} sim-s per wall-s")
    for d in sorted((RES / "dev-agv105-probe").glob("*/D/agv*")):
        start, jobs, exits, last, res = cell(d)
        name = d.parts[-3]
        print(f"\n== {name} / {d.name}: {jobs} jobs")
        if start is None:
            print("   not started")
            continue
        span = wall_h(start, last)
        print(f"   wall {span:.2f} h since start, {len(exits)} / {jobs} jobs exited, "
              f"{'finished' if res else 'NOT finished (timeout or still running)'}")
        for h in (0.25, 0.5, 1, 2, 3):
            n = sum(1 for t in exits if wall_h(start, t) <= h)
            if h <= span + 1e-9:
                print(f"   exited by {h:>4} wall-h: {n}")
        full, n_full, arr_span = FULL.get(name, (None, None, None))
        if res:
            mk = float(res["makespan"])
            speed = mk / (span * 3600.0)
            print(f"   makespan {mk:.0f} sim-s, {speed:.2f} sim-s per wall-s")
            if full:
                print(f"   full cell {full} ({n_full} jobs, arrivals over {arr_span:.0f} s): at least "
                      f"{arr_span / speed / 3600:.1f} wall-h before the drain")
        elif exits and full:
            rate = len(exits) / span
            print(f"   exit rate {rate:.0f} jobs per wall-h; at that rate the full cell ({n_full} jobs) needs about "
                  f"{n_full / rate:.0f} wall-h")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
