"""
@file sim_rules.py
@brief rq3-size-m1000 (2026-10-20, user): the column's two rule groups (mdd = MDD-PTECT, srt = SRT-PTECT) played in the
       oracle's frozen simulator (../rq3-size-oracle/sizesim.py on m1000; frozen v5 code and sim.py of rq2-rebase-v5)
       instead of evaluate.py on the cluster. Same instances and window tardiness (the simulator's MDD-PTECT equals the
       grid's at 15-200 machines), far faster because no policy observations are built. The final-policy group (t15)
       stays on the cluster through task.py.
       Writes m1000/<group>_s<seed>/ as task.py does: episodes.csv (policy, kind, seed, window_tardiness in job-s,
       harness), DONE, run.json (seconds, peak RSS). The mdd seed-0 task is the check: it must equal the local probe's
       evaluate.py value (../dev-size-1000-probe, 53,335.636 k job-s), else no DONE and exit 1.
@par Usage
@code{.sh}
python results/rq3-size-m1000/sim_rules.py mdd 0
@endcode
"""
import csv
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
PAIRS = {"mdd": "MDD-PTECT", "srt": "SRT-PTECT", "tect": "MDD-TECT"}   # tect added 10-21
PROBE = R / "dev-size-1000-probe" / "m1000" / "ref" / "episodes.csv"


def probe_value():
    with open(PROBE) as f:
        return float(next(csv.DictReader(f))["window_tardiness"]) / 1000.0


def main(group, seed):
    out = HERE / "m1000" / f"{group}_s{seed}"
    if (out / "DONE").exists():
        print(f"m1000/{group}_s{seed}: done, skipped")
        return
    spec = importlib.util.spec_from_file_location("rb_run", R / "rq2-rebase-v5" / "run.py")    # frozen code first
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    spec = importlib.util.spec_from_file_location("sizesim", R / "rq3-size-oracle" / "sizesim.py")
    ss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss)
    ss.apply(rb.sim, "m1000")
    t = time.time()
    v = rb.sim.Episode("B2", seed).play([PAIRS[group]])[0]
    sec = time.time() - t
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    check = None
    if group == "mdd" and seed == 0:
        p = probe_value()
        check = {"probe_evaluate_py": p, "simulator": v, "match": abs(v - p) <= 1e-6 * abs(p) + 1e-4}
        print(f"check: MDD-PTECT seed 0 at 1000 machines: simulator {v:,.6f} vs probe (evaluate.py) {p:,.6f} k job-s "
              f"-> {'OK' if check['match'] else 'MISMATCH'}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"group": group, "seed": seed, "pair": PAIRS[group], "tardiness_k": v,
                                              "seconds": sec, "peak_rss_gb": rss_gb, "check": check,
                                              "stamp": rb.stamp()}, indent=1))
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "kind", "seed", "window_tardiness", "harness"])
        w.writerow([PAIRS[group], "pdr", seed, v * 1000.0, "oracle simulator (sim_rules.py)"])
    if check is not None and not check["match"]:
        raise SystemExit(1)
    (out / "DONE").write_text(f"{sec:.0f}\n")
    print(f"m1000/{group}_s{seed}: {v:,.3f} k job-s in {sec:.0f} s, peak {rss_gb:.1f} GB")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
