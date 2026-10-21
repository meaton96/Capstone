"""
@file analyze.py
@brief eval-v5-slot (2026-10-21): PPO decision interval sweep. Per arm (slot length X; 900 s = the final-policy run
       train-v5-cmix): each training seed and the family mean vs MDD-PTECT on B2 test seeds 0-39 (pooled window
       tardiness, seed-bootstrap 95% CI, Wilcoxon), the spread over training seeds, and the training wall time. Next to
       it, the switching ceiling at the same interval (../rq2-replan 1a, exact forecast) and the planner with a 10-min
       arrival error (1b), for the joint figure. -> analysis.out, summary.csv, curve.csv
"""
import importlib.util
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent
_s = importlib.util.spec_from_file_location("rq3an", R / "rq3-gen-v5" / "analyze.py")
rq3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rq3)
H = {"config_hash": str, "instance_hash": str}
SLOTS = (60, 120, 180, 300, 600, 900, 1200, 1500, 1800)
REF = "MDD-PTECT"


def arm_table(x):
    f = HERE / "base" / f"slot{x}" / "episodes.csv"
    if not (f.exists() and (f.parent / "DONE").exists()):
        return None
    e = pd.read_csv(f, dtype=H)
    e["policy"] = e.policy.map(lambda p: "s" + re.search(r"_s(\d+)", p).group(1) if "checkpoint" in str(p) else p)
    return e.pivot_table(index="seed", columns="policy", values="window_tardiness") / 1000.0


def train_hours(x):
    name = "train-v5-cmix" if x == 900 else f"train-v5-slot{x}"
    hs = []
    for f in sorted((R / name).glob("train_s*.out")):
        m = re.findall(r"Total training time: ([\d.]+)s", f.read_text())
        if m:
            hs.append(float(m[-1]) / 3600)
    return hs


def replan(variant):
    d = R / "rq2-replan" / "out" / variant
    fs = list(d.glob("s*.json"))
    if len(fs) < 40:
        return None
    real = {json.loads(f.read_text())["seed"]: json.loads(f.read_text())["real"] for f in fs}
    ref = {json.loads(f.read_text())["seed"]: json.loads(f.read_text())["mdd_ptect"] for f in fs}
    return rq3.compare(real, ref, np.random.default_rng(0))


def main():
    rng = np.random.default_rng(0)
    print("== eval-v5-slot: PPO decision interval sweep (final-policy recipe, equal simulated experience), B2 test seeds "
          "0-39, % vs MDD-PTECT pooled; < 0 better")
    rows, curve = [], []
    for x in SLOTS:
        t = arm_table(x)
        hs = train_hours(x)
        oracle = replan(f"exact-d{x}") if x in (60, 120, 180, 300, 600, 900, 1200, 1800) else None
        planner = replan(f"jit10-d{x}") if x in (120, 300, 900, 1800) else None
        if t is None:
            print(f"\n-- slot {x} s: not evaluated yet")
            continue
        seeds = sorted(c for c in t.columns if c.startswith("s"))
        t["family"] = t[seeds].mean(axis=1)
        per = {k: 100 * (t[k].sum() / t[REF].sum() - 1) for k in seeds}
        c = rq3.compare(t["family"].to_dict(), t[REF].to_dict(), rng)
        print(f"\n-- slot {x} s ({len(seeds)} training seeds; training {np.mean(hs):.1f} h each)" if hs else
              f"\n-- slot {x} s ({len(seeds)} training seeds)")
        print(f"   family mean vs MDD-PTECT {c['pooled_%']:+6.2f} [{c['ci_lo']:+6.2f}, {c['ci_hi']:+6.2f}] "
              f"{c['better']}/{c['worse']} p={c['wilcoxon_p']:.2g} | training seeds "
              + " ".join(f"{k} {v:+.2f}" for k, v in per.items())
              + f" (sd {np.std(list(per.values()), ddof=1):.2f})")
        if oracle:
            print(f"   switching ceiling at {x} s (exact forecast, rq2-replan): {oracle['pooled_%']:+.2f}"
                  + (f"; planner at 10-min arrival error: {planner['pooled_%']:+.2f}" if planner else ""))
        rows.append({"slot_s": x, "n_train_seeds": len(seeds), **{f"family_{k}": v for k, v in c.items()},
                     **{f"seed_{k}_%": v for k, v in per.items()}, "seed_sd_%": float(np.std(list(per.values()), ddof=1)),
                     "train_hours_mean": float(np.mean(hs)) if hs else None,
                     "ceiling_exact_%": oracle["pooled_%"] if oracle else None,
                     "planner_jit10_%": planner["pooled_%"] if planner else None})
        curve.append({"slot_s": x, "ppo_family_%": c["pooled_%"], "ppo_ci_lo": c["ci_lo"], "ppo_ci_hi": c["ci_hi"],
                      "ppo_seed_min_%": min(per.values()), "ppo_seed_max_%": max(per.values()),
                      "ceiling_exact_%": oracle["pooled_%"] if oracle else None,
                      "planner_jit10_%": planner["pooled_%"] if planner else None})
    pd.DataFrame(rows).to_csv(HERE / "summary.csv", index=False)
    pd.DataFrame(curve).to_csv(HERE / "curve.csv", index=False)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
