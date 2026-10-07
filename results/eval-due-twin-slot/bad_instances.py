"""
@file bad_instances.py
@brief eval-due-twin-slot: where do the slot policies lose to MDD-TECT? Per held-out seed: the policy's gap to MDD-TECT
       (seeds 1 and 2), the instance's regime mix inside the 6 h agent window (share of surge / short-fleet blocks, mean
       c and op mean, weighted by overlap), warm-up, MDD-TECT's tardiness level and its gap to the seed's own best pair,
       and the oracle's gain. Prints the worst seeds, the rest for contrast, and Spearman correlations.
@par Usage
@code{.sh}
.venv/bin/python results/eval-due-twin-slot/bad_instances.py > results/eval-due-twin-slot/bad_instances.out
@endcode
"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "env"))
_spec = importlib.util.spec_from_file_location("fleet", REPO / "results/rq2-twin-fleet/run.py")
fleet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fleet)
WINDOW = 21600.0


def features(seed):
    sc = fleet.scenario("B2", seed)
    w0 = float(sc["stochastic"].get("warmupSeconds") or 0.0)
    w1 = w0 + WINDOW
    acc = {"surge": 0.0, "short": 0.0, "normal": 0.0}
    c = op = 0.0
    for b in sc["_meta"]["blocks"]:
        ov = max(0.0, min(w1, b["end"]) - max(w0, b["start"])) / WINDOW
        acc[b["load_profile"]] += ov
        c += ov * b["due_date_allowance"]
        op += ov * b["op_mean"]
    return {"warmup_h": w0 / 3600, "surge_share": acc["surge"], "short_share": acc["short"], "mean_c": c,
            "mean_op": op, "jobs": sum(w0 <= j["arrivalTime"] < w1 for j in sc["jobs"])}


def main():
    e = pd.read_csv(HERE / "episodes.csv")
    e["tard"] = e.window_tardiness / 1000
    rows = []
    for s in sorted(e.seed.unique()):
        t = json.loads((REPO / f"results/rq2-twin-fleet/tasks/B2_s{s}.json").read_text())
        fx = {k: v["tard"] for k, v in t["fixed"].items()}
        mt = fx["MDD-TECT"]
        r = {"seed": s, "mddtect": mt, "mddtect_vs_own_best_%": 100 * (mt - min(fx.values())) / min(fx.values()),
             "own_best": min(fx, key=fx.get), "oracle_gain_%": 100 * (mt - t["oracle"]) / mt, **features(s)}
        for k in (1, 2):
            p = e[(e.seed == s) & (e.policy.str.contains(f"train-due-twin-slot_s{k}/checkpoint$"))].tard.iloc[0]
            r[f"gap_s{k}_%"] = 100 * (p - mt) / mt
        r["gap_mean_%"] = (r["gap_s1_%"] + r["gap_s2_%"]) / 2
        rows.append(r)
    d = pd.DataFrame(rows).set_index("seed").sort_values("gap_mean_%", ascending=False)
    pd.set_option("display.width", 220)
    cols = ["gap_s1_%", "gap_s2_%", "mddtect", "mddtect_vs_own_best_%", "own_best", "oracle_gain_%", "surge_share",
            "short_share", "mean_c", "mean_op", "warmup_h", "jobs"]
    print("== worst 8 seeds (mean of s1/s2 gap to MDD-TECT)")
    print(d[cols].head(8).round(2).to_string())
    print("\n== group means: worst 8 vs the other 32")
    num = [c for c in cols if c != "own_best"]
    print(pd.DataFrame({"worst8": d[num].head(8).mean(), "rest": d[num].iloc[8:].mean()}).round(2).to_string())
    print("\n== Spearman correlation with the mean policy gap (40 seeds)")
    print(d[num + ["gap_mean_%"]].corr(method="spearman")["gap_mean_%"].drop(["gap_s1_%", "gap_s2_%", "gap_mean_%"])
          .round(2).to_string())
    d.to_csv(HERE / "bad_instances.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
