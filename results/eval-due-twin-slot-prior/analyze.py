"""
@file analyze.py
@brief eval-due-twin-slot-prior: action prior (train-due-twin-slot-prior-kl) vs the 100k baseline (train-due-twin-slot
       s0-2 from ../eval-due-twin-slot, s3-4 from this run) on held-out B2 seeds 0-39, against MDD-TECT and the oracle.

Per policy: mean / median gap to MDD-TECT, instances lost, worst instance, Wilcoxon vs MDD-TECT, oracle gap closed.
Per arm (5 training seeds each): mean of the per-seed mean gaps, spread over seeds, Mann-Whitney between arms.
Bad instances: the top quartile of the 400k policies' gap (../eval-due-twin-slot-long/workup/features_0_39.csv), the
instances where they lost because MDD-TECT was right: does the prior stop those losses?
@par Usage
@code{.sh}
.venv/bin/python results/eval-due-twin-slot-prior/analyze.py
@endcode
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

HERE = Path(__file__).resolve().parent
B2 = HERE.parent / "rq2-twin-fleet" / "tasks"
WORKUP = HERE.parent / "eval-due-twin-slot-long" / "workup" / "features_0_39.csv"


def arm(policy):
    if "prior-kl" in policy:
        return "init only" if "checkpoint_init" in policy else "prior"
    if "checkpoint_init" in policy:
        return "init"
    return "baseline"


def main():
    e = pd.concat([pd.read_csv(HERE / "episodes.csv"),
                   pd.read_csv(HERE.parent / "eval-due-twin-slot" / "episodes.csv")], ignore_index=True)
    e = e[~(e.policy.str.contains("checkpoint_init") & ~e.policy.str.contains("prior-kl"))]
    e = e.drop_duplicates(["policy", "seed"])
    e["tard"] = e.window_tardiness / 1000.0
    fixed = pd.DataFrame({s: {k: v["tard"] for k, v in json.loads((B2 / f"B2_s{s}.json").read_text())["fixed"].items()}
                          for s in sorted(e.seed.unique())}).T
    oracle = pd.Series({s: json.loads((B2 / f"B2_s{s}.json").read_text())["oracle"] for s in fixed.index})
    check = [abs(r.tard - fixed.loc[r.seed, r.policy]) for r in e[e.kind == "pdr"].itertuples()]
    print(f"{e.seed.nunique()} seeds; consistency vs B2 fixed: max |diff| {max(check):.2e}")
    mt = fixed["MDD-TECT"]
    print(f"MDD-TECT mean tardiness {mt.mean():.2f}; oracle {oracle.mean():.2f} ({100 * (oracle.mean() / mt.mean() - 1):+.1f}%)\n")
    bad = set()
    if WORKUP.exists():
        w = pd.read_csv(WORKUP, index_col="seed")
        bad = set(w[w.gap >= w.gap.quantile(0.75)].index)
    rows = []
    for name, g in e[e.kind != "pdr"].groupby("policy"):
        p = g.set_index("seed").tard.reindex(fixed.index)
        gap = 100 * (p - mt) / mt
        den = (mt - oracle).where(lambda x: x > 1e-9)
        rows.append({"policy": name.replace("ckpt:", "").replace("/checkpoint", "").replace(".pt", ""), "arm": arm(name),
                     "mean_gap_%": 100 * (p.mean() / mt.mean() - 1), "median_gap_%": gap.median(),
                     "lost": int((gap > 1e-3).sum()), "tied": int((gap.abs() <= 1e-3).sum()), "worst_%": gap.max(),
                     "bad_q_mean_gap_%": gap[gap.index.isin(bad)].mean() if bad else np.nan,
                     "rest_mean_gap_%": gap[~gap.index.isin(bad)].mean() if bad else np.nan,
                     "wilcoxon_p": wilcoxon(p, mt).pvalue if (p != mt).any() else 1.0,
                     "oracle_closed_med": ((mt - p) / den).median()})
    t = pd.DataFrame(rows).set_index("policy").sort_values(["arm", "policy"])
    pd.set_option("display.width", 220)
    print(t.round(3).to_string())
    print("\n(gaps: % of MDD-TECT tardiness, < 0 = better, |gap| <= 0.001% counted as a tie (float noise ~1e-5); bad_q = the 10 instances the 400k policies lost most on)")
    print("\n== arms (one value per training seed)")
    arms = {a: t[t.arm == a] for a in ("baseline", "prior")}
    for a, d in arms.items():
        if len(d):
            print(f"  {a:9s} n={len(d)}  mean gap {d['mean_gap_%'].mean():+.2f}% (seeds {d['mean_gap_%'].min():+.2f} .. "
                  f"{d['mean_gap_%'].max():+.2f})  median gap {d['median_gap_%'].mean():+.2f}%  lost {d.lost.mean():.1f}/40  "
                  f"worst {d['worst_%'].mean():+.1f}%  bad-quartile gap {d['bad_q_mean_gap_%'].mean():+.2f}%  "
                  f"rest {d['rest_mean_gap_%'].mean():+.2f}%")
    if all(len(d) >= 3 for d in arms.values()):
        for col in ("mean_gap_%", "lost", "worst_%", "bad_q_mean_gap_%"):
            p = mannwhitneyu(arms["prior"][col], arms["baseline"][col]).pvalue
            print(f"  Mann-Whitney prior vs baseline on {col}: p = {p:.3f}")
    dl = HERE / "decisions.csv"
    if dl.exists():
        a = pd.read_csv(dl)
        a = a[a.policy.str.startswith("ckpt") & ~a.policy.str.contains("init")]
        print("\n== slot choices (share of slots, seeds 0-39)")
        print(a.groupby("policy").rule.value_counts(normalize=True).round(3).groupby(level=0).head(4).to_string())


if __name__ == "__main__":
    main()
