"""
@file analyze.py
@brief rq2-oracle-screen: headroom per regime from the switching-oracle runs.

@details For each regime, over its finished seeds (time in system = -return of the flow_time reward):
  - best_pair: the fixed pair with the lowest mean time in system (what a single rule would be chosen as);
  - static_%: best_pair's mean gap to the per-seed best fixed pair (headroom from picking a pair per instance);
  - switch_%: the oracle's mean gain over the per-seed best fixed pair (headroom from switching within an episode);
  - total_%: the oracle's mean gain over best_pair (what an RL policy could at most hope to beat the best rule by,
    at this oracle's granularity), with min / max over seeds;
  - spread_%: mean gap of the worst pair to the per-seed best (how much the rule choice matters at all).
Gains are positive = better than the reference. Writes summary.csv next to this script and prints it sorted by total_%.

@par Usage
@code{.sh}
.venv/bin/python results/rq2-oracle-screen/analyze.py [results/rq2-oracle-screen]
@endcode
"""

import json
import sys
from pathlib import Path

import pandas as pd


def regime_rows(root: Path) -> pd.DataFrame:
    rows = []
    for result in sorted(root.glob("*/s*/result.json")):
        r = json.loads(result.read_text())
        fixed = {pair: -ret for pair, ret in r["fixed_returns"].items()}   # time in system, lower is better
        rows.append({"regime": result.parent.parent.name, "seed": r["seed"], "oracle": -r["oracle_return"],
                     "schedule": "|".join(r["schedule"]), **{f"fix:{k}": v for k, v in fixed.items()}})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for regime, g in df.groupby("regime"):
        fixed = g[[c for c in g.columns if c.startswith("fix:")]].dropna(axis=1, how="all")
        fixed.columns = [c[4:] for c in fixed.columns]
        per_seed_best = fixed.min(axis=1)
        per_seed_worst = fixed.max(axis=1)
        best_pair = fixed.mean().idxmin()
        static = (fixed[best_pair] / per_seed_best - 1) * 100
        switch = (1 - g["oracle"] / per_seed_best) * 100
        total = (1 - g["oracle"] / fixed[best_pair]) * 100
        spread = (per_seed_worst / per_seed_best - 1) * 100
        job_rules = g["schedule"].str.split("|").explode().str.split("-").str[0].value_counts(normalize=True)
        out.append({
            "regime": regime, "seeds": len(g), "best_pair": best_pair,
            "static_%": static.mean(), "switch_%": switch.mean(), "total_%": total.mean(),
            "total_min_%": total.min(), "total_max_%": total.max(), "seeds_total>2%": int((total > 2).sum()),
            "spread_%": spread.mean(),
            "oracle_job_rules": ", ".join(f"{k} {v:.0%}" for k, v in job_rules.head(3).items()),
        })
    return pd.DataFrame(out).sort_values("total_%", ascending=False)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    df = regime_rows(root)
    if df.empty:
        print("no finished seeds yet")
        return
    summary = summarize(df)
    summary.to_csv(root / "summary.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
