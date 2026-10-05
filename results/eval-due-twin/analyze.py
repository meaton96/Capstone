"""
@file analyze.py
@brief eval-due-twin: twin-trained policies vs the fixed H15 pairs and the switching oracle on held-out seeds 0-39
       (tardiness, 6 h regime-block episodes). Fixed pairs and oracle come from ../rq2-twin-fleet/tasks/B2_s<seed>.json.

Per policy (each checkpoint, plus the untrained init):
  - gap to the best-on-average fixed pair (MDD-TECT on B2) and to each seed's best fixed pair, % of tardiness (< 0 = better)
  - oracle gap closed: (best_avg_pair - policy) / (best_avg_pair - oracle), median and mean over seeds
  - seeds where the policy beats the best-on-average pair
Consistency: MDD-TECT / SRT-TECT here must equal the B2 task values.
@par Usage
@code{.sh}
.venv/bin/python results/eval-due-twin/analyze.py
@endcode
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
B2 = HERE.parent / "rq2-twin-fleet" / "tasks"


def main():
    e = pd.read_csv(HERE / "episodes.csv")
    e["tard"] = e["window_tardiness"] / 1000.0
    ref = {}
    for s in sorted(e.seed.unique()):
        t = json.loads((B2 / f"B2_s{s}.json").read_text())
        ref[s] = {"fixed": {k: v["tard"] for k, v in t["fixed"].items()}, "oracle": t["oracle"]}
    fixed = pd.DataFrame({s: r["fixed"] for s, r in ref.items()}).T
    best_avg = fixed.mean().idxmin()
    oracle = pd.Series({s: r["oracle"] for s, r in ref.items()})
    own_best = fixed.min(axis=1)
    check = [abs(row.tard - ref[row.seed]["fixed"][row.policy]) for row in e[e.kind == "pdr"].itertuples()]
    print(f"eval-due-twin: {e.seed.nunique()} seeds; consistency vs B2 fixed: max |diff| {max(check) if check else float('nan'):.2e}")
    print(f"best-on-average fixed pair: {best_avg} (mean tardiness {fixed[best_avg].mean():.2f}); oracle mean {oracle.mean():.2f}\n")
    rows = []
    for name, g in e[e.kind != "pdr"].groupby("policy"):
        p = g.set_index("seed")["tard"]
        base = fixed.loc[p.index, best_avg]
        den = (base - oracle[p.index]).where(lambda x: x > 1e-9)
        rows.append({"policy": name, "seeds": len(p), "mean_tard": p.mean(),
                     "gap_best_avg_%": 100 * ((p - base) / base).median(),
                     "gap_own_best_%": 100 * ((p - own_best[p.index]) / own_best[p.index]).median(),
                     "oracle_gap_closed_med": ((base - p) / den).median(),
                     "oracle_gap_closed_mean": ((base - p) / den).mean(),
                     "beats_best_avg": int((p < base).sum())})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).set_index("policy").round(3).to_string())
    print("\n(gaps: median % of tardiness, < 0 = better than the pair; oracle gap closed: 1 = as good as the oracle)")


if __name__ == "__main__":
    main()
