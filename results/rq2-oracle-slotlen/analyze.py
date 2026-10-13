"""
@file analyze.py
@brief rq2-oracle-slotlen: switching headroom vs slot length. 900 s from rq2-twin-fleet (B2 tasks), the rest from
       seg<G>/s<seed>.json. Per slot length: oracle gain over MDD-TECT (the best pair on average) and over each seed's
       own best fixed pair, the share of the 900 s gain kept, and how often the oracle switches.
@par Usage
@code{.sh}
.venv/bin/python results/rq2-oracle-slotlen/analyze.py
@endcode
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
B2 = HERE.parent / "rq2-twin-fleet" / "tasks"


def main():
    rows = []
    for f in sorted(B2.glob("B2_s*.json")):
        t = json.loads(f.read_text())
        fx = {k: v["tard"] for k, v in t["fixed"].items()}
        rows.append({"segment": 900, "seed": t["seed"], "oracle": t["oracle"], "mddtect": fx["MDD-TECT"],
                     "own_best": min(fx.values()), "switches": sum(a != b for a, b in zip(t["schedule"], t["schedule"][1:])),
                     "stages": len(t["schedule"])})
    for f in sorted(HERE.glob("seg*/s*.json")):
        t = json.loads(f.read_text())
        rows.append({"segment": int(t["segment"]), "seed": t["seed"], "oracle": t["oracle"], "mddtect": t["fixed"]["MDD-TECT"],
                     "own_best": t["best_fixed"], "switches": sum(a != b for a, b in zip(t["schedule"], t["schedule"][1:])),
                     "stages": t["stages"]})
    d = pd.DataFrame(rows)
    d["gain_vs_mddtect_%"] = 100 * (d.mddtect - d.oracle) / d.mddtect
    d["gain_vs_own_best_%"] = 100 * (d.own_best - d.oracle) / d.own_best
    g = d.groupby("segment")
    s = pd.DataFrame({"seeds": g.seed.nunique(), "stages": g.stages.first(),
                      "mean_tard_oracle": g.oracle.mean(), "mean_tard_mddtect": g.mddtect.mean(),
                      "gain_vs_mddtect_mean_%": 100 * (1 - g.oracle.mean() / g.mddtect.mean()),
                      "gain_vs_mddtect_median_%": g["gain_vs_mddtect_%"].median(),
                      "gain_vs_own_best_median_%": g["gain_vs_own_best_%"].median(),
                      "switches_mean": g.switches.mean()})
    s["share_of_900s_gain"] = s["gain_vs_mddtect_mean_%"] / s.loc[900, "gain_vs_mddtect_mean_%"]
    pd.set_option("display.width", 200)
    print("== switching headroom by slot length (B2, twin, tardiness; window 21,600 s)")
    print(s.round(3).to_string())
    common = set.intersection(*[set(x.seed) for _, x in g])
    print(f"\n(seeds present at every slot length: {len(common)}; 10,800 s = 2 stages, i.e. one switch at 3 h at most)")


if __name__ == "__main__":
    main()
