"""
@file analyze.py
@brief rq2-twin-scen: rank the training regimes by tardiness switching headroom (twin, H15 oracle). Works on partial
       results (counts seeds per regime).

Per regime:
  - total_%   oracle vs the regime's best-on-average fixed pair (what a policy must beat), median / mean / share > 2%
  - switch_%  oracle vs each seed's own best fixed pair (headroom from switching alone)
  - saved     job-seconds of tardiness the oracle saves per window vs the regime's best pair (absolute, see the
              rq2-twin-due caveat: tardiness is a small base, so report it next to the percentage)
  - tard      mean tardiness of the best pair (1000 job-s per window) and windows with no tardiness under some pair
  - late_%    share of exited jobs late under the best pair

@par Usage
@code{.sh}
.venv/bin/python results/rq2-twin-scen/analyze.py > results/rq2-twin-scen/analysis.out
@endcode
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    rows = []
    for f in sorted((HERE / "tasks").glob("*.json")):
        r = json.loads(f.read_text())
        for pair, v in r["fixed"].items():
            rows.append({"regime": r["regime"], "c": r["allowance"], "agvs": r["agvs"], "load": r["load"],
                         "seed": r["seed"], "pair": pair, "tard": v["tard"], "tis": v["tis"],
                         "late": v["late_exited"], "exited": v["jobs_exited"], "oracle": r["oracle"]})
    if not rows:
        print("no tasks yet")
        return
    d = pd.DataFrame(rows)
    out = []
    for regime, g in d.groupby("regime"):
        piv = g.pivot(index="seed", columns="pair", values="tard")
        best_avg = piv.mean().idxmin()
        oracle = g.groupby("seed")["oracle"].first()
        base = piv[best_avg]
        ok = base > 1e-9
        total = 100 * (base[ok] - oracle[ok]) / base[ok]
        own = piv.min(axis=1)
        sw = 100 * (own[own > 1e-9] - oracle[own > 1e-9]) / own[own > 1e-9]
        bg = g[g.pair == best_avg]
        out.append({"regime": regime, "seeds": len(piv), "best_pair": best_avg,
                    "total_med_%": total.median(), "total_mean_%": total.mean(), "gt2_%": 100 * (total > 2).mean(),
                    "switch_med_%": sw.median(), "saved_js": 1000 * (base - oracle).mean(),
                    "tard": base.mean(), "zero_windows": int((piv.min(axis=1) <= 1e-9).sum()),
                    "late_%": 100 * bg.late.sum() / max(bg.exited.sum(), 1),
                    "tis": bg.tis.mean()})
    t = pd.DataFrame(out).set_index("regime").sort_values("total_med_%", ascending=False)
    pd.set_option("display.width", 200)
    print(f"rq2-twin-scen: {len(d.groupby(['regime', 'seed']))} tasks, {t.seeds.min()}-{t.seeds.max()} seeds per regime\n")
    print(t.round(2).to_string())
    print("\nBy lever (median over regimes of total_med_%):")
    for lever in ("c", "agvs", "load"):
        per = {k: t.loc[[r for r in t.index if r in set(d[d[lever] == k].regime)], "total_med_%"].median()
               for k in sorted(d[lever].unique())}
        print(f"  {lever}: " + ", ".join(f"{k}: {v:.2f}" for k, v in per.items()))
    t.to_csv(HERE / "regimes.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
