"""
@file analyze.py
@brief rq2-direct-headroom: per cell (rule / direct x hindsight / expected8), gain of the one-step rollout over the
       MDD-TECT base on B2 test seeds 0-39 (window tardiness), paired direct - rule difference, and how often the
       rollout leaves the base / the rule set. Reads <setting>/<cell>/s*.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
setting = sys.argv[1] if len(sys.argv) > 1 else "B2"
cells = {}
for d in sorted((HERE / setting).glob("*")):
    rows = {r["seed"]: r for r in (json.loads(f.read_text()) for f in d.glob("s*.json"))}
    if rows:
        cells[d.name] = rows


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


print(f"== {setting}: one-step rollout over MDD-TECT, gain in window tardiness (% of base, + = better)")
print(f"{'cell':22s} {'n':>3s} {'mean gain %':>12s} {'95% CI':>8s} {'pooled %':>9s} {'decisions':>9s} "
      f"{'deviate':>8s} {'outside rules':>13s} {'capped':>7s}")
for name, rows in cells.items():
    r = list(rows.values())
    m, h = ci([x["gain_pct"] for x in r])
    pooled = 100 * (sum(x["base"] for x in r) - sum(x["rollout"] for x in r)) / sum(x["base"] for x in r)
    print(f"{name:22s} {len(r):3d} {m:12.2f} {h:8.2f} {pooled:9.2f} {np.mean([x['decisions_rolled'] for x in r]):9.1f} "
          f"{np.mean([x['deviations'] for x in r]):8.1f} {np.mean([x['outside_rule_set'] for x in r]):13.1f} "
          f"{np.mean([x['capped'] for x in r]):7.1f}")
print("\n== paired: direct - rule (percentage points of base, same seeds)")
for mode in ("hindsight", "expected8", "expected32"):
    a, b = cells.get(f"direct-{mode}"), cells.get(f"rule-{mode}")
    if not a or not b:
        continue
    seeds = sorted(set(a) & set(b))
    m, h = ci([a[s]["gain_pct"] - b[s]["gain_pct"] for s in seeds])
    print(f"{mode:10s} n={len(seeds):2d}  {m:+.2f} +- {h:.2f}")
