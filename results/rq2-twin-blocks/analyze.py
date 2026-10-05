"""
@file analyze.py
@brief rq2-twin-blocks: in-episode switching headroom with one regime per episode (single) vs regime blocks (blocks).
       Works on partial results.

Per setting, over seeds:
  - switch_%   oracle vs the seed's own best fixed pair: the in-episode switching headroom (the key number)
  - total_%    oracle vs the setting's best-on-average fixed pair (what a policy must beat)
  - saved/h    job-seconds of tardiness the oracle saves per simulated hour vs the seed's best fixed pair
  - blocks only: blocksel_% = best fixed pair chosen per block with hindsight (slot tardiness of the fixed runs summed
    per block) vs the seed's best fixed pair. Not achievable (state carries over), but it shows how much of the oracle's
    gain is picking the right pair per regime; oracle_vs_blocksel_% is the rest.
Then the oracle's job-rule shares by the slot's regime (c tercile, load profile), to see whether choices track regimes.

@par Usage
@code{.sh}
.venv/bin/python results/rq2-twin-blocks/analyze.py > results/rq2-twin-blocks/analysis.out
@endcode
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WINDOW = {"single": 5400.0, "blocks": 21600.0, "long": 21600.0, "blocks-base": 21600.0}


def pct(base, val):
    return 100.0 * (base - val) / base if base > 1e-9 else np.nan


def main():
    import sys
    dirs = [HERE] + [Path(a) for a in sys.argv[1:]]    # extra task dirs, e.g. ../rq2-twin-blocks-ctrl
    tasks = [json.loads(f.read_text()) for d in dirs for f in sorted((d / "tasks").glob("*.json"))]
    if not tasks:
        print("no tasks yet")
        return
    rows, shares = [], defaultdict(Counter)
    for t in tasks:
        st = t["setting"]
        fixed = {k: v["tard"] for k, v in t["fixed"].items()}
        own = min(fixed.values())
        r = {"setting": st, "seed": t["seed"], "oracle": t["oracle"], "own_best": own,
             "own_best_pair": t["best_fixed_pair"], **{f"fx_{k}": v for k, v in fixed.items()},
             "switch_%": pct(own, t["oracle"]),
             "saved_per_h": 1000.0 * (own - t["oracle"]) / (WINDOW[st] / 3600.0)}
        if st.startswith("blocks"):
            slots = {k: np.array(v["slots"]) for k, v in t["fixed"].items()}
            sb = np.array(t["slot_block"])
            blocksel = sum(min(s[sb == b].sum() for s in slots.values()) for b in np.unique(sb))
            r["blocksel_%"] = pct(own, blocksel)
            r["oracle_vs_blocksel_%"] = pct(blocksel, t["oracle"])
            r["block_best_pairs"] = len({min(slots, key=lambda k: slots[k][sb == b].sum()) for b in np.unique(sb)})
        rows.append(r)
        cs = [reg["c"] for reg in t["regimes"]]
        for k, pair in enumerate(t["schedule"]):
            reg = t["regimes"][t["slot_block"][k]]
            job = pair.split("-")[0]
            shares[(st, "c<2.0" if reg["c"] < 2.0 else "c2.0-2.25" if reg["c"] < 2.25 else "c>=2.25")][job] += 1
            shares[(st, reg["load"])][job] += 1
            if k > 0:
                same_block = t["slot_block"][k] == t["slot_block"][k - 1]
                changed = pair != t["schedule"][k - 1]
                shares[(st, "changes at block boundary" if not same_block else "changes within block")][
                    "changed" if changed else "kept"] += 1
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"rq2-twin-blocks: {len(d)} tasks " + ", ".join(f"{s}: {n} seeds" for s, n in d.setting.value_counts().items()))
    out = []
    for st, g in d.groupby("setting"):
        fx = g[[c for c in g.columns if c.startswith("fx_")]]
        best_avg = fx.mean().idxmin()
        total = [pct(b, o) for b, o in zip(g[best_avg], g.oracle)]
        rec = {"setting": st, "seeds": len(g), "best_avg_pair": best_avg[3:],
               "switch_med_%": g["switch_%"].median(), "switch_mean_%": g["switch_%"].mean(),
               "switch_gt2_%": 100 * (g["switch_%"] > 2).mean(),
               "total_med_%": np.nanmedian(total), "total_mean_%": np.nanmean(total),
               "saved_per_h_js": g["saved_per_h"].mean(), "best_tard_per_h": 1000 * g.own_best.mean() / (WINDOW[st] / 3600)}
        if st.startswith("blocks"):
            rec.update({"blocksel_med_%": g["blocksel_%"].median(),
                        "oracle_vs_blocksel_med_%": g["oracle_vs_blocksel_%"].median(),
                        "distinct_block_best_pairs": g["block_best_pairs"].mean()})
        out.append(rec)
    print(pd.DataFrame(out).set_index("setting").round(2).T.to_string())
    print("\nOracle job-rule shares by slot regime (and how often the chosen pair changes):")
    for (st, key), c in sorted(shares.items()):
        n = sum(c.values())
        print(f"  {st:6s} {key:28s} " + ", ".join(f"{k} {100 * v / n:.0f}%" for k, v in c.most_common()) + f"  (n={n})")
    d.drop(columns=[c for c in d.columns if c.startswith("fx_")]).to_csv(HERE / "per_seed.csv", index=False,
                                                                         float_format="%.4f")


if __name__ == "__main__":
    main()
