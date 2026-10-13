"""
@file analyze.py
@brief base-eplen: does each instance's preferred fixed rule wash out as episodes get longer?

Per (window, block), over seeds 0-39 and the 15 H15 pairs:
  - the best pair on average, and the share of instances whose own best pair is that pair
  - instance-selection headroom: mean gain of each instance's own best pair (hindsight) over the best-on-average pair;
    if the per-instance preference is sampling noise, this falls as the window grows
  - rank agreement: mean Spearman between an instance's pair ranking and the average ranking
  - tardiness per job exited (MDD-TECT), jobs exited per window
  - 6 h chunks of the long windows: the same statistics per chunk (warm floor), and how often a chunk's best pair is
    its episode's best pair
Consistency: window 6 h / block 1.5 h must reproduce rq2-twin-fleet B2's fixed pairs.
@par Usage
@code{.sh}
.venv/bin/python results/base-eplen/analyze.py
@endcode
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
B2 = HERE.parent / "rq2-twin-fleet" / "tasks"


def stats(t):
    """t: DataFrame instances x pairs of tardiness."""
    avg_best = t.mean().idxmin()
    own = t.min(axis=1)
    mean_rank = t.mean().rank()
    rho = np.nanmean([spearmanr(row, mean_rank)[0] for _, row in t.iterrows()])
    return {"best_on_avg": avg_best, "own_best_is_avg_best_%": 100 * (t.idxmin(axis=1) == avg_best).mean(),
            "hindsight_gain_mean_%": 100 * ((t[avg_best] - own) / t[avg_best].where(t[avg_best] > 0)).mean(),
            "hindsight_gain_median_%": 100 * ((t[avg_best] - own) / t[avg_best].where(t[avg_best] > 0)).median(),
            "rank_rho_mean": rho, "distinct_own_best": t.idxmin(axis=1).nunique()}


def main():
    recs, chunk_recs = [], []
    for f in sorted(HERE.glob("w*_b*/s*.json")):
        r = json.loads(f.read_text())
        recs.append(r)
    if not recs:
        print("no results yet")
        return
    check = []
    out, chunk_out = [], []
    for (m, w, b), grp in pd.DataFrame([{"w": r["window"], "b": r["block"], "m": r.get("mix", "B2"), "r": r} for r in recs]).groupby(["m", "w", "b"]):
        rs = [x for x in grp.r]
        t = pd.DataFrame({r["seed"]: {p: v["tard"] for p, v in r["fixed"].items()} for r in rs}).T
        timed = sum(any(v["timed_out"] for v in r["fixed"].values()) for r in rs)
        jobs = np.mean([r["fixed"]["MDD-TECT"]["jobs_exited"] for r in rs])
        st = stats(t)
        out.append({"mix": m, "window_h": w / 3600, "block_h": b / 3600, "seeds": len(rs), "timed_out": timed, "jobs_exited": jobs,
                    "mddtect_tard_per_job": np.mean([r["fixed"]["MDD-TECT"]["tard"] * 1000 / max(1, r["fixed"]["MDD-TECT"]["jobs_exited"]) for r in rs]),
                    **st})
        if w == 21600 and b == 5400 and m == "B2":
            for r in rs:
                task = B2 / f"B2_s{r['seed']}.json"
                if task.exists():
                    tk = json.loads(task.read_text())
                    check.append(max(abs(r["fixed"][p]["tard"] - tk["fixed"][p]["tard"]) for p in r["fixed"]))
        n_chunks = len(rs[0]["fixed"]["MDD-TECT"]["chunks"])
        if n_chunks > 1:
            rows = []
            for r in rs:
                ep_best = min(r["fixed"], key=lambda p: r["fixed"][p]["tard"])
                for k in range(n_chunks):
                    rows.append({"seed": r["seed"], "chunk": k, "ep_best": ep_best,
                                 **{p: v["chunks"][k] for p, v in r["fixed"].items()}})
            c = pd.DataFrame(rows)
            pairs = list(rs[0]["fixed"])
            trend = c.groupby("chunk")["MDD-TECT"].mean()
            print(f"stability {m} w{w / 3600:.0f}h b{b / 3600:.1f}h: MDD-TECT tardiness per 6 h chunk (mean over seeds) "
                  + " ".join(f"{v:.0f}" for v in trend.values[:: max(1, len(trend) // 12)])
                  + "  (rising = backlog grows: not stationary over this window)")
            ct = c.set_index(["seed", "chunk"])[pairs]
            cs = stats(ct[ct.min(axis=1) > 0])
            chunk_out.append({"mix": m, "window_h": w / 3600, "block_h": b / 3600, "chunks": len(ct),
                              "chunk_best_is_episode_best_%": 100 * (ct.idxmin(axis=1).values == c.ep_best.values).mean(),
                              **{f"chunk_{k}": v for k, v in cs.items()}})
    pd.set_option("display.width", 240)
    print(f"consistency (6 h / 1.5 h vs rq2-twin-fleet B2): {len(check)} seeds, max |diff| {max(check) if check else float('nan'):.2e}\n")
    print("== whole episodes: does the per-instance best pair wash out with length?")
    print(pd.DataFrame(out).round(3).to_string(index=False))
    if chunk_out:
        print("\n== 6 h chunks inside the long episodes (warm floor): per-chunk preference")
        print(pd.DataFrame(chunk_out).round(3).to_string(index=False))
    print("\n(hindsight gain = picking each instance's own best fixed pair after the fact vs the best-on-average pair; "
          "rank rho = agreement of an instance's pair ranking with the average ranking)")


if __name__ == "__main__":
    main()
