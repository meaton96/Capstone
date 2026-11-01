"""
@file analyze.py
@brief rq3-size-m1000 (2026-10-20): the reduced 1000-machine column of the floor-size grid. 5 test seeds, so per policy
       vs MDD-PTECT: pooled % over the 5 seeds, each seed's %, wins / losses / identical (|diff| < 1e-6 k job-s), and
       an exact sign test over the non-identical seeds (no CI: 5 seeds). Identical = relative difference <= 1e-6
       (the simulator's rule runs and evaluate.py's policy runs differ by float32 rounding, ~1e-8). Next to it, the same pick's pooled % at
       200 machines in ../rq3-size-grid (seeds 0-39) for the trend. Reproduction checks against ../dev-size-1000-probe
       (local evaluate.py), seed 0: MDD-PTECT from the oracle simulator (mdd / srt run there since 10-20) and the final
       policy from the cluster's evaluate.py; match = relative difference <= 1e-6 (float32 metrics).
       -> analysis.out, summary.csv
"""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
R = HERE.parent
H = {"config_hash": str, "instance_hash": str}
LABEL = {"mdd": "MDD-PTECT", "srt": "SRT-PTECT", "tect": "MDD-TECT", "t15": "T15 s3 (final policy)", "t50": "T50", "t100": "T100",
         "tmix": "Tmix"}
GRID_ARM = {"t15": "T15", "t50": "T50", "t100": "T100", "tmix": "Tmix"}
SIM_GROUPS = {"mdd", "srt", "tect"}              # run in the oracle simulator since 10-20 (sim_rules.py)


def load():
    rows = []
    for f in sorted((HERE / "m1000").glob("*_s*/episodes.csv")):
        if not (f.parent / "DONE").exists():
            continue
        g, s = f.parent.name.rsplit("_s", 1)
        e = pd.read_csv(f, dtype=H)
        rows.append({"group": g, "seed": int(s), "tard": float(e.window_tardiness.iloc[0]) / 1000.0})
    return pd.DataFrame(rows).pivot_table(index="seed", columns="group", values="tard") if rows else pd.DataFrame()


def main():
    print("== rq3-size-m1000: 1000 machines, 520 AGVs on duty, twin, test seeds 0-4, window tardiness; < 0 better")
    t = load()
    if t.empty or "mdd" not in t.columns:
        print("MDD-PTECT not run yet")
        return
    picks = json.loads((HERE / "picks.json").read_text()) if (HERE / "picks.json").exists() else {}
    picks["t15"] = 3
    gs = R / "rq3-size-grid" / "summary.csv"
    grid = pd.read_csv(gs) if gs.exists() else pd.DataFrame()
    print(f"MDD-PTECT: {t.mdd.notna().sum()} / 5 seeds, mean {t.mdd.mean():,.1f} k job-s")
    probe = R / "dev-size-1000-probe" / "m1000"
    for g, pg in (("mdd", "ref"), ("t15", "s3")):
        f = probe / pg / "episodes.csv"
        if f.exists() and g in t.columns and 0 in t.index and not np.isnan(t.loc[0, g]):
            v = float(pd.read_csv(f, dtype=H).window_tardiness.iloc[0]) / 1000.0
            where = "oracle simulator" if g in SIM_GROUPS else "cluster evaluate.py"
            ok = abs(v - t.loc[0, g]) <= 1e-6 * abs(v) + 1e-4       # float32 rounding of evaluate.py's metrics
            print(f"   reproduction, seed 0 {LABEL[g]}: {where} {t.loc[0, g]:,.4f} vs local probe (evaluate.py) "
                  f"{v:,.4f} k job-s, relative difference {abs(v - t.loc[0, g]) / abs(v):.1e}"
                  + ("  (match)" if ok else "  ** DIFFERS **"))
    out = []
    for g in ("srt", "tect", "t15", "t50", "t100", "tmix"):
        if g not in t.columns:
            continue
        both = t[["mdd", g]].dropna()
        if both.empty:
            continue
        d = both[g] - both.mdd
        per = 100 * (both[g] / both.mdd - 1)
        tol = 1e-6 * both.mdd.abs() + 1e-4                # relative: the simulator and evaluate.py differ by float32 rounding
        w, l, same = int((d < -tol).sum()), int((d > tol).sum()), int((d.abs() <= tol).sum())
        p = binomtest(w, w + l).pvalue if w + l else 1.0
        pooled = 100 * (both[g].sum() / both.mdd.sum() - 1)
        lab = LABEL[g] + (f" s{picks[g]}" if g in ("t50", "t100", "tmix") and g in picks else "")
        at200 = None
        if g in GRID_ARM and not grid.empty and g in picks:
            r = grid[(grid.floor == "m200") & (grid.vs == "MDD-PTECT") & (grid.policy == f"{GRID_ARM[g]} s{picks[g]}")]
            at200 = float(r["pooled_%"].iloc[0]) if len(r) else None
        print(f"   {lab:24s} {pooled:+6.2f}% over {len(both)} seeds | per seed " + " ".join(f"{v:+.2f}" for v in per)
              + f" | W/L/same {w}/{l}/{same}, sign test p={p:.2g}"
              + (f" | at 200 machines {at200:+.2f}%" if at200 is not None else ""))
        out.append({"group": g, "policy": lab, "n_seeds": len(both), "pooled_%": pooled, "wins": w, "losses": l,
                    "identical": same, "sign_p": p, "m200_pooled_%": at200,
                    **{f"s{s}_%": v for s, v in per.items()}})
    pd.DataFrame(out).to_csv(HERE / "summary.csv", index=False)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
