"""
@file analyze.py
@brief agv-ratio-twin-v5 (2026-10-14): knees and the fleet-size fit on the v5 rules (twin, free flow). Per (M, util,
       op, seed, rule): knee5 / knee2 on mean time in system (the smallest fleet within 5% / 2% of the best fleet,
       unfinished jobs included; runs go to empty) and knee5 on total tardiness; per cell the mean over seeds and
       rules (as agv-ratio-twin); the log-linear fit knee5 = a * M^b * (util / 0.85)^c * (op / 360 s)^d; and the
       comparison with agv-ratio-twin's SRT_TECT / SRT_ECT knees. Writes analysis.out and knees.csv.
@par Usage
@code{.sh}
.venv/bin/python results/agv-ratio-twin-v5/analyze.py
@endcode
"""
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "agv-ratio-twin"


def load(folder):
    fr = []
    for f in sorted(folder.glob("m*_u*_op*.csv")):
        m = re.match(r"m(\d+)_u([\d.]+)_op(\d+)", f.stem)
        x = pd.read_csv(f)
        x["util"], x["op"] = float(m.group(2)), int(m.group(3))
        fr.append(x)
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def knees(d, metric_tis="mean_tis"):
    out = []
    for (m, u, op, seed, rule), g in d.groupby(["m", "util", "op", "seed", "rule"]):
        g = g.sort_values("agv_count").drop_duplicates("agv_count", keep="last").set_index("agv_count")
        row = {"m": m, "util": u, "op": op, "seed": seed, "rule": rule, "fleets": len(g),
               "best_tis": g[metric_tis].min(), "largest": int(g.index.max())}
        for tol, k in ((0.05, "knee5"), (0.02, "knee2")):
            row[k] = int(g.index[g[metric_tis] <= (1 + tol) * g[metric_tis].min()].min())
        if "tardiness_sum" in g:
            t = g.tardiness_sum
            row["knee5_tard"] = int(g.index[t <= 1.05 * t.min()].min())
        row["knee_at_largest"] = row["knee5"] == row["largest"]
        out.append(row)
    return pd.DataFrame(out)


def fit(k):
    c = k.groupby(["m", "util", "op"]).knee5.mean().reset_index()
    X = np.column_stack([np.ones(len(c)), np.log(c.m), np.log(c.util / 0.85), np.log(c.op / 360.0)])
    beta, *_ = np.linalg.lstsq(X, np.log(c.knee5), rcond=None)
    pred = np.exp(X @ beta)
    err = np.abs(pred / c.knee5 - 1)
    return beta, float(np.median(err)), float(err.max()), c.assign(pred=pred.round(1))


def main():
    pd.set_option("display.width", 220)
    d = load(HERE)
    if d.empty:
        print("no runs yet")
        return
    print(f"== agv-ratio-twin-v5: {len(d)} twin runs (MDD-PTECT / MDD-PECT, due dates c ~ U[1.75, 2.5])")
    k = knees(d)
    k.to_csv(HERE / "knees.csv", index=False)
    cell = k.groupby(["m", "util", "op", "rule"]).agg(knee5=("knee5", "mean"), knee2=("knee2", "mean"),
                                                      knee5_tard=("knee5_tard", "mean"),
                                                      at_largest=("knee_at_largest", "sum")).reset_index()
    print("\n== knees per cell and rule (mean over 3 seeds); at_largest = seeds whose knee is the largest fleet run")
    print(cell.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    beta, med, mx, c = fit(k)
    print(f"\n== fit over the {len(c)} cells (seeds and both rules pooled, as agv-ratio-twin): "
          f"knee5 ~ {np.exp(beta[0]):.3f} * M^{beta[1]:.2f} * (util/0.85)^{beta[2]:.2f} * (op/360 s)^{beta[3]:.2f}; "
          f"median error {100 * med:.0f}%, max {100 * mx:.0f}%")
    print("   agv-ratio-twin (SRT_TECT / SRT_ECT, no due dates): 0.087 * M^1.29 * (util/0.85)^0.53 * (op/360 s)^-0.98")
    for rule in ("MDD_PTECT", "MDD_PECT"):
        b, md, mxx, _ = fit(k[k.rule == rule])
        print(f"   {rule} alone: {np.exp(b[0]):.3f} * M^{b[1]:.2f} * (util/0.85)^{b[2]:.2f} * (op/360 s)^{b[3]:.2f}; "
              f"median error {100 * md:.0f}%")
    old = load(OLD)
    if len(old):
        ko = knees(old).groupby(["m", "util", "op"]).knee5.mean().rename("knee5_old (SRT rules)")
        kn = k.groupby(["m", "util", "op"]).knee5.mean().rename("knee5_v5")
        both = pd.concat([ko, kn], axis=1).dropna()
        both["v5_vs_old_%"] = 100 * (both["knee5_v5"] / both["knee5_old (SRT rules)"] - 1)
        print("\n== knee5 on the v5 rules vs agv-ratio-twin's SRT rules (same grid; instances differ by the due dates)")
        print(both.round(1).to_string())
    print("\n== MDD-PECT vs MDD-PTECT total tardiness at each cell's knee5 fleet (MDD-PTECT's), mean over seeds, %")
    t = d.pivot_table(index=["m", "util", "op", "seed", "agv_count"], columns="rule", values="tardiness_sum")
    kp = k[k.rule == "MDD_PTECT"].set_index(["m", "util", "op", "seed"]).knee5
    rows = []
    for key, n in kp.items():
        if (*key, n) in t.index:
            r = t.loc[(*key, n)]
            rows.append({"m": key[0], "util": key[1], "op": key[2], "seed": key[3], "knee": n,
                         "PECT_vs_PTECT_%": 100 * (r.MDD_PECT / r.MDD_PTECT - 1)})
    if rows:
        print(pd.DataFrame(rows).groupby(["m", "util", "op"])["PECT_vs_PTECT_%"].mean().round(1).to_string())


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
