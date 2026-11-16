"""
@file analyze.py
@brief rq3-alpha-transport (2026-10-28): stage 1's pre-declared fit of the alpha law's size / transport term
       (README.md has the design and the gate, written before any result).
  1. alpha curves of the 14 fit cells (fitting seeds 2000-2039): pooled window tardiness of MDD-PTECT per alpha, % vs
     alpha 3, alpha* (grid argmin) and the alphas within 1% of the best; the cells' floor statistics.
  2. Candidate laws (homogeneous floors, so the v7 spread factor max(0, 3 - 6 sigma) is 3):
       L0  alpha 3 everywhere (the v7 law, no size term);
       L1-L3  alpha = 3 h(x), h(x) = clip((x1 - x) / (x1 - x0), 0, 1), x0 >= x of B2 (so the law is v7's on B2), with
         L1 x = tau = mean travel time to a candidate machine / mean op time (floor geometry, AGV speed, job mix);
         L2 x = tau / a, a = AGVs on duty per machine (full fleet): travel per op per AGV;
         L3 x = AGV utilization under MDD-PTECT at alpha 3 (a measured fleet KPI).
     x is a floor property: the mean over the cell's fitting instances. A cell's regret at alpha is the % its pooled
     tardiness at alpha (linear interpolation between the two neighbouring grid alphas) exceeds the pooled tardiness
     at its best grid alpha. The fit minimizes the mean regret over cells (equal weight) by grid search over (x0,
     x1 - x0); ties within 0.01 points go to the gentlest slope, then the median knot.
  3. Leave-one-cell-out regret; selection: the lowest, then the simplest within 0.5 points (L1 < L2 < L3).
  4. Gate F1: the selected law's leave-one-cell-out regret is at least 1.0 point below L0's. law.json carries the
     selected law, its alpha for every cell (fit, held-out, stage-2 test) and F1, for ../eval-alpha-transport-size.
Outputs: analysis.out (stdout), alpha_curves.csv, cells.csv, fit.json, law.json.
@par Usage (from the repo root)
@code{.sh}
.venv/bin/python results/rq3-alpha-transport/analyze.py [--root DIR]
@endcode
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parent
A = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
FIT_SEEDS = list(range(2000, 2040))
## @brief Fit cells; m15-m100 at r052 take their curves from ../rq3-alpha-size (cell m<M>_s0) and their AGV
##        utilization from this experiment's util runs.
REUSED = {"m15_b2": "m15_s0", "m20_r052": "m20_s0", "m30_r052": "m30_s0", "m50_r052": "m50_s0",
          "m75_r052": "m75_s0", "m100_r052": "m100_s0"}
FIT_CELLS = list(REUSED) + ["m150_r052", "m200_r052", "m50_r035", "m50_r080", "m100_r035", "m100_r080",
                            "m200_r035", "m200_r080"]
HELD_OUT = ["m75_r035", "m150_r080"]
B2 = "m15_b2"
LAWS = {"L1": ("tau", "travel / op time"), "L2": ("tau_per_agv", "travel / op time / AGVs per machine"),
        "L3": ("agv_util", "AGV utilization under MDD-PTECT")}
TIE = 0.01
SIMPLER_WITHIN = 0.5
F1_MARGIN = 1.0


def read(root, cell, seeds):
    """{seed: json} of a cell's runs (this experiment's out/, or ../rq3-alpha-size's for the reused curves)."""
    out = {}
    for s in seeds:
        f = Path(root) / "out" / cell / f"s{s}.json"
        if f.exists():
            out[s] = json.loads(f.read_text())
    return out


def curve(root, cell, seeds):
    """(seeds, instances x alphas tardiness) of MDD-PTECT over the alpha grid."""
    if cell in REUSED:
        runs = {s: json.loads((R / "rq3-alpha-size" / "out" / REUSED[cell] / f"s{s}.json").read_text())
                for s in seeds if (R / "rq3-alpha-size" / "out" / REUSED[cell] / f"s{s}.json").exists()}
    else:
        runs = read(root, cell, seeds)
    ss = sorted(runs)
    return ss, np.array([[runs[s][f"MDD-PTECT@{a:g}"]["total"] for a in A] for s in ss])


def floor_stats(root, cell):
    """The cell's floor statistics, means over its fitting instances, and the instance count."""
    runs = read(root, cell, FIT_SEEDS)
    st = [r["_stats"] for r in runs.values()]
    tau = float(np.mean([s["travel_mean_s"] / s["p_mean_s"] for s in st]))
    a = float(np.mean([s["agvs_per_machine"] for s in st]))
    u = [s["agv_util"] for s in st if s.get("agv_util") is not None]
    return {"n": len(st), "machines": st[0]["machines"] if st else None, "tau": tau, "agvs_per_machine": a,
            "tau_per_agv": tau / a, "agv_util": float(np.mean(u)) if u else float("nan"),
            "travel_s": float(np.mean([s["travel_mean_s"] for s in st])),
            "p_s": float(np.mean([s["p_mean_s"] for s in st]))}


def hinge(x, x0, d):
    return np.clip((x0 + d - x) / d, 0.0, 1.0)


def regrets(P, alpha):
    """% regret of a pooled curve P (over A) at alpha (array, interpolated)."""
    return 100.0 * (np.interp(alpha, A, P) / P.min() - 1.0)


def fit(cells, x, xb2):
    """Grid search of (x0, d) minimizing the mean regret over `cells` ({cell: pooled curve}); returns x0, d, regret."""
    names = list(cells)
    X = np.array([x[c] for c in names])
    lo, hi = xb2, max(X.max(), xb2) + 1e-9
    span = hi - lo
    x0s = np.linspace(lo, hi, 241)
    ds = np.linspace(span / 200.0, 5.0 * span, 400)
    H = hinge(X[None, None, :], x0s[:, None, None], ds[None, :, None])          # (x0, d, cell)
    reg = np.zeros(H.shape[:2])
    for k, c in enumerate(names):
        reg += regrets(cells[c], 3.0 * H[:, :, k])
    reg /= len(names)
    best = reg.min()
    i0, i1 = np.nonzero(reg <= best + TIE)
    dmax = ds[i1].max()
    cand = sorted(x0s[i0[ds[i1] == dmax]])
    x0 = cand[len(cand) // 2]
    return float(x0), float(dmax), float(reg[np.argmin(np.abs(x0s - x0)), np.argmin(np.abs(ds - dmax))])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE), help="experiment folder holding out/ (dry runs)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    print("== rq3-alpha-transport stage 1: the alpha law's size / transport term (MDD-PTECT, homogeneous floors)")
    print("   fitting seeds 2000-2039; pooled window tardiness; regret = % over the cell's best grid alpha")

    # 0. checks and floor statistics
    print("\n-- 0. checks")
    pooled, stats_of, ok = {}, {}, True
    for c in FIT_CELLS + HELD_OUT:
        st = floor_stats(root, c)
        stats_of[c] = st
        line = f"   {c:10s} statistics from {st['n']} fitting instances"
        if c in FIT_CELLS:
            ss, T = curve(root, c, FIT_SEEDS)
            pooled[c] = T.sum(axis=0)
            line += f"; curve on {len(ss)} fitting seeds"
            ok &= len(ss) == 40
            if c in REUSED:                                   # the util rerun reproduces the reused alpha-3 value
                runs = read(root, c, FIT_SEEDS)
                same = sum(abs(runs[s]["MDD-PTECT@3"]["total"] - T[ss.index(s), list(A).index(3.0)]) < 1e-9
                           for s in runs if s in ss)
                line += f"; util rerun = ../rq3-alpha-size at alpha 3 on {same} / {len(runs)}"
                ok &= same == len(runs) == 40
        ok &= st["n"] == 40
        print(line)
    print(f"   all complete and reproduced: {ok}")

    # 1. curves
    print("\n-- 1. alpha curves (fitting seeds), % vs alpha 3; floor statistics (means over fitting instances)")
    print("   cell        tau     a     tau/a  util  | alpha*  within 1%      | " +
          " ".join(f"{x:>6g}" for x in A))
    rows = []
    for c in FIT_CELLS:
        P = pooled[c]
        pct = 100.0 * (P / P[list(A).index(3.0)] - 1.0)
        star = A[int(np.argmin(P))]
        within = A[P <= P.min() * 1.01]
        st = stats_of[c]
        print(f"   {c:10s} {st['tau']:.4f} {st['agvs_per_machine']:.3f} {st['tau_per_agv']:.4f} "
              f"{st['agv_util']:.3f} | {star:5g}  {within.min():g}-{within.max():g}{'':9s} | " +
              " ".join(f"{v:+6.1f}" for v in pct))
        rows += [{"cell": c, "alpha": float(x), "pct_vs_alpha3": float(v), "pooled": float(p)}
                 for x, v, p in zip(A, pct, P)]
    for c in HELD_OUT:
        st = stats_of[c]
        print(f"   {c:10s} {st['tau']:.4f} {st['agvs_per_machine']:.3f} {st['tau_per_agv']:.4f} "
              f"{st['agv_util']:.3f} | held out (curve on test seeds only, stage 2)")
    with open(root / "alpha_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(root / "cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell"] + list(stats_of[FIT_CELLS[0]]))
        w.writeheader()
        w.writerows([{"cell": c, **stats_of[c]} for c in FIT_CELLS + HELD_OUT])

    # 2-3. fits and leave-one-cell-out
    print("\n-- 2. candidate laws (alpha = 3 h(x), h = clip((x1 - x) / (x1 - x0), 0, 1), x0 >= B2's x)")
    l0 = float(np.mean([regrets(pooled[c], 3.0) for c in FIT_CELLS]))
    fits = {"L0": {"variable": None, "regret": l0, "loo": l0, "per_cell": {c: float(regrets(pooled[c], 3.0))
                                                                            for c in FIT_CELLS}}}
    print(f"   L0 alpha 3 everywhere: mean regret {l0:.2f}%, leave-one-cell-out {l0:.2f}%")
    for name, (var, label) in LAWS.items():
        x = {c: stats_of[c][var] for c in FIT_CELLS + HELD_OUT}
        x0, d, reg = fit({c: pooled[c] for c in FIT_CELLS}, x, x[B2])
        loo = {}
        for c in FIT_CELLS:
            lx0, ld, _ = fit({k: pooled[k] for k in FIT_CELLS if k != c}, x, x[B2])
            loo[c] = float(regrets(pooled[c], 3.0 * hinge(x[c], lx0, ld)))
        per = {c: float(regrets(pooled[c], 3.0 * hinge(x[c], x0, d))) for c in FIT_CELLS}
        fits[name] = {"variable": var, "label": label, "x0": x0, "x1": x0 + d, "regret": reg,
                      "loo": float(np.mean(list(loo.values()))), "per_cell": per, "loo_per_cell": loo}
        print(f"   {name} x = {label}: x0 {x0:.4f}, x1 {x0 + d:.4f}; mean regret {reg:.2f}%, "
              f"leave-one-cell-out {fits[name]['loo']:.2f}%")
    print("\n   per cell: regret % (in sample / leave-one-out) and the law's alpha")
    print("   cell        L0     " + "  ".join(f"{n:>20s}" for n in LAWS))
    for c in FIT_CELLS:
        parts = []
        for n, (var, _) in LAWS.items():
            f_ = fits[n]
            al = 3.0 * hinge(stats_of[c][var], f_["x0"], f_["x1"] - f_["x0"])
            parts.append(f"{f_['per_cell'][c]:5.2f} / {f_['loo_per_cell'][c]:5.2f} a{al:4.2f}")
        print(f"   {c:10s} {fits['L0']['per_cell'][c]:5.2f}  " + "  ".join(f"{p:>20s}" for p in parts))

    best = min(LAWS, key=lambda n: fits[n]["loo"])
    sel = next(n for n in LAWS if fits[n]["loo"] <= fits[best]["loo"] + SIMPLER_WITHIN)
    f1 = fits[sel]["loo"] <= l0 - F1_MARGIN
    print(f"\n-- 3. selection: lowest leave-one-cell-out {best} ({fits[best]['loo']:.2f}%); simplest within "
          f"{SIMPLER_WITHIN} points: {sel}")
    print(f"-- 4. gate F1, selected law's leave-one-cell-out regret <= L0's - {F1_MARGIN}: "
          f"{fits[sel]['loo']:.2f} vs {l0:.2f} -> {'PASS' if f1 else 'FAIL'}")

    var = LAWS[sel][0]
    x0, x1 = fits[sel]["x0"], fits[sel]["x1"]
    cells = {}
    for c in FIT_CELLS + HELD_OUT:
        h = float(hinge(stats_of[c][var], x0, x1 - x0))
        cells[c] = {"x": round(stats_of[c][var], 6), "h": h, "alpha": round(3.0 * h, 3),
                    "best_alpha_fit": float(A[int(np.argmin(pooled[c]))]) if c in pooled else None}
    assert cells[B2]["h"] == 1.0, "the law must be v7's on B2"
    print(f"\n   selected law {sel}: alpha = max(0, 3 - 6 sigma) x clip(({x1:.4f} - x) / ({x1:.4f} - {x0:.4f}), 0, 1), "
          f"x = {LAWS[sel][1]}")
    print("   cell        x        h      alpha   best alpha (fitting seeds)")
    for c, v in cells.items():
        print(f"   {c:10s} {v['x']:.4f}  {v['h']:.3f}  {v['alpha']:5.3f}   {v['best_alpha_fit']}")
    (root / "fit.json").write_text(json.dumps(fits, indent=1))
    (root / "law.json").write_text(json.dumps({
        "law": sel, "variable": var, "label": LAWS[sel][1], "x0": x0, "x1": x1,
        "form": "alpha = min(4, max(0, 3 - 6 sigma) x h(x) x offset), h(x) = clip((x1 - x) / (x1 - x0), 0, 1)",
        "F1": "PASS" if f1 else "FAIL", "loo": {n: fits[n]["loo"] for n in fits}, "cells": cells}, indent=1))


if __name__ == "__main__":
    main()
