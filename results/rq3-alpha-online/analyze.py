"""
@file analyze.py
@brief rq3-alpha-online stage 1 analysis (pre-declared 10-29, README "Analysis"): the per-regime alpha surfaces, the
       candidate online laws, leave-one-cell-out selection, gate G1, and law_online.json for stage 2
       (../eval-alpha-online).
Surfaces: per cell, window tardiness of MDD-PTECT pooled over its seeds at every (alpha_full, alpha_short) of the 7 x 7
grid (fit cells: fitting seeds 2000-2039; held-out cells: test seeds 0-39). Regret of a cell at a pair = % its pooled
tardiness there exceeds that at its best grid pair; off-grid pairs are interpolated bilinearly.
Laws (homogeneous fit cells, so the spread factor is 3): alpha = 3 h(x), h(x) = clip((x1 - x) / (x1 - x0), 0, 1),
x = tau / a, tau the cell's mean travel / op time, a AGVs per machine:
  S0   static, ../rq3-alpha-transport's knots: a = the full fleet in every block (today's law);
  S    static, knots refit here on the surfaces (the fair static reference);
  O1   online, S0's knots: a = the AGVs on duty, so short-fleet blocks read their own x;
  O2   online, one hinge refit;
  O3   online, separate hinges for full-fleet and short-fleet blocks (4 knots);
  M    comparator: S0 in full-fleet blocks, alpha 0 (MDD-TECT) in short-fleet blocks (../rq3-size-portfolio's map rule).
Fit: grid search minimizing the mean regret over the fit cells, x0 >= B2's full-fleet x (so B2's full-fleet blocks
keep v7's alpha 3) for the full hinge; ties within 0.01 points go to the gentlest slope, then the median knot.
Selection: leave-one-cell-out regret; the simplest within 0.5 points of the lowest, O1 < O2 < O3.
Gate G1: the selected law's leave-one-cell-out regret is at least 1.0 point below S's.
@par Usage (from the repo root)
@code{.sh}
.venv/bin/python results/rq3-alpha-online/analyze.py                       # real outputs
.venv/bin/python results/rq3-alpha-online/analyze.py --synthetic O3 --out-dir <scratch>   # pipeline test
@endcode
"""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parent
_spec = importlib.util.spec_from_file_location("ao_run", HERE / "run.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
A = np.array(RUN.ALPHAS)
FIT, HELD_OUT = RUN.FIT, RUN.HELD_OUT
LAW0 = json.loads((R / "rq3-alpha-transport" / "law.json").read_text())
X0_S0, X1_S0 = LAW0["x0"], LAW0["x1"]
## @brief Stage 2's policy cells (../eval-alpha-online): B2, the 15-machine spread-1 floor (the het arm's home) and
##        ../eval-alpha-transport-size's six size / fleet cells. The held-out cells get law runs only.
STAGE2_CELLS = ["m15_b2", "m15_s1", "m50_r052", "m75_r052", "m100_r052", "m100_r035", "m100_r080", "m200_r052"]
TIE = 0.01
SIMPLER = 0.5
G1_MARGIN = 1.0
KNOT_STEP = 0.01
KNOT_STEP_O3 = 0.02


# ── data ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def cell_seeds(cell):
    return RUN.FIT_SEEDS if cell in FIT else RUN.TEST_SEEDS


def load_surface(cell, root):
    """(pooled 7x7 tardiness, per-seed array, seeds, per-regime pooled 7x7 {regime: array}) of a cell."""
    per, seeds, by_regime = [], [], {}
    for s in cell_seeds(cell):
        f = Path(root) / "out" / cell / f"s{s}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if d.get("_kind") != "surf":
            continue
        g = np.array([[d[f"{af:g}|{ash:g}"]["total"] for ash in RUN.ALPHAS] for af in RUN.ALPHAS])
        per.append(g)
        seeds.append(s)
        prof = {str(b["block"]): b["load_profile"] for b in d["_blocks"]}
        for reg in ("normal", "surge", "short"):
            r = np.array([[sum(v for k, v in d[f"{af:g}|{ash:g}"]["blocks"].items() if prof.get(k) == reg)
                           for ash in RUN.ALPHAS] for af in RUN.ALPHAS])
            by_regime[reg] = by_regime.get(reg, 0) + r
    if not per:
        return None
    per = np.array(per)
    return per.sum(axis=0), per, seeds, by_regime


def diag_reference(cell):
    """Folder of the earlier fixed-alpha runs of a cell's instances (../rq3-alpha-size for B2, the grid at 15-100
    machines and the spread cells; ../rq3-alpha-transport for 150-200 machines and the r035 / r080 fleets)."""
    m, f = cell.split("_", 1)
    if f in ("b2", "r052") and int(m[1:]) <= 100:
        return R / "rq3-alpha-size" / "out" / f"{m}_s0"
    if f.startswith("s"):
        return R / "rq3-alpha-size" / "out" / cell
    return R / "rq3-alpha-transport" / "out" / cell


def load_stats(cell, root):
    """The cell's floor statistics from its fitting instances (surf outputs for fit cells, stats outputs for held-out
    cells): mean tau, machines, AGVs per machine full / short, mean sigma."""
    rows = []
    for s in RUN.FIT_SEEDS:
        f = Path(root) / "out" / cell / f"s{s}.json"
        if f.exists():
            rows.append(json.loads(f.read_text())["_stats"])
    if not rows:
        return None
    a_full = float(np.median([r["agvs_per_machine"] for r in rows]))
    shorts = [r["agvs_per_machine_short"] for r in rows if r["agvs_per_machine_short"] < r["agvs_per_machine"] - 1e-9]
    return {"n": len(rows), "machines": rows[0]["machines"], "tau": float(np.mean([r["tau"] for r in rows])),
            "a_full": a_full,                                   # the fleet rule's counts (equal in every instance
            "a_short": float(np.median(shorts)) if shorts else a_full,   # that has a short block, ~80% of them)
            "n_short": len(shorts), "sigma": float(np.mean([r["sigma"] for r in rows]))}


# ── laws ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def h(x, x0, x1):
    return np.clip((x1 - x) / (x1 - x0), 0.0, 1.0)


def spread_factor(sigma):
    return max(0.0, 3.0 - 6.0 * sigma)


def law_pair(form, knots, st):
    """(alpha_full, alpha_short) of a law on a cell (the cell's mean sigma for spread cells)."""
    sf = spread_factor(st["sigma"])
    xf, xs = st["tau"] / st["a_full"], st["tau"] / st["a_short"]
    if form in ("S0", "S"):
        a = min(4.0, sf * float(h(xf, *knots[0])))
        return a, a
    if form == "M":
        return min(4.0, sf * float(h(xf, *knots[0]))), 0.0
    if form in ("O1", "O2"):
        return min(4.0, sf * float(h(xf, *knots[0]))), min(4.0, sf * float(h(xs, *knots[0])))
    if form == "O3":
        return min(4.0, sf * float(h(xf, *knots[0]))), min(4.0, sf * float(h(xs, *knots[1])))
    raise ValueError(form)


def interp2(T, af, ash):
    """Bilinear interpolation of the 7x7 grid T at (af, ash) (arrays broadcast)."""
    af, ash = np.clip(af, 0, A[-1]), np.clip(ash, 0, A[-1])
    i = np.clip(np.searchsorted(A, af, side="right") - 1, 0, len(A) - 2)
    j = np.clip(np.searchsorted(A, ash, side="right") - 1, 0, len(A) - 2)
    u = (af - A[i]) / (A[i + 1] - A[i])
    v = (ash - A[j]) / (A[j + 1] - A[j])
    return ((1 - u) * (1 - v) * T[i, j] + u * (1 - v) * T[i + 1, j] + (1 - u) * v * T[i, j + 1]
            + u * v * T[i + 1, j + 1])


def regret(T, af, ash):
    return 100.0 * (interp2(T, af, ash) / T.min() - 1.0)


def knot_grid(step, lo):
    x0 = np.round(np.arange(lo, 1.0 + 1e-9, step), 4)
    w = np.round(np.arange(step, 1.0 + 1e-9, step), 4)
    X0, W = np.meshgrid(x0, w, indexing="ij")
    return X0.ravel(), (X0 + W).ravel()


def pick(mean_reg, widths, n_knots):
    """Index of the best candidate: lowest mean regret, ties within TIE to the gentlest slope, then the median."""
    best = mean_reg.min()
    idx = np.flatnonzero(mean_reg <= best + TIE)
    wmax = widths[idx].max()
    idx = idx[widths[idx] >= wmax - 1e-9]
    return int(idx[len(idx) // 2])


class Candidates:
    """Every knot candidate of a form, scored once per cell: regs[c] = regret of cell c at each candidate (flattened),
    widths = total hinge width (the tie-break). Fits and leave-one-out refits are argmins over subsets of cells."""

    def __init__(self, form, cells, S, surf, x0_lo):
        self.form, self.cells = form, list(cells)
        if form in ("S", "O2"):
            X0, X1 = knot_grid(KNOT_STEP, x0_lo)
            self.knots = [[(float(a), float(b))] for a, b in zip(X0, X1)]
            self.widths = X1 - X0
            self.regs = {}
            for c in cells:
                st, T = S[c], surf[c]
                sf = spread_factor(st["sigma"])
                af = np.minimum(4.0, sf * h(st["tau"] / st["a_full"], X0, X1))
                ash = af if form == "S" else np.minimum(4.0, sf * h(st["tau"] / st["a_short"], X0, X1))
                self.regs[c] = regret(T, af, ash).astype(np.float32)
        else:                                                   # O3: joint full x short hinges, coarser grid
            X0f, X1f = knot_grid(KNOT_STEP_O3, x0_lo)
            X0s, X1s = knot_grid(KNOT_STEP_O3, 0.0)
            self.grid = (X0f, X1f, X0s, X1s)
            self.knots = None
            self.widths = ((X1f - X0f)[:, None] + (X1s - X0s)[None, :]).ravel()
            self.regs = {}
            for c in cells:
                st, T = S[c], surf[c]
                sf = spread_factor(st["sigma"])
                af = np.minimum(4.0, sf * h(st["tau"] / st["a_full"], X0f, X1f))
                ash = np.minimum(4.0, sf * h(st["tau"] / st["a_short"], X0s, X1s))
                self.regs[c] = regret(T, af[:, None], ash[None, :]).astype(np.float32).ravel()
        self.total = sum(self.regs[c].astype(np.float64) for c in cells)

    def knot(self, k):
        if self.knots is not None:
            return self.knots[k]
        X0f, X1f, X0s, X1s = self.grid
        i, j = divmod(k, len(X0s))
        return [(float(X0f[i]), float(X1f[i])), (float(X0s[j]), float(X1s[j]))]

    def best(self, without=None):
        tot = self.total if without is None else self.total - self.regs[without]
        n = len(self.cells) - (without is not None)
        return pick(tot / n, self.widths, None)


def fit(form, cells, S, surf, x0_lo):
    """(knots, in-sample regret per cell, leave-one-out regret per cell) of a form on the fit cells; forms without
    fitted knots score in sample."""
    if form in ("S0", "O1", "M"):
        knots = [(X0_S0, X1_S0)]
        regs = np.array([regret(surf[c], *law_pair(form, knots, S[c])) for c in cells])
        return knots, regs, dict(zip(cells, map(float, regs)))
    cand = Candidates(form, cells, S, surf, x0_lo)
    k = cand.best()
    knots = cand.knot(k)
    regs = np.array([float(cand.regs[c][k]) for c in cells])
    lo = {}
    for c in cells:
        lo[c] = float(regret(surf[c], *law_pair(form, cand.knot(cand.best(without=c)), S[c])))
    return knots, regs, lo


# ── synthetic pipeline test ──────────────────────────────────────────────────────────────────────────────────────

def synthetic(S, planted):
    """Surfaces with a planted law: T = 100 (1 + 0.05 (af - af*)^2 + 0.05 (ash - ash*)^2) plus a 2% bump, (af*, ash*)
    the planted law's pair on the cell."""
    out = {}
    form, knots = planted
    for c, st in S.items():
        af0, as0 = law_pair(form, knots, st)
        g = np.array([[100.0 * (1 + 0.05 * (af - af0) ** 2 + 0.05 * (ash - as0) ** 2) for ash in A] for af in A])
        out[c] = g
    return out


def fake_stats():
    """Cell statistics without outputs (pipeline test): stage 1 cells.csv for tau / a, the fleet rule for a_short,
    ../rq3-alpha-size for the spread cells."""
    rows = {r["cell"]: r for r in csv.DictReader(open(R / "rq3-alpha-transport" / "cells.csv"))}
    S = {}
    for c in FIT + HELD_OUT:
        if c in rows:
            r = rows[c]
            m = int(r["machines"])
            on = round(float(r["agvs_per_machine"]) * m)
            short = 2 if c == "m15_b2" else round(on / 2)
            S[c] = {"n": 0, "machines": m, "tau": float(r["tau"]), "a_full": float(r["agvs_per_machine"]),
                    "a_short": short / m, "sigma": 0.0}
        else:
            base = {"m15": "m15_b2", "m50": "m50_r052", "m100": "m100_r052"}[c.split("_")[0]]
            b = rows[base]
            m = int(b["machines"])
            on = round(float(b["agvs_per_machine"]) * m)
            short = 2 if base == "m15_b2" else round(on / 2)
            S[c] = {"n": 0, "machines": m, "tau": float(b["tau"]), "a_full": float(b["agvs_per_machine"]),
                    "a_short": short / m, "sigma": {"0.5": 0.16, "1": 0.31}[c.split("_s")[1]]}
    return S


# ── report ───────────────────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--synthetic", default=None, choices=["O2", "O3"], help="pipeline test with a planted law")
    ap.add_argument("--min-n", type=int, default=40)
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("== rq3-alpha-online stage 1: per-regime alpha surfaces and the online law (MDD-PTECT, frozen v5 simulator)")
    if a.synthetic:
        S = fake_stats()
        planted = ("O2", [(0.15, 0.45)]) if a.synthetic == "O2" else ("O3", [(0.18, 0.30), (0.30, 0.60)])
        surf = synthetic(S, planted)
        print(f"   SYNTHETIC surfaces, planted {planted}")
        nseeds = {c: 0 for c in surf}
        by_regime = {}
    else:
        S, surf, nseeds, by_regime = {}, {}, {}, {}
        for c in FIT + HELD_OUT:
            st = load_stats(c, a.root)
            sf = load_surface(c, a.root)
            if st is None or sf is None:
                print(f"   {c}: missing outputs (stats {st is not None}, surface {sf is not None})")
                continue
            S[c], surf[c], nseeds[c], by_regime[c] = st, sf[0], len(sf[2]), sf[3]

    print("\n-- 0. checks")
    fit_cells = [c for c in FIT if c in surf and (a.synthetic or nseeds[c] >= a.min_n)]
    held = [c for c in HELD_OUT if c in surf and (a.synthetic or nseeds[c] >= a.min_n)]
    for c in FIT + HELD_OUT:
        if c in surf:
            st = S[c]
            print(f"   {c:10s} seeds {nseeds[c]:3d}, stats n {st['n']:2d}: tau {st['tau']:.3f}, a {st['a_full']:.3f} / "
                  f"{st['a_short']:.3f}, x {st['tau'] / st['a_full']:.3f} / {st['tau'] / st['a_short']:.3f}, "
                  f"sigma {st['sigma']:.3f}")
    if not a.synthetic:
        n, bad, cells_checked = 0, 0, 0
        for c in fit_cells + held:
            ref_dir = diag_reference(c)
            hit = 0
            for s in cell_seeds(c):
                f, ref = Path(a.root) / "out" / c / f"s{s}.json", ref_dir / f"s{s}.json"
                if not (f.exists() and ref.exists()):
                    continue
                d, r = json.loads(f.read_text()), json.loads(ref.read_text())
                for al in RUN.ALPHAS:
                    k = f"MDD-PTECT@{al:g}"
                    if k in r:
                        n, hit = n + 1, hit + 1
                        bad += abs(d[f"{al:g}|{al:g}"]["total"] - r[k]["total"]) > 1e-6
            cells_checked += hit > 0
        print(f"   diagonal (a | a) = the earlier fixed-alpha run of the same instance: {n - bad} / {n} equal, "
              f"{cells_checked} cells")

    print("\n-- 1. surfaces (% vs alpha 3 | 3; best pair, best single alpha, room = best single over best pair)")
    rows = []
    for c in fit_cells + held:
        T = surf[c]
        i, j = np.unravel_index(np.argmin(T), T.shape)
        diag = np.array([T[k, k] for k in range(len(A))])
        kd = int(np.argmin(diag))
        base = T[list(A).index(3.0), list(A).index(3.0)]
        room = 100 * (diag[kd] / T.min() - 1)
        role = "fit" if c in fit_cells else "held out"
        print(f"   {c:10s} ({role:8s}) best pair {A[i]:g} | {A[j]:g} {100 * (T.min() / base - 1):+6.2f}%   best single "
              f"{A[kd]:g} {100 * (diag[kd] / base - 1):+6.2f}%   room {room:5.2f}")
        rows.append({"cell": c, "role": role, "best_full": A[i], "best_short": A[j], "best_pair_vs3": 100 * (T.min() / base - 1),
                     "best_single": A[kd], "best_single_vs3": 100 * (diag[kd] / base - 1), "room": room})
    if fit_cells:
        print(f"   mean room over the fit cells: {np.mean([r['room'] for r in rows if r['role'] == 'fit']):.2f} points")
    with open(out_dir / "surfaces.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "role", "best_full", "best_short", "best_pair_vs3", "best_single",
                                          "best_single_vs3", "room"])
        w.writeheader()
        w.writerows(rows)
    if by_regime:
        print("   per regime (exploratory): best alpha_short with alpha_full at the cell's best pair, short-block "
              "tardiness only")
        for c in fit_cells:
            Ts = by_regime[c].get("short")
            if Ts is None or not np.any(Ts):
                continue
            i, _ = np.unravel_index(np.argmin(surf[c]), surf[c].shape)
            print(f"     {c:10s} short blocks: best alpha_short {A[int(np.argmin(Ts[i]))]:g}")

    if len(fit_cells) < 3:
        print("\n   too few fit cells for the fit")
        return
    print("\n-- 2. laws (mean regret over the fit cells: in sample / leave-one-cell-out)")
    x0_lo = min(S[c]["tau"] / S[c]["a_full"] for c in fit_cells if c == "m15_b2") if "m15_b2" in fit_cells else 0.0
    res = {}
    for form in ("S0", "S", "O1", "O2", "O3", "M"):
        knots, regs, lo = fit(form, fit_cells, S, surf, x0_lo)
        res[form] = {"knots": knots, "in": float(np.mean(regs)), "loo": float(np.mean(list(lo.values()))),
                     "per_cell_in": dict(zip(fit_cells, map(float, regs))), "per_cell_loo": lo}
        kn = "; ".join(f"x0 {k[0]:.3f} x1 {k[1]:.3f}" for k in knots)
        print(f"   {form:3s} {kn:40s} {res[form]['in']:6.2f}% / {res[form]['loo']:6.2f}%")
    print("   per cell, leave-one-out regret:")
    print("     " + " ".join(f"{c:>9s}" for c in fit_cells))
    for form in res:
        print(f"   {form:3s} " + " ".join(f"{res[form]['per_cell_loo'][c]:9.2f}" for c in fit_cells))
    print("   the laws' pairs (alpha_full | alpha_short) per cell, in-sample knots:")
    for form in ("S0", "S", "O1", "O2", "O3", "M"):
        print(f"   {form:3s} " + " ".join(
            "{:>4.2f}|{:<4.2f}".format(*law_pair(form, res[form]["knots"], S[c])) for c in fit_cells))

    print("\n-- 3. selection and gate (pre-declared)")
    best = min(res[f]["loo"] for f in ("O1", "O2", "O3"))
    sel = next(f for f in ("O1", "O2", "O3") if res[f]["loo"] <= best + SIMPLER)
    g1 = res[sel]["loo"] <= res["S"]["loo"] - G1_MARGIN
    print(f"   selected {sel} (leave-one-out {res[sel]['loo']:.2f}%; lowest online {best:.2f}%; simplest within "
          f"{SIMPLER} points)")
    print(f"   G1: {sel} {res[sel]['loo']:.2f}% vs S {res['S']['loo']:.2f}% (needs {G1_MARGIN:.1f} points lower): "
          f"{'PASS' if g1 else 'FAIL'}")
    print(f"   one hinge for both fleet states? O2 {res['O2']['loo']:.2f}% vs O3 {res['O3']['loo']:.2f}% leave-one-out")
    x1_sel = max(k[1] for k in res[sel]["knots"])
    if x1_sel > 0.534:
        print(f"   NOTE: the selected law's x1 {x1_sel:.3f} exceeds m200_r035's full-fleet x 0.534 (not a fit cell)")

    if held:
        print("\n-- 4. held-out cells (test-seed surfaces; interpolated, stage 2 runs the laws exactly)")
        for c in held:
            line = []
            for form in ("S0", sel, "M"):
                af, ash = law_pair(form, res[form]["knots"], S[c])
                line.append(f"{form} {af:.2f}|{ash:.2f} {float(regret(surf[c], af, ash)):5.2f}")
            print(f"   {c:10s} " + "   ".join(line))

    spec = {"stage": 1, "G1": "PASS" if g1 else "FAIL", "selected": sel,
            "laws": {"static": {"form": "static", "knots": {"full": [X0_S0, X1_S0]}},
                     "online": {"form": "online", "source": sel,
                                "knots": {"full": list(res[sel]["knots"][0]),
                                          "short": list(res[sel]["knots"][-1])}},
                     "map": {"form": "static", "knots": {"full": [X0_S0, X1_S0]}, "short_alpha": 0.0}},
            "fit": {f: {"knots": res[f]["knots"], "in": res[f]["in"], "loo": res[f]["loo"]} for f in res},
            "cells": {c: {**S[c], "x_full": S[c]["tau"] / S[c]["a_full"], "x_short": S[c]["tau"] / S[c]["a_short"],
                          "pair": {f: law_pair(f, res[f]["knots"], S[c]) for f in ("S0", sel, "M")}} for c in S},
            "law_cells": [c for c in dict.fromkeys(STAGE2_CELLS + HELD_OUT) if c in S]}
    (out_dir / "fit.json").write_text(json.dumps(spec["fit"], indent=1))
    (out_dir / "law_online.json").write_text(json.dumps(spec, indent=1))
    print(f"\n   wrote {out_dir / 'law_online.json'} (G1 {spec['G1']}, selected {sel})")


if __name__ == "__main__":
    main()
