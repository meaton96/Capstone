"""
@file analyze.py
@brief rq3-alpha-size (2026-10-21): the pre-declared analysis and gate of README.md.
  1. α curves per cell (fitting seeds 2000-2039, pooled window tardiness, % vs MDD-PTECT@3), α* for PTECT and PECT, and
     the α within 1% of the best.
  2. Formulas for the PTECT α, fitted on the 18 fitting cells: M0 constant α; M1 a - b·σ; M2 a - c·τ; M3 a - b·σ - c·τ
     (clipped at 0), with σ = std of log machine speed factors and τ = mean travel time to a candidate machine / mean op
     time, both per instance. A formula's α is rounded to the nearest grid value. Regret of a cell = % its pooled
     tardiness at the formula's α exceeds its pooled tardiness at its own best α; the fit minimizes the mean regret over
     cells (cells weighted equally). Selection: leave-one-cell-out regret, the simplest model within 0.5 points of the best.
  3. Test seeds 0-39: the selected formula (fitted on all 18 cells) vs α 3, α 0 (TECT), M0 and the cell's best α (chosen
     on fitting seeds); held-out cells (30 / 75 machines at s 0.75) vs α 3 and the hindsight best α on their test seeds.
     Pooled %, seed-bootstrap 95% CI, Wilcoxon, Holm across cells.
  4. α* per regime block (normal / surge / short fleet), fitting seeds.
  5. Gate G1-G3 as declared in README.md.
  -> analysis.out, summary.csv (per cell), alpha_curves.csv, fit.json
"""
import importlib.util
import io
import itertools
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent
_s = importlib.util.spec_from_file_location("rq3an", R / "rq3-gen-v5" / "analyze.py")
rq3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rq3)
_s = importlib.util.spec_from_file_location("ar", HERE / "run.py")
ar = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ar)
A = np.array(ar.ALPHAS)
FIT, TEST = ar.FIT_SEEDS, ar.TEST_SEEDS
FIT_CELLS = [f"{f}_s{s:g}" for f in ar.FLOORS[::-1] for s in ar.SPREADS]      # m15 ... m100
HELD = ar.HELD_OUT
REF = "MDD-PTECT@3"


def load(cell):
    """{seed: {"T": {config: total}, "blocks": {regime: {config: tardiness}}, "stats": {...}}} for one cell."""
    out = {}
    for f in (HERE / "out" / cell).glob("s*.json"):
        d = json.loads(f.read_text())
        seed = int(f.stem[1:])
        cfg = {k: v for k, v in d.items() if not k.startswith("_")}
        reg = {}
        for b in d["_blocks"]:
            r = "short" if (b.get("agv_count") or 99) <= 2 or b.get("load_profile") == "short" else b.get("load_profile")
            for k, v in cfg.items():
                reg.setdefault(r, {}).setdefault(k, 0.0)
                reg[r][k] += v["blocks"].get(str(b["block"]), 0.0)
        st = dict(d["_stats"])
        st["tau"] = st["travel_mean_s"] / st["p_mean_s"]
        out[seed] = {"T": {k: v["total"] for k, v in cfg.items()}, "blocks": reg, "stats": st}
    return out


def grid_array(data, seeds, rule="PTECT"):
    """(instances x alphas) tardiness array, plus σ and τ per instance."""
    ss = [s for s in seeds if s in data]
    T = np.array([[data[s]["T"][f"MDD-{rule}@{a:g}"] for a in A] for s in ss])
    sig = np.array([data[s]["stats"]["sigma"] for s in ss])
    tau = np.array([data[s]["stats"]["tau"] for s in ss])
    return ss, T, sig, tau


MID = (A[1:] + A[:-1]) / 2.0


def idx_of(alpha):
    """Nearest grid index of each α (array)."""
    return np.searchsorted(MID, np.clip(alpha, 0.0, None))


def regret(cells, model, params):
    """Mean over cells of the % a cell's pooled tardiness at the formula's α exceeds that at the cell's best α."""
    out = []
    for c in cells:
        T, sig, tau = c
        a = alpha_of(model, params, sig, tau)
        got = T[np.arange(len(T)), idx_of(a)].sum()
        best = T.sum(axis=0).min()
        out.append(100.0 * (got / best - 1.0))
    return float(np.mean(out))


def alpha_of(model, p, sig, tau):
    if model == "M0":
        return np.full_like(sig, p[0])
    if model == "M1":
        return np.maximum(0.0, p[0] - p[1] * sig)
    if model == "M2":
        return np.maximum(0.0, p[0] - p[1] * tau)
    return np.maximum(0.0, p[0] - p[1] * sig - p[2] * tau)


GRIDS = {"M0": [A],
         "M1": [np.arange(0, 5.01, 0.1), np.arange(0, 15.01, 0.25)],
         "M2": [np.arange(0, 5.01, 0.1), np.arange(0, 60.01, 0.5)],
         "M3": [np.arange(0, 5.01, 0.1), np.arange(0, 15.01, 0.5), np.arange(0, 60.01, 1.0)]}
N_PARAMS = {"M0": 1, "M1": 2, "M2": 2, "M3": 3}


def fit(model, cells):
    """Grid search minimizing the mean regret; vectorized over the first parameter (the intercept)."""
    if model == "M0":
        rs = [regret(cells, "M0", (a,)) for a in A]
        i = int(np.argmin(rs))
        return (float(A[i]),), float(rs[i])
    a_grid = GRIDS[model][0]
    best, arg = np.inf, None
    prep = [(T, sig, tau, T.sum(axis=0).min(), np.arange(len(T))[None, :]) for T, sig, tau in cells]
    for rest in itertools.product(*GRIDS[model][1:]):
        tot = np.zeros(len(a_grid))
        for T, sig, tau, best_c, rows in prep:
            if model == "M1":
                sub = rest[0] * sig
            elif model == "M2":
                sub = rest[0] * tau
            else:
                sub = rest[0] * sig + rest[1] * tau
            alpha = np.maximum(0.0, a_grid[:, None] - sub[None, :])
            got = T[rows, idx_of(alpha)].sum(axis=1)
            tot += 100.0 * (got / best_c - 1.0)
        tot /= len(prep)
        i = int(np.argmin(tot))
        if tot[i] < best - 1e-12:
            best, arg = float(tot[i]), (float(a_grid[i]),) + tuple(float(x) for x in rest)
    return arg, best


def main():
    rng = np.random.default_rng(0)
    print("== rq3-alpha-size: best PECT / PTECT α vs floor size and machine-speed spread; pre-declared analysis and gate")
    data = {c: load(c) for c in FIT_CELLS + HELD}
    ready = [c for c in FIT_CELLS if sum(s in data[c] for s in FIT) == len(FIT) and sum(s in data[c] for s in TEST) == len(TEST)]
    print(f"cells complete: {len(ready)} / {len(FIT_CELLS)} fitting cells; held-out "
          + ", ".join(f"{c} {sum(s in data[c] for s in TEST)}/40" for c in HELD))
    # 1. α curves
    rows, curves = [], []
    print("\n-- 1. α curves, fitting seeds, pooled % vs MDD-PTECT@3 (PTECT row; PECT's best in brackets)")
    print("   cell        " + " ".join(f"{a:>6g}" for a in A) + "   α*PTECT  α*PECT  within 1% (PTECT)   σ     τ")
    astar = {}
    for c in ready:
        ss, T, sig, tau = grid_array(data[c], FIT, "PTECT")
        _, P, _, _ = grid_array(data[c], FIT, "PECT")
        ref = sum(data[c][s]["T"][REF] for s in ss)
        pt, pe = T.sum(axis=0), P.sum(axis=0)
        pct = 100 * (pt / ref - 1)
        a_pt, a_pe = A[int(np.argmin(pt))], A[int(np.argmin(pe))]
        near = [f"{a:g}" for a, v in zip(A, pt) if v <= pt.min() * 1.01]
        astar[c] = a_pt
        print(f"   {c:10s} " + " ".join(f"{v:+6.1f}" for v in pct) + f"   {a_pt:>6g}  {a_pe:>6g}"
              f" [{100 * (pe.min() / ref - 1):+.1f}]  {','.join(near):18s} {sig.mean():.2f} {tau.mean():.3f}")
        for a, v, w in zip(A, pt, pe):
            curves.append({"cell": c, "alpha": a, "PTECT_pct": 100 * (v / ref - 1), "PECT_pct": 100 * (w / ref - 1)})
        rows.append({"cell": c, "alpha_star_PTECT": a_pt, "alpha_star_PECT": a_pe, "best_PTECT_pct": float(pct.min()),
                     "best_PECT_pct": float(100 * (pe.min() / ref - 1)), "sigma": float(sig.mean()), "tau": float(tau.mean())})
    pd.DataFrame(curves).to_csv(HERE / "alpha_curves.csv", index=False)
    if len(ready) < len(FIT_CELLS):
        print("\n(fit, test and gate need all 18 fitting cells)")
        pd.DataFrame(rows).to_csv(HERE / "summary.csv", index=False)
        return
    # 2. formulas, leave-one-cell-out
    cells = {c: grid_array(data[c], FIT, "PTECT")[1:] for c in FIT_CELLS}
    print("\n-- 2. formulas for the PTECT α (fitting seeds; regret = % over the cell's best α, mean over cells)")
    fits, loo = {}, {}
    for m in ("M0", "M1", "M2", "M3"):
        p, r = fit(m, list(cells.values()))
        fits[m] = (p, r)
        lr = []
        for h in FIT_CELLS:
            ph, _ = fit(m, [v for k, v in cells.items() if k != h])
            lr.append(regret([cells[h]], m, ph))
        loo[m] = float(np.mean(lr))
        print(f"   {m}: params {tuple(round(float(x), 3) for x in p)}  in-sample regret {r:.2f}%  leave-one-cell-out {loo[m]:.2f}%")
    best_loo = min(loo.values())
    sel = min((m for m in loo if loo[m] <= best_loo + 0.5), key=lambda m: (N_PARAMS[m], loo[m]))
    print(f"   selected: {sel} (simplest within 0.5 points of the best leave-one-cell-out regret {best_loo:.2f}%)")
    json.dump({"fits": {m: {"params": [float(x) for x in p], "regret": r, "loo": loo[m]} for m, (p, r) in fits.items()},
               "selected": sel}, open(HERE / "fit.json", "w"), indent=1)
    # 3. test seeds
    print("\n-- 3. test seeds 0-39: selected formula vs references (pooled %, 95% CI, W/L, Wilcoxon)")
    psel, p0 = fits[sel][0], fits["M0"][0]
    tests, pv = [], []
    for c in FIT_CELLS + HELD:
        ss, T, sig, tau = grid_array(data[c], TEST, "PTECT")
        if len(ss) < 40:
            print(f"   {c}: test seeds incomplete ({len(ss)})")
            continue
        model = dict(zip(ss, T[np.arange(len(ss)), idx_of(alpha_of(sel, psel, sig, tau))]))
        m0 = dict(zip(ss, T[np.arange(len(ss)), idx_of(alpha_of("M0", p0, sig, tau))]))
        a3 = {s: data[c][s]["T"][REF] for s in ss}
        a0 = {s: data[c][s]["T"]["MDD-PTECT@0"] for s in ss}
        if c in astar:
            bi = int(np.where(A == astar[c])[0][0])
            best = dict(zip(ss, T[:, bi]))
            lab_best = f"cell best α {astar[c]:g} (fitting)"
        else:
            bi = int(np.argmin(T.sum(axis=0)))
            best = dict(zip(ss, T[:, bi]))
            lab_best = f"hindsight best α {A[bi]:g} (test)"
        r3, r0, rm0, rb = (rq3.compare(model, x, rng) for x in (a3, a0, m0, best))
        pv.append((c, r3))
        mean_a = float(np.mean(alpha_of(sel, psel, sig, tau)))
        print(f"   {c:10s} formula α ≈ {mean_a:.2f} | vs α 3 {r3['pooled_%']:+6.2f} [{r3['ci_lo']:+.2f}, {r3['ci_hi']:+.2f}] "
              f"{r3['better']}/{r3['worse']} p={r3['wilcoxon_p']:.2g} | vs α 0 {r0['pooled_%']:+6.2f} | vs M0 {rm0['pooled_%']:+6.2f}"
              f" | vs {lab_best} {rb['pooled_%']:+6.2f}")
        tests.append({"cell": c, "held_out": c in HELD, "formula_alpha_mean": mean_a,
                      **{f"vs_a3_{k}": v for k, v in r3.items()}, "vs_a0_%": r0["pooled_%"], "vs_M0_%": rm0["pooled_%"],
                      "vs_best_%": rb["pooled_%"], "best_alpha": float(A[bi])})
    # Holm on vs α 3 (fitting cells)
    fp = sorted([(c, r) for c, r in pv if c in FIT_CELLS], key=lambda x: x[1]["wilcoxon_p"])
    k = len(fp)
    holm, ok = {}, True
    for i, (c, r) in enumerate(fp):
        ok = ok and r["wilcoxon_p"] <= 0.05 / (k - i)
        holm[c] = ok
    tdf = pd.DataFrame(tests)
    tdf["holm_sig_vs_a3"] = tdf.cell.map(holm).fillna(False)
    # 4. per regime
    print("\n-- 4. α* per regime block (PTECT, fitting seeds)")
    for c in FIT_CELLS:
        parts = []
        for reg in ("normal", "surge", "short"):
            v = np.array([sum(data[c][s]["blocks"].get(reg, {}).get(f"MDD-PTECT@{a:g}", 0.0) for s in FIT if s in data[c])
                          for a in A])
            if v.sum() > 0:
                parts.append(f"{reg} {A[int(np.argmin(v))]:g}")
        print(f"   {c:10s} " + ", ".join(parts))
    # 5. gate
    print("\n-- 5. gate (README)")
    g1 = astar["m15_s0"] - astar["m100_s0"]
    print(f"   G1 size matters at s 0: α*(15) − α*(100) = {astar['m15_s0']:g} − {astar['m100_s0']:g} = {g1:g} "
          f"-> {'PASS' if g1 >= 1.0 else 'FAIL'} (need ≥ 1.0)")
    fit_rows = tdf[~tdf.held_out]
    has_size = sel in ("M2", "M3")
    beats_m1 = loo[sel] <= loo["M1"] - 1.0
    gap = float(fit_rows["vs_best_%"].mean())
    n_sig = int((fit_rows.holm_sig_vs_a3 & (fit_rows["vs_a3_pooled_%"] < 0)).sum())
    worse = fit_rows[(fit_rows.holm_sig_vs_a3) & (fit_rows["vs_a3_pooled_%"] > 0)]
    g2 = has_size and beats_m1 and gap <= 2.0 and n_sig >= len(fit_rows) / 2 and worse.empty
    print(f"   G2 the formula works: selected {sel} (size term: {has_size}); LOO regret {loo[sel]:.2f} vs M1 {loo['M1']:.2f} "
          f"(need ≥ 1 point better: {beats_m1}); test gap to the cell's best α {gap:.2f}% (need ≤ 2); Holm-significantly "
          f"better than α 3 in {n_sig} / {len(fit_rows)} cells (need ≥ half); significantly worse in {len(worse)} -> "
          f"{'PASS' if g2 else 'FAIL'}")
    held = tdf[tdf.held_out]
    g3 = (not held.empty and len(held) == len(HELD) and bool((held["vs_a3_pooled_%"] < 0).all() and (held["vs_a3_ci_hi"] < 0).all())
          and bool((held["vs_best_%"] <= 3.0).all()))
    print("   G3 it transfers (held-out cells): " + "; ".join(
        f"{r.cell} vs α 3 {r['vs_a3_pooled_%']:+.2f} [CI hi {r['vs_a3_ci_hi']:+.2f}], vs hindsight best α {r['vs_best_%']:+.2f}"
        for _, r in held.iterrows()) + f" -> {'PASS' if g3 else 'FAIL'}")
    print(f"   overall: {'GO (G1-G3 pass)' if (g1 >= 1.0 and g2 and g3) else 'see README: report the α curves as a guideline'}")
    pd.DataFrame(rows).merge(tdf, on="cell", how="outer").to_csv(HERE / "summary.csv", index=False)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
