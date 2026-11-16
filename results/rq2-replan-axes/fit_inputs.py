"""
@file fit_inputs.py
@brief rq2-replan-axes, design input for §8's fit (README §17; drafted 11-09, not yet approved). It builds the table of
       the 35 distinct conditions of axes 1, 2 and 6 with the covariates the decision rules brought into the fit, and
       describes the design matrix. It fits nothing and reads no reading:
       (1) the conditions: Δ* / p̄ with its 90% bootstrap and the rule 9 flag (as analysis.out), and the covariates, each
           the mean over the condition's 40 seeds: m, τ / p̄, T_reg / p̄, the on-duty AGV share, the machine busy share
           u, the time-weighted mean due-date allowance c of the window's blocks, and the surge and short-fleet time
           shares of the window;
       (2) the on-duty AGV share rebuilt from the instance's AGV schedule (axis 1 ran before it was recorded), checked
           against the recorded value on axes 2 and 6;
       (3) the design: each covariate's range, how many conditions move it away from B2, the correlations of the
           covariates as the fit codes them (ratios in log2, utilizations and time shares as they are) and their
           variance inflation;
       (4) the kept share of every fixed grid interval D1 across the 35 conditions (the constant-interval rules the fit
           is compared with; descriptive, the curves are those of analysis.out).
@par Usage
@code{.sh}
.venv/bin/python results/rq2-replan-axes/fit_inputs.py > results/rq2-replan-axes/fit_inputs.out
@endcode
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402
import run as R  # noqa: E402

LOG_COLS = ("m", "tau_p", "treg_p", "c")                 # ratios: log2 in the fit (README §17)
SHARE_COLS = ("agv", "u", "surge", "short")              # utilizations and time shares: as they are
NAMES = {"m": "m", "tau_p": "τ / p̄", "treg_p": "T_reg / p̄", "agv": "AGV on duty", "u": "u (machines)", "c": "mean c",
         "surge": "surge share", "short": "short-fleet share"}


def machines(arm, k):
    return int(k) if arm == "size" else (R.FLEET_M if arm == "fleet" else 15)


def instance_terms(arm, k, seed, desc):
    """On-duty AGV share (recorded, or rebuilt from the schedule: busy AGV-s / on-duty AGV-s), mean c and the regime
    shares of the window, from the instance alone (no simulation)."""
    e = R.AxEpisode(arm, k, seed)
    t0, t1 = e.warmup, e.warmup + e.window
    n = len(e.twin().agvs)
    rebuilt = desc["agv_busy_share"] * n * t1 / R.on_duty_seconds(e.sched, n, t1)
    over = [(b, min(b["end"], t1) - max(b["start"], t0)) for b in e.blocks if b["end"] > t0 and b["start"] < t1]
    span = sum(w for _, w in over)

    def c_of(b):
        v = b["due_date_allowance"]
        return float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)

    return {"agv_rebuilt": rebuilt, "agv_recorded": desc.get("agv_busy_on_duty"),
            "c": sum(c_of(b) * w for b, w in over) / span,
            "surge": sum(w for b, w in over if b["load_profile"] == "surge") / e.window,
            "short": sum(w for b, w in over if b["load_profile"] == "short") / e.window}


def distinct_keys(data, base):
    """The distinct conditions of axes 1, 2 and 6 in print order: aliases (B2, reg k 0.25, m50 = r 0.52) once, the
    `all` units check left out (as axis2_checks.py (3))."""
    seen, keys = set(), []
    for key in sorted((k for k in data if k[0] != "all"), key=lambda k: (A.ARMS.index(k[0]), str(k[1]))):
        if id(data[key]) in seen or key not in base or 60.0 not in data[key]:
            continue
        seen.add(id(data[key]))
        keys.append(key)
    return keys


def main():
    data, base = A.load(HERE)
    # ── (1) the conditions ──────────────────────────────────────────────────────────────────────────────────────────────
    conds = []
    for key in distinct_keys(data, base):
        arm, k = key
        ds, seeds, g = A.level_curve(data[key], {s: v["best"] for s, v in base[key].items()})
        d = A.dstar(ds, g, A.X_PRIMARY)
        descs = [base[key][s]["describe"] for s in seeds]
        terms = [instance_terms(arm, k, s, base[key][s]["describe"]) for s in seeds]
        p = float(np.mean([x["p_mean_plan"] for x in descs]))
        conds.append({
            "key": key, "label": A.label(*key), "n": len(seeds), "ds": ds, "g": g, "p": p,
            "dstar_p": d["dstar"] * (k if arm == "proc" else 1.0) / p,        # Δ = k D1 in proc (README §4)
            "upper": d["best_d"] == ds[0], "best_d": d["best_d"], "g_best": d["g_best"],
            "m": machines(arm, k),
            "tau_p": float(np.mean([x["transport_mean"] for x in descs])) / p,
            "treg_p": float(np.mean([x["block_seconds"] for x in descs])) / p,
            "agv": float(np.mean([t["agv_recorded"] if t["agv_recorded"] is not None else t["agv_rebuilt"] for t in terms])),
            "agv_src": "recorded" if all(t["agv_recorded"] is not None for t in terms) else "rebuilt",
            "u": float(np.mean([x["machine_busy_share"] for x in descs])),
            "c": float(np.mean([t["c"] for t in terms])),
            "surge": float(np.mean([t["surge"] for t in terms])), "short": float(np.mean([t["short"] for t in terms])),
            "terms": terms})

    # Δ* / p̄ and its bootstrap as analysis.out / dstar.csv print them
    dcsv = {}
    for r in csv.DictReader(open(HERE / "dstar.csv")):
        dcsv[(r["arm"], r["k"], r.get("ell", ""), r.get("c", ""), r.get("block", ""), r.get("mix", ""))] = r


    def csv_row(key):
        arm, k = key
        if arm == "ax2":
            ell, c, b, mix = k
            for r in dcsv.values():
                if r["arm"] == "ax2" and float(r["ell"]) == ell and float(r["c"]) == c and float(r["block"]) == b \
                        and r["mix"] == mix:
                    return r
        for r in dcsv.values():
            if r["arm"] == arm and r["k"] and float(r["k"]) == k:
                return r
        return None


    print(f"(1) the {len(conds)} distinct conditions of axes 1, 2 and 6 (40 seeds each; ≤ = upper bound, rule 9; covariates "
          f"are means over the seeds)")
    print(f"{'condition':34s} {'Δ*/p̄':>6s} {'90% boot':>13s} {'best':>5s} {'m':>4s} {'τ/p̄':>6s} {'Treg/p̄':>7s} "
          f"{'AGV':>5s} {'u':>5s} {'c':>5s} {'surge':>5s} {'short':>5s}")
    for c in conds:
        r = csv_row(c["key"])
        lo, hi = float(r["boot_lo_over_p"]), float(r["boot_hi_over_p"])
        assert abs(float(r["dstar_over_p"]) - c["dstar_p"]) < 1e-6 * max(1.0, c["dstar_p"]), c["label"]
        c["boot"] = (lo, hi)
        print(f"{c['label']:34s} {'≤' if c['upper'] else ' '}{c['dstar_p']:5.2f} [{lo:4.2f}, {hi:4.2f}] {c['best_d']:5.0f} "
              f"{c['m']:4d} {c['tau_p']:6.3f} {c['treg_p']:7.2f} {c['agv']:5.2f}{'*' if c['agv_src'] == 'rebuilt' else ' '}"
              f"{c['u']:5.2f} {c['c']:5.2f} {c['surge']:5.2f} {c['short']:5.2f}")
    print(f"   {sum(c['upper'] for c in conds)} of {len(conds)} are upper bounds; * = on-duty AGV share rebuilt from the AGV "
          f"schedule (axis 1 ran before it was recorded)")

    # ── (2) the rebuilt on-duty AGV share against the recorded one ─────────────────────────────────────────────────────
    diffs = [abs(t["agv_rebuilt"] - t["agv_recorded"]) / t["agv_recorded"] for c in conds for t in c["terms"]
             if t["agv_recorded"] is not None]
    print()
    print(f"(2) on-duty AGV share rebuilt from the schedule vs recorded, every fixed-pair task of axes 2 and 6 "
          f"({len(diffs)} tasks): largest relative difference {max(diffs):.1e}")

    # ── (3) the design ──────────────────────────────────────────────────────────────────────────────────────────────────
    b2 = next(c for c in conds if c["key"] == ("proc", 1.0))
    print()
    print("(3) the design: range over the 35 conditions, and how many move the covariate away from B2 "
          "(by more than 10% for the ratios, 0.10 for the shares)")
    for col in LOG_COLS + SHARE_COLS:
        v = np.array([c[col] for c in conds], float)
        if col in LOG_COLS:
            moved = int(np.sum(np.abs(np.log2(v / b2[col])) > np.log2(1.1)))
            print(f"   {NAMES[col]:18s} {v.min():8.3f} - {v.max():8.3f}  ({np.log2(v.max() / v.min()):4.2f} octaves); B2 "
                  f"{b2[col]:.3f}; moved at {moved:2d}")
        else:
            moved = int(np.sum(np.abs(v - b2[col]) > 0.10))
            print(f"   {NAMES[col]:18s} {v.min():8.3f} - {v.max():8.3f}; B2 {b2[col]:.3f}; moved at {moved:2d}")
    X = np.column_stack([np.log2([c[col] for c in conds]) for col in LOG_COLS] + [[c[col] for c in conds]
                                                                                  for col in SHARE_COLS])
    cols = LOG_COLS + SHARE_COLS
    corr = np.corrcoef(X.T)
    print("   correlations (log2 of the ratios m, τ / p̄, T_reg / p̄ and c; the utilizations and time shares as they are):")
    print("   " + " " * 18 + "".join(f"{NAMES[c][:9]:>10s}" for c in cols))
    for i, ci in enumerate(cols):
        print(f"   {NAMES[ci]:18s}" + "".join(f"{corr[i, j]:10.2f}" for j in range(len(cols))))
    i_agv, i_tau = cols.index("agv"), cols.index("tau_p")
    print(f"   log2 AGV share against log2 τ / p̄: {np.corrcoef(np.log2(X[:, i_agv]), X[:, i_tau])[0, 1]:.2f} "
          f"(the share as it is: {corr[i_agv, i_tau]:.2f})")


    def vif(Xs):
        out = []
        for j in range(Xs.shape[1]):
            y, Z = Xs[:, j], np.column_stack([np.ones(len(Xs)), np.delete(Xs, j, 1)])
            res = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
            out.append(np.var(y) / max(1e-12, np.var(res)))
        return out


    print("   variance inflation, all 8 covariates: " + ", ".join(f"{NAMES[c]} {v:.1f}" for c, v in zip(cols, vif(X))))
    sub = [j for j, c in enumerate(cols) if c != "tau_p"]
    print("   ... without τ / p̄ (the proposed M2): " + ", ".join(f"{NAMES[cols[j]]} {v:.1f}" for j, v in zip(sub, vif(X[:, sub]))))
    sub = [j for j, c in enumerate(cols) if c != "agv"]
    print("   ... with τ / p̄ in place of the AGV share: "
          + ", ".join(f"{NAMES[cols[j]]} {v:.1f}" for j, v in zip(sub, vif(X[:, sub]))))
    exact = [c for c in conds if not c["upper"]]
    print(f"   the {len(exact)} conditions with Δ* measured (not a bound): " + "; ".join(c["label"] for c in exact))

    # ── (4) constant intervals ─────────────────────────────────────────────────────────────────────────────────────────
    print()
    print("(4) every fixed grid interval D1 (B2-equivalent; Δ = k D1 in proc) across the 35 conditions: kept share of the "
          "best gain")
    grid = conds[0]["ds"]
    for d1 in grid:
        ks = [c["g"][c["ds"].index(d1)] / max(c["g"]) for c in conds]
        lo = int(np.argmin(ks))
        print(f"   D1 {d1:6.0f} s: keeps >= 85% at {sum(k >= 0.85 for k in ks):2d} of {len(ks)}; lowest {ks[lo]:6.1%} "
              f"({conds[lo]['label']}); median {np.median(ks):6.1%}")


if __name__ == "__main__":
    main()
