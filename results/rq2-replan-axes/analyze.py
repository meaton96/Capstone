"""
@file analyze.py
@brief rq2-replan-axes (2026-10-29, plan §5b): gain curves over the re-plan interval Δ, the best interval Δ* (the right
       edge of the plateau within X of the best gain, X fixed in README §3), checks C0 / C1 and the pre-declared
       readings S / R of axis 1 (README §4). Gain g(Δ) = re-planning gain over planning once, in % of the plan-once
       tardiness, pooled over the level's seeds: 100 (1 - Σ real(Δ) / Σ plan-once), > 0 better. Plan-once = the planner
       with one plan at the warm-up = each instance's best fixed v5 pair under the exact forecast (the <level>-fixed task).
       The curve against MDD-PTECT (rq2-replan's reference) is printed beside it. Axis 6 (README §13, floor size and fleet
       ratio): check C2 (instances = the floor-size grid's), reading S_m (slope on log2 m: per-machine vs floor-level
       decision rate), R, R120 and the single-core planning time per interval; its sections print only once axis 6
       outputs exist, so axis 1's output is unchanged before that. Axis 2 (README §15, load x due-date allowance at two
       regime block lengths, plus regime-mix cells): check C3 (planner tasks = their fixed-pair task; one job set per
       (ℓ, block, mix, seed) across c), readings R_ℓ, R_c and R_mix (the centre cell's own grid edge keeps 1 - X of the
       best gain along each line through the centre, per block length), R and R120 at every cell, the slopes on log2 u
       and log2 c (reported) and T_c / Δ. Its sections print only once axis 2 outputs exist and use their own random
       stream, so axes 1 and 6 come out unchanged.
@par Usage
@code{.sh}
.venv/bin/python results/rq2-replan-axes/analyze.py                  # out/ -> analysis.out, curve.csv, dstar.csv
.venv/bin/python results/rq2-replan-axes/analyze.py --root dryrun    # the dry run (writes under dryrun/)
.venv/bin/python results/rq2-replan-axes/analyze.py --preview        # the estimator on rq2-replan's 1a curve -> preview.out
@endcode
"""
import argparse
import io
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RQ2 = HERE.parent / "rq2-replan" / "out"
D1_GRID = (120, 180, 300, 450, 600, 900, 1800, 3600)     # = run.py's (kept here so the analysis loads no twin code)
LEVELS = {"all": (0.125, 10.0, 32.0), "proc": (0.5, 1.0, 2.0, 4.0, 16.0), "reg": (0.25, 0.5, 1.0, 2.0, 4.0),
          "size": (15.0, 20.0, 30.0, 50.0, 100.0), "fleet": (0.35, 0.52, 0.8)}   # axis 6: m15 = B2, r 0.52 = size m50
ARMS = ("all", "proc", "reg", "size", "fleet", "ax2")  # print order
# Axis 2 (README §15): cells (ℓ, c, block, mix); the centres (ℓ 1, c 2.125, B2's mix) are axis 1's levels
AX2_CENTRE = {5400: ("proc", 1.0), 1350: ("reg", 0.25)}
LEVELS["ax2"] = tuple([(ell, c, b, "b2") for b in (5400, 1350) for ell in (0.7, 1.0, 1.3) for c in (1.5, 2.125, 3.0)]
                      + [(1.0, 2.125, b, m) for b in (5400, 1350) for m in ("nosurge", "noshort")])
AX2_LINES = {"R_ℓ": ((0.7, 2.125, "b2"), (1.3, 2.125, "b2")), "R_c": ((1.0, 1.5, "b2"), (1.0, 3.0, "b2")),
             "R_mix": ((1.0, 2.125, "nosurge"), (1.0, 2.125, "noshort"))}   # (ℓ, c, mix) off the centre, per block
AX2_SEED = 2               # axis 2's bootstrap stream (axes 1 and 6 keep theirs)
PREFIX = {"size": "m", "fleet": "r"}                   # level letter (k otherwise)
FLOOR_RATE = -1.0          # axis 6 S_m: slope under the floor-level decision rate (Δ* ∝ p̄ / m); 0 = per-machine rate
PROC_TAU_BOUND = 0.282     # axis 1 proc arm (§12): |slope of log2(Δ*/p̄) on log2(τ/p̄)| <= 0.282 (90% CI end)
D_R120 = 120.0             # axis 6 R120: D1 120 s must keep 1 - X of each level's best gain
X_PRIMARY = 0.15           # plateau band: within 15% of the level's best re-planning gain (README §3; proposed, the user decides)
X_SENS = (0.10, 0.20)      # sensitivity only
G_MIN = 1.0                # a level whose best gain is below 1 point has no plateau to measure ("flat")
SLOPE_MARGIN = 0.25        # reading S: |slope of log2(Δ*/p̄) on log2 k| below this = invariant (equivalence margin)
BOOT = 2000


def parse(v):
    if v.startswith("ax2-"):                       # ax2-l<ℓ>-c<c>-b<block>[-<mix>]-d<D1> | -fixed
        p = v.split("-")
        k = (float(p[1][1:]), float(p[2][1:]), int(p[3][1:]), p[4] if len(p) == 6 else "b2")
        return "ax2", k, (None if p[-1] == "fixed" else float(p[-1][1:]))
    arm, k, d = v.split("-")
    return arm, float(k[1:]), (None if d == "fixed" else float(d[1:]))


def load(root):
    """Planner outputs {(arm, k): {D1: {seed: obj}}} and plan-once baselines {(arm, k): {seed: best fixed pair}}."""
    data, base = {}, {}
    for f in (root / "out").glob("*/s*.json"):
        d = json.loads(f.read_text())
        arm, k, d1 = parse(d["variant"])
        if d1 is None:
            base.setdefault((arm, k), {})[d["seed"]] = d
        else:
            data.setdefault((arm, k), {}).setdefault(d1, {})[d["seed"]] = d
    for x in (data, base):                          # B2 is the k = 1 level of both proc and reg
        if ("proc", 1.0) in x:
            x.setdefault(("reg", 1.0), x[("proc", 1.0)])
    if axis6(data):                                 # axis 6: m15 is B2, fleet r 0.52 is size m50 (README §13)
        for x in (data, base):
            if ("proc", 1.0) in x:
                x.setdefault(("size", 15.0), x[("proc", 1.0)])
            if ("size", 50.0) in x:
                x.setdefault(("fleet", 0.52), x[("size", 50.0)])
    if axis2(data):                                 # axis 2: the centres are B2 (5,400 s) and reg k 0.25 (1,350 s)
        for x in (data, base):
            for b, key in AX2_CENTRE.items():
                if key in x:
                    x.setdefault(("ax2", (1.0, 2.125, b, "b2")), x[key])
    return data, base


def axis6(data):
    return any(arm in ("size", "fleet") for arm, _ in data)


def axis2(data):
    return any(arm == "ax2" for arm, _ in data)


def label(arm, k):
    if arm == "ax2":
        ell, c, b, mix = k
        return f"ax2 ℓ {ell:g} c {c:g} block {b}" + ("" if mix == "b2" else f" {mix}")
    return f"{arm} {PREFIX.get(arm, 'k')} {k:g}"


def ax2_name(k):
    ell, c, b, mix = k
    return f"l{ell:g}-c{c:g}-b{b}" + ("" if mix == "b2" else f"-{mix}")


def gain(rows, seeds, base):
    """Pooled gain over `base` ({seed: tardiness}) in % of it; base None = MDD-PTECT (each task's reference run)."""
    real = sum(rows[s]["real"] for s in seeds)
    ref = sum((rows[s]["mdd_ptect"] if base is None else base[s]) for s in seeds)
    return -100.0 * (real / ref - 1.0)


def dstar(ds, g, x):
    """Plateau of the curve g over the sorted grid ds: the contiguous run of grid points around the best one whose gain
    is within x (relative) of the best. Δ* = its right edge, interpolated in log Δ to where the curve crosses the band
    (right-censored when the coarsest point is inside); the left edge likewise; width = right / left."""
    ds, g = list(ds), list(g)
    ib = int(np.argmax(g))
    gb = g[ib]
    out = {"best_d": ds[ib], "g_best": gb, "flat": gb < G_MIN}
    if out["flat"]:
        return {**out, "dstar": None, "right_grid": None, "left": None, "width": None, "cens_right": None,
                "cens_left": None}
    thr = (1.0 - x) * gb

    def edge(step):
        i = ib
        while 0 <= i + step < len(ds) and g[i + step] >= thr:
            i += step
        j = i + step
        if not 0 <= j < len(ds):
            return ds[i], ds[i], True
        f = (g[i] - thr) / (g[i] - g[j])                     # in [0, 1): the crossing between i and j
        return ds[i], math.exp(math.log(ds[i]) + f * (math.log(ds[j]) - math.log(ds[i]))), False

    r_grid, r, cr = edge(+1)
    _, lft, cl = edge(-1)
    return {**out, "dstar": r, "right_grid": r_grid, "cens_right": cr, "left": lft, "cens_left": cl,
            "width": r / lft}


def level_curve(rows_by_d, base, seeds=None):
    """Grid, common seeds (with a baseline), gains over `base` (None = MDD-PTECT)."""
    ds = sorted(rows_by_d)
    common = set.intersection(*(set(rows_by_d[d]) for d in ds)) if ds else set()
    if base is not None:
        common &= set(base)
    seeds = sorted(common) if seeds is None else seeds
    return ds, seeds, [gain(rows_by_d[d], seeds, base) for d in ds] if seeds else []


def boot_dstar(rows_by_d, base, seeds, x, rng, n=None, idx=None):
    ds = sorted(rows_by_d)
    real = np.array([[rows_by_d[d][s]["real"] for s in seeds] for d in ds])
    ref = np.array([(rows_by_d[ds[0]][s]["mdd_ptect"] if base is None else base[s]) for s in seeds])
    n = n or len(seeds)
    idx = rng.integers(0, len(seeds), (BOOT, n)) if idx is None else idx
    out = []
    for row in idx:
        g = -100.0 * (real[:, row].sum(1) / ref[row].sum() - 1.0)
        out.append(dstar(ds, g, x))
    return out


def describe(rows_by_d, seeds):
    recs = [rows_by_d[d][s]["describe"] for d in rows_by_d for s in seeds if s in rows_by_d[d]]
    df = pd.DataFrame(recs)
    return df.mean(numeric_only=True).to_dict()


def q(v, lo=5, hi=95):
    v = [x for x in v if x is not None]
    return (np.percentile(v, lo), np.median(v), np.percentile(v, hi)) if v else (None, None, None)


# ── Checks ────────────────────────────────────────────────────────────────────────────────────────────────────

def check_c0(data):
    """C0 (units): the `all` arm at k must be the k = 1 run in other units: real / k = real(k = 1), same pair schedule.
    Exact at k = 2^n; at other k float32 rounding of the scaled times may move ties (reported, not required)."""
    base = data.get(("proc", 1.0), {})
    rows, ok = [], True
    for (arm, k), by_d in sorted(data.items()):
        if arm != "all":
            continue
        for d1, by_s in sorted(by_d.items()):
            for s, x in sorted(by_s.items()):
                b = base.get(d1, {}).get(s)
                if b is None:
                    continue
                rel = x["real"] / k / b["real"] - 1.0
                same = x["schedule"] == b["schedule"]
                exact = abs(math.log2(k) - round(math.log2(k))) < 1e-12
                passed = (abs(rel) < 1e-6 and same) if exact else None
                ok &= passed is not False
                rows.append(f"   all k {k:g} d1 {d1:g} s{s}: real/k {x['real'] / k:.6f} vs k=1 {b['real']:.6f} "
                            f"(rel {rel:+.2e}), schedule {'same' if same else 'DIFFERS'}"
                            f"{'' if passed is None else (' PASS' if passed else ' FAIL')}"
                            f"{'' if exact else ' (k not a power of 2: reported only)'}")
    return ok, rows


def check_c1(data):
    """C1 (regression): proc-k1 = rq2-replan's exact-d<D1> (frozen code ba26f3f5) where the grids overlap."""
    base = data.get(("proc", 1.0), {})
    rows, ok, n = [], True, 0
    for d1, by_s in sorted(base.items()):
        for s, x in sorted(by_s.items()):
            f = RQ2 / f"exact-d{int(d1)}" / f"s{s}.json"
            if not f.exists():
                continue
            o = json.loads(f.read_text())
            same = abs(x["real"] - o["real"]) < 1e-6 and x["schedule"] == o["schedule"] \
                and abs(x["mdd_ptect"] - o["mdd_ptect"]) < 1e-6
            ok &= same
            n += 1
            if not same or n <= 6:
                rows.append(f"   proc k 1 d1 {d1:g} s{s}: {x['real']:.6f} vs rq2-replan {o['real']:.6f}, MDD-PTECT "
                            f"{x['mdd_ptect']:.6f} vs {o['mdd_ptect']:.6f}: {'PASS' if same else 'FAIL'}")
    rows.append(f"   {n} tasks compared")
    return ok, rows


# ── Readings ──────────────────────────────────────────────────────────────────────────────────────────────────

def reading(arm, data, res, rng):
    """S: slope of log2(Δ*/p̄) on log2 k over the arm's non-flat levels (seed bootstrap, joint over levels); R: transfer
    regret of B2's plateau edge (in B2-equivalent seconds) at every level."""
    lv = [k for k in LEVELS[arm] if (arm, k) in res and not res[(arm, k)]["d"]["flat"]]
    flat = [k for k in LEVELS[arm] if (arm, k) in res and res[(arm, k)]["d"]["flat"]]
    out = [f"   {arm}: flat levels (re-planning gain < {G_MIN} point, any Δ will do): "
           f"{', '.join(f'k {k:g}' for k in flat)}"] if flat else []
    if len(lv) < 3:
        return out + [f"   {arm}: {len(lv)} complete non-flat levels; reading S needs >= 3"]
    seeds = sorted(set.intersection(*(set(res[(arm, k)]["seeds"]) for k in lv)))
    idx = rng.integers(0, len(seeds), (BOOT, len(seeds)))
    pb = {k: res[(arm, k)]["desc"]["p_mean_plan"] for k in lv}
    scale = {k: (k if arm in ("all", "proc") else 1.0) for k in lv}
    boots = {k: boot_dstar(data[(arm, k)], res[(arm, k)]["base"], seeds, X_PRIMARY, rng, idx=idx) for k in lv}
    xs = np.log2(lv)

    def slope(ds):
        if any(d is None for d in ds):                   # a resample with a flat level: no Δ* there
            return None
        y = np.log2([ds[i] * scale[k] / pb[k] for i, k in enumerate(lv)])
        return float(np.polyfit(xs, y, 1)[0])
    point = slope([res[(arm, k)]["d"]["dstar"] for k in lv])
    bs = [b for b in (slope([boots[k][b]["dstar"] for k in lv]) for b in range(BOOT)) if b is not None]
    lo, hi = np.percentile(bs, 5), np.percentile(bs, 95)
    if lo > -SLOPE_MARGIN and hi < SLOPE_MARGIN:
        verdict = "INVARIANT (90% CI inside the margin)"
    elif lo > 0 or hi < 0:
        verdict = "MOVES Δ* (90% CI excludes 0): enters the fit"
    elif len(seeds) < 40:
        verdict = "INDETERMINATE: extend to seeds 20-39 (README §6 rule 5)"
    else:
        verdict = "INDETERMINATE at 40 seeds (rule 5 spent): README §11, decision a"
    out.append(f"   {arm} S: slope of log2(Δ*/p̄) on log2 k = {point:+.3f} [90% {lo:+.3f}, {hi:+.3f}] over k "
               f"{', '.join(f'{k:g}' for k in lv)} (seeds {len(seeds)}, {len(bs)} resamples) -> {verdict}")
    # A finer grid point can only raise the best gain, hence the band, so a plateau that reaches the finest point makes
    # Δ* an upper bound even when the crossing itself is interior (README §11: how rule 6 applies to that case).
    right = [k for k in lv if res[(arm, k)]["d"]["cens_right"]]
    left = [k for k in lv if res[(arm, k)]["d"]["cens_left"]]
    at_end = [k for k in left if res[(arm, k)]["d"]["best_d"] == res[(arm, k)]["curve"][0][0]]
    if right:
        out.append(f"      Δ* right-censored at the coarsest grid point (rule 6: add D1 7200 s there): k "
                   f"{', '.join(f'{k:g}' for k in right)}")
    if left:
        finest = min(res[(arm, k)]["curve"][0][0] for k in left)
        ks, ke = (", ".join(f"{k:g}" for k in x) or "-" for x in (left, at_end))
        if finest > 60:
            out.append(f"      plateau reaches the finest grid point D1 {finest:g} s at k {ks} (the best gain is there at "
                       f"k {ke}): Δ* is an upper bound (rule 6 as read in README §11: add D1 60 s at these levels)")
        elif at_end:
            out.append(f"      best gain still at the finest point run, D1 60 s, at k {ke}: Δ* stays an upper bound there "
                       f"(README §11, decision b)")
        else:
            out.append(f"      plateau reaches D1 60 s at k {ks}, but the best gain lies inside the grid at every level "
                       f"(rule 6 done: the curve turns down below the best point)")
    if (arm, 1.0) in res and not res[(arm, 1.0)]["d"]["flat"]:
        d_ref = res[(arm, 1.0)]["d"]["right_grid"]
        worst = []
        for k in lv:
            ds, sd, g = res[(arm, k)]["curve"]
            if d_ref not in ds:
                continue
            reg = (max(g) - g[ds.index(d_ref)]) / max(g)
            worst.append(reg)
            out.append(f"      R at k {k:g}: B2's plateau edge D1 {d_ref:g} s keeps {100 * (1 - reg):.0f}% of the best "
                       f"re-planning gain ({g[ds.index(d_ref)]:+.2f} of {max(g):+.2f} points)")
        if worst:
            out.append(f"   {arm} R: {'PASS' if max(worst) <= X_PRIMARY else 'FAIL'} (worst regret "
                       f"{100 * max(worst):.0f}% of the best gain; PASS = within X = {100 * X_PRIMARY:.0f}% at every "
                       f"non-flat level)")
    return out


# ── Axis 6 (README §13) ─────────────────────────────────────────────────────────────────────────────────────────

_GRID = {}


def grid_ref(arm, k, seed):
    """The floor-size grid's own MDD-PTECT window tardiness (1000 job-s) for this instance, or None: m50 / m100 from
    ../rq3-size-oracle (seeds 0-39), the fleet levels from ../rq3-size-headroom (seeds 0-19), m20 / m30 from
    ../rq3-size-grid's pdr runs (seeds 0-39)."""
    res = HERE.parent
    if arm == "fleet":
        f = res / "rq3-size-headroom" / "out" / "m50" / {0.35: "r035", 0.8: "r080"}[k] / "v5" / f"s{seed}.json"
        return json.loads(f.read_text())["fixed"]["MDD-PTECT"] if f.exists() else None
    m = int(k)
    if m in (50, 100):
        f = res / "rq3-size-oracle" / "out" / f"m{m}" / f"s{seed}.json"
        return json.loads(f.read_text())["fixed"]["MDD-PTECT"] if f.exists() else None
    if m not in _GRID:
        fs = sorted((res / "rq3-size-grid" / f"m{m}").glob("pdr_s*-*/episodes.csv"))
        e = pd.concat([pd.read_csv(f, dtype={"config_hash": str, "instance_hash": str}) for f in fs]) if fs else None
        _GRID[m] = {} if e is None else {int(s): float(v) / 1000.0 for s, v in
                                        e[e.policy == "MDD-PTECT"][["seed", "window_tardiness"]].values}
    return _GRID[m].get(seed)


def check_c2(data, fixed):
    """C2 (instances): every axis 6 task's MDD-PTECT run equals its level's fixed-pair task (same instance), and the
    fixed-pair task equals the floor-size grid's own value where one exists, within 1e-6 relative. m15 (B2) and fleet
    r 0.52 (size m50) are other levels' tasks and are not counted twice."""
    rows, ok, n_in, n_grid, n_bad = [], True, 0, 0, 0
    for arm in ("size", "fleet"):
        for k in LEVELS[arm]:
            if (arm, k) in (("size", 15.0), ("fleet", 0.52)) or (arm, k) not in fixed:
                continue
            fx = fixed[(arm, k)]
            for s, x in sorted(fx.items()):
                v = x["fixed"]["MDD-PTECT"]
                for d1, by_s in sorted(data.get((arm, k), {}).items()):
                    if s in by_s:
                        n_in += 1
                        if abs(by_s[s]["mdd_ptect"] / v - 1.0) > 1e-6:
                            ok = False
                            n_bad += 1
                            if n_bad <= 10:
                                    rows.append(f"   {label(arm, k)} d1 {d1:g} s{s}: task MDD-PTECT {by_s[s]['mdd_ptect']:.6f} "
                                            f"vs fixed-pair task {v:.6f}: FAIL")
                g = grid_ref(arm, k, s)
                if g is None:
                    continue
                n_grid += 1
                same = abs(v / g - 1.0) < 1e-6
                ok &= same
                n_bad += not same
                if (not same and n_bad <= 10) or n_grid <= 4:
                    rows.append(f"   {label(arm, k)} s{s}: MDD-PTECT {v:.6f} vs the grid's {g:.6f}: "
                                f"{'PASS' if same else 'FAIL'}")
    rows.append(f"   {n_grid} instances compared with the grid, {n_in} planner tasks with their fixed-pair task; "
                f"{n_bad} mismatches{' (the first 10 listed)' if n_bad > 10 else ''}")
    return ok, rows


def size_verdict(lo, hi):
    if -SLOPE_MARGIN < lo and hi < SLOPE_MARGIN:
        return "PER-MACHINE RATE (90% CI inside ±0.25): Δ / p̄ carries across sizes; m leaves the fit (§13 rule 2)"
    if FLOOR_RATE - SLOPE_MARGIN < lo and hi < FLOOR_RATE + SLOPE_MARGIN:
        return ("FLOOR-LEVEL RATE (90% CI inside −1 ± 0.25): decisions per interval Δ m u / p̄ carry; the fit's size "
                "variable (§13 rule 3)")
    if lo > 0 or hi < 0:
        return "MOVES Δ* (90% CI excludes 0): log m enters the fit as a primary variable (§13 rule 4)"
    return "INDETERMINATE: log m enters the fit as a covariate, R decides (§13 rule 5)"


def reading6(arm, data, res, rng):
    """Axis 6 readings (README §13). size: S_m = slope of log2(Δ*/p̄) on log2 m over the non-flat levels (per-machine
    rate 0, floor-level rate −1), also over the grid's floors alone (m20-m100), with the transport attribution bound from
    axis 1's proc arm; fleet: the slope is reported, not read (R decides). Both: R (B2's grid edge keeps 1 − X of the
    best gain at every non-flat level), R120 (D1 120 s does), upper bounds at the finest point (rule 9), and T_c / Δ."""
    lv = [k for k in LEVELS[arm] if (arm, k) in res and not res[(arm, k)]["d"]["flat"]]
    flat = [k for k in LEVELS[arm] if (arm, k) in res and res[(arm, k)]["d"]["flat"]]
    out = [f"   {arm}: flat levels (re-planning gain < {G_MIN} point, any Δ will do; rule 7): "
           f"{', '.join(label(arm, k) for k in flat)}"] if flat else []
    if len(lv) >= 3:
        seeds = sorted(set.intersection(*(set(res[(arm, k)]["seeds"]) for k in lv)))
        idx = rng.integers(0, len(seeds), (BOOT, len(seeds)))      # joint: a seed gives related instances at every size
        pb = {k: res[(arm, k)]["desc"]["p_mean_plan"] for k in lv}
        boots = {k: boot_dstar(data[(arm, k)], res[(arm, k)]["base"], seeds, X_PRIMARY, rng, idx=idx) for k in lv}

        def slope(ks, ds):
            return float(np.polyfit(np.log2(ks), np.log2([ds[k] / pb[k] for k in ks]), 1)[0])
        sets = [("all levels", lv)] + ([("grid floors only, m20-m100", [k for k in lv if k != 15.0])]
                                       if arm == "size" else [])
        for name, ks in sets:
            if len(ks) < 3:
                out.append(f"   {arm} S ({name}): {len(ks)} non-flat levels; needs >= 3")
                continue
            point = slope(ks, {k: res[(arm, k)]["d"]["dstar"] for k in ks})
            bs = [slope(ks, {k: boots[k][b]["dstar"] for k in ks}) for b in range(BOOT)
                  if all(boots[k][b]["dstar"] is not None for k in ks)]
            lo, hi = np.percentile(bs, 5), np.percentile(bs, 95)
            verdict = size_verdict(lo, hi) if arm == "size" else "reported, not read (README §13: R decides the fleet arm)"
            if arm == "size" and name != "all levels":
                verdict = "sensitivity: " + verdict.split(" (")[0]
            out.append(f"   {arm} S ({name}): slope of log2(Δ*/p̄) on log2 {PREFIX[arm]} = {point:+.3f} [90% {lo:+.3f}, "
                       f"{hi:+.3f}] over {PREFIX[arm]} {', '.join(f'{k:g}' for k in ks)} (seeds {len(seeds)}, {len(bs)} "
                       f"resamples) -> {verdict}")
            if arm == "size" and name == "all levels":
                tp = [res[(arm, k)]["desc"]["transport_mean"] / pb[k] for k in ks]
                st = float(np.polyfit(np.log2(ks), np.log2(tp), 1)[0])
                bound = PROC_TAU_BOUND * abs(st)
                beyond = hi < -bound or lo > bound
                out.append(f"      attribution: τ / p̄ {tp[0]:.3f} -> {tp[-1]:.3f} ({st:+.2f} octaves per octave of m); axis 1's "
                           f"proc bound lets transport alone move the slope by at most ±{bound:.2f}: the CI "
                           f"{'lies beyond it (more than transport alone)' if beyond else 'overlaps it'}")
    else:
        out.append(f"   {arm}: {len(lv)} non-flat levels; S needs >= 3, R decides (rule 7)")
    at_end = [k for k in lv if res[(arm, k)]["d"]["best_d"] == res[(arm, k)]["curve"][0][0]]
    right = [k for k in lv if res[(arm, k)]["d"]["cens_right"]]
    if at_end:
        out.append(f"      best gain at the finest point, D1 {res[(arm, at_end[0])]['curve'][0][0]:g} s, at "
                   f"{', '.join(label(arm, k) for k in at_end)}: Δ* an upper bound there (rule 9)")
    if right:
        out.append(f"      Δ* right-censored at the coarsest point (rule 6: add D1 7200 s): "
                   f"{', '.join(label(arm, k) for k in right)}")
    ref = res.get(("proc", 1.0))
    if ref is not None and not ref["d"]["flat"]:
        d_ref = ref["d"]["right_grid"]
        for name, dd in (("R", d_ref), ("R120", D_R120)):
            kept = {}
            for k in lv:
                ds, _, g = res[(arm, k)]["curve"]
                if dd in ds:
                    kept[k] = g[ds.index(dd)] / max(g)
            if not kept:
                continue
            ok = min(kept.values()) >= 1.0 - X_PRIMARY
            what = f"B2's plateau edge D1 {d_ref:g} s" if name == "R" else f"D1 {dd:g} s"
            out.append(f"   {arm} {name}: {what} keeps " + ", ".join(f"{100 * v:.1f}% ({label(arm, k)})"
                                                                for k, v in kept.items())
                       + f" of the best gain -> {'PASS' if ok else 'FAIL'} (>= {100 * (1 - X_PRIMARY):.0f}% at every "
                         f"non-flat level)" + (" -> §13 rule 6: on-duty AGV utilization "
                                              + ("leaves the fit" if ok else "enters the fit as a covariate")
                                              if arm == "fleet" and name == "R" else ""))
    for k in LEVELS[arm]:
        if (arm, k) not in res:
            continue
        by_d = data[(arm, k)]
        tc = {d: np.mean([by_d[d][s]["plan_seconds"] / by_d[d][s]["replans"] for s in by_d[d]]) / d for d in by_d}
        out.append(f"      T_c / Δ {label(arm, k)}: " + "  ".join(f"{d:g}: {tc[d]:.3f}" for d in sorted(tc)
                                                                if d in (60, 120, 300, 900)))
    return out


# ── Axis 2 (README §15) ─────────────────────────────────────────────────────────────────────────────────────────

def check_c3(data, fixed):
    """C3 (instances): every axis 2 planner task's MDD-PTECT run equals its cell's fixed-pair task within 1e-6 relative,
    the c levels of each (ℓ, block, mix, seed) share one job set (the fixed-pair tasks' jobs_hash), and the tasks equal
    the local 11-04 dry run (dryrun2/out: the 6 fixed pairs, and D1 900 s, seeds 0-4) where both exist, which catches a
    cluster copy that builds other instances. The centres are axis 1's tasks (C1) and are not counted again."""
    rows, ok, n_in, n_bad, groups = [], True, 0, 0, {}
    dry, n_dry, bad_dry = HERE / "dryrun2" / "out", 0, 0
    for k in LEVELS["ax2"]:
        if (k[0], k[1], k[3]) == (1.0, 2.125, "b2") or ("ax2", k) not in fixed:
            continue
        for s, x in sorted(fixed[("ax2", k)].items()):
            v = x["fixed"]["MDD-PTECT"]
            if "jobs_hash" in x:
                groups.setdefault((k[0], k[2], k[3], s), {})[k[1]] = x["jobs_hash"]
            for f, mine, key in ((dry / f"{ax2_name(k)}-fixed" / f"s{s}.json", x, "fixed"),
                                 (dry / f"{ax2_name(k)}-d900" / f"s{s}.json", data.get(("ax2", k), {}).get(900.0, {}).get(s), "real")):
                if mine is None or not f.exists():
                    continue
                o = json.loads(f.read_text())
                a, b = ([mine["fixed"][q] for q in sorted(o["fixed"])], [o["fixed"][q] for q in sorted(o["fixed"])]) \
                    if key == "fixed" else ([mine["real"]], [o["real"]])
                n_dry += 1
                if not all(math.isclose(u, w, rel_tol=1e-6, abs_tol=1e-9) for u, w in zip(a, b)):
                    ok, bad_dry = False, bad_dry + 1
                    if bad_dry <= 10:
                        rows.append(f"   {label('ax2', k)} s{s}: {'fixed pairs' if key == 'fixed' else 'D1 900 s'} differ "
                                    f"from the local dry run (dryrun2): FAIL")
            for d1, by_s in sorted(data.get(("ax2", k), {}).items()):
                if s in by_s:
                    n_in += 1
                    if not math.isclose(by_s[s]["mdd_ptect"], v, rel_tol=1e-6, abs_tol=1e-9):  # tardiness may be 0
                        ok, n_bad = False, n_bad + 1
                        if n_bad <= 10:
                            rows.append(f"   {label('ax2', k)} d1 {d1:g} s{s}: task MDD-PTECT {by_s[s]['mdd_ptect']:.6f} "
                                        f"vs fixed-pair task {v:.6f}: FAIL")
    multi = {g: h for g, h in groups.items() if len(h) > 1}
    split = {g: h for g, h in multi.items() if len(set(h.values())) > 1}
    ok &= not split
    for (ell, b, mix, s), h in list(split.items())[:10]:
        rows.append(f"   ℓ {ell:g} block {b} {mix} s{s}: job sets differ across c ({h}): FAIL")
    rows.append(f"   {n_in} planner tasks compared with their fixed-pair task, {n_bad} mismatches"
                f"{' (the first 10 listed)' if n_bad > 10 else ''}; {len(multi)} (ℓ, block, mix, seed) groups with "
                f"two or more c levels, {len(split)} with different job sets; {n_dry} tasks compared with the local dry "
                f"run, {bad_dry} differ")
    return ok, rows


def reading2(data, res, rng):
    """Axis 2 readings (README §15). Per block length, the centre cell's own grid edge must keep 1 - X of the best gain
    at the other levels of each line through the centre (R_ℓ: ℓ 0.7 / 1.3 at c 2.125; R_c: c 1.5 / 3 at ℓ 1; R_mix:
    nosurge / noshort); a factor leaves §8's fit only if its line passes at both block lengths. Reported: R (B2's grid
    edge) and R120 at every cell, the upper bounds (rule 9), the slopes of log2(Δ*/p̄) on log2 u (measured machine busy
    share) and log2 c over each block's 3 x 3 cells (joint seed bootstrap), and T_c / Δ."""
    cells = [k for k in LEVELS["ax2"] if ("ax2", k) in res]
    lv = [k for k in cells if not res[("ax2", k)]["d"]["flat"]]
    flat = [k for k in cells if res[("ax2", k)]["d"]["flat"]]
    out = [f"   flat cells (re-planning gain < {G_MIN} point, any Δ will do; rule 7): "
           f"{', '.join(label('ax2', k) for k in flat)}"] if flat else []

    def kept(k, dd):
        ds, _, g = res[("ax2", k)]["curve"]
        return g[ds.index(dd)] / max(g) if dd in ds else None
    verdict = {}
    for b in (5400, 1350):
        centre = (1.0, 2.125, b, "b2")
        if ("ax2", centre) not in res or res[("ax2", centre)]["d"]["flat"]:
            out.append(f"   block {b}: centre cell missing or flat; R_ℓ / R_c / R_mix not read")
            continue
        edge = res[("ax2", centre)]["d"]["right_grid"]
        for name, line in AX2_LINES.items():
            ks = [(ell, c, b, mix) for ell, c, mix in line]
            got = {k: kept(k, edge) for k in ks if k in lv}
            got = {k: v for k, v in got.items() if v is not None}
            missing = [k for k in ks if ("ax2", k) not in res]
            if missing or not got:
                out.append(f"   block {b} {name}: incomplete ({', '.join(label('ax2', k) for k in missing) or 'no edge'})")
                continue
            ok = min(got.values()) >= 1.0 - X_PRIMARY
            verdict.setdefault(name, []).append(ok)
            out.append(f"   block {b} {name}: the centre's grid edge D1 {edge:g} s keeps "
                       + ", ".join(f"{100 * v:.1f}% ({label('ax2', k)})" for k, v in got.items())
                       + f" -> {'PASS' if ok else 'FAIL'} (>= {100 * (1 - X_PRIMARY):.0f}% at every non-flat cell)")
    what = {"R_ℓ": ("load", "u (the measured machine busy share)"), "R_c": ("c", "the mean c"),
            "R_mix": ("the regime type", "the surge and short-fleet time shares")}
    for name, vs in verdict.items():
        if len(vs) < 2:
            continue
        if all(vs):
            out.append(f"   {name}: PASS at both block lengths -> {what[name][0]} leaves §8's fit (§15 rules 2-4)")
        else:
            where = "both block lengths" if not any(vs) else "one block length"
            out.append(f"   {name}: FAIL at {where} -> covariate in §8's fit: {what[name][1]} (§15 rules 2-4)")
    ref = res.get(("proc", 1.0))
    if ref is not None and not ref["d"]["flat"]:
        d_ref = ref["d"]["right_grid"]
        for name, dd in (("R", d_ref), ("R120", D_R120)):
            got = {k: kept(k, dd) for k in lv}
            got = {k: v for k, v in got.items() if v is not None}
            if got:
                what_d = f"B2's grid edge D1 {dd:g} s" if name == "R" else f"D1 {dd:g} s"
                out.append(f"   {name} ({what_d}, reported): " + ", ".join(f"{ax2_name(k)} {100 * v:.1f}%"
                                                                         for k, v in got.items())
                           + f" -> {'PASS' if min(got.values()) >= 1 - X_PRIMARY else 'FAIL'}")
    at_end = [k for k in lv if res[("ax2", k)]["d"]["best_d"] == res[("ax2", k)]["curve"][0][0]]
    right = [k for k in lv if res[("ax2", k)]["d"]["cens_right"]]
    if at_end:
        out.append(f"      best gain at the finest point run (D1 60 s on the full grid) at {len(at_end)} of {len(lv)} "
                   f"cells ({', '.join(ax2_name(k) for k in at_end)}): Δ* an upper bound there (rule 9)")
    if right:
        out.append(f"      Δ* right-censored at the coarsest point (rule 6: add D1 7200 s): "
                   f"{', '.join(ax2_name(k) for k in right)}")
    for b in (5400, 1350):
        ks = [k for k in lv if k[2] == b and k[3] == "b2"]
        if len(ks) < 5:
            out.append(f"   block {b} slopes: {len(ks)} non-flat 3 x 3 cells; needs >= 5")
            continue
        seeds = sorted(set.intersection(*(set(res[("ax2", k)]["seeds"]) for k in ks)))
        idx = rng.integers(0, len(seeds), (BOOT, len(seeds)))
        boots = {k: boot_dstar(data[("ax2", k)], res[("ax2", k)]["base"], seeds, X_PRIMARY, rng, idx=idx) for k in ks}
        pb = {k: res[("ax2", k)]["desc"]["p_mean_plan"] for k in ks}
        u = {k: res[("ax2", k)]["desc"]["machine_busy_share"] for k in ks}
        xmat = np.column_stack([np.ones(len(ks)), np.log2([u[k] for k in ks]), np.log2([k[1] for k in ks])])

        def fit(ds):
            return np.linalg.lstsq(xmat, np.log2([ds[k] / pb[k] for k in ks]), rcond=None)[0][1:]
        point = fit({k: res[("ax2", k)]["d"]["dstar"] for k in ks})
        bs = np.array([fit({k: boots[k][i]["dstar"] for k in ks}) for i in range(BOOT)
                       if all(boots[k][i]["dstar"] is not None for k in ks)])
        lo, hi = np.percentile(bs, 5, axis=0), np.percentile(bs, 95, axis=0)
        out.append(f"   block {b} slopes (reported, not read): log2(Δ*/p̄) on log2 u {point[0]:+.3f} [90% {lo[0]:+.3f}, "
                   f"{hi[0]:+.3f}] and on log2 c {point[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}] over {len(ks)} cells "
                   f"(u = machine busy share {min(u.values()):.2f}-{max(u.values()):.2f}; seeds {len(seeds)}, {len(bs)} "
                   f"resamples; upper-bound cells at their bound)")
    for k in cells:
        by_d = data[("ax2", k)]
        tc = {d: np.mean([by_d[d][s]["plan_seconds"] / by_d[d][s]["replans"] for s in by_d[d]]) / d for d in by_d}
        out.append(f"      T_c / Δ {ax2_name(k)}: " + "  ".join(f"{d:g}: {tc[d]:.3f}" for d in sorted(tc)
                                                            if d in (60, 120, 300, 900)))
    return out


def main(root, rng):
    data, fixed = load(root)
    n_out = sum(len(s) for by in data.values() for s in by.values()) + sum(len(b) for b in fixed.values())
    print(f"== rq2-replan-axes: {n_out} task outputs under {root.name}/out; X = {100 * X_PRIMARY:.0f}% "
          f"(sensitivity {', '.join(f'{100 * x:.0f}%' for x in X_SENS)}); gain over plan-once")
    ok0, r0 = check_c0(data)
    ok1, r1 = check_c1(data)
    print(f"\n-- C0 units (the `all` arm): {'PASS' if ok0 else 'FAIL'}" + ("" if r0 else " (no tasks yet)"))
    print("\n".join(r0))
    print(f"\n-- C1 regression vs rq2-replan: {'PASS' if ok1 else 'FAIL'}")
    print("\n".join(r1))
    res, curve_rows, dstar_rows = {}, [], []
    rng2 = np.random.default_rng(AX2_SEED)
    print("\n-- gain curves per B2-equivalent interval D1: re-planning gain over plan-once [% vs MDD-PTECT]")
    for (arm, k), by_d in sorted(data.items(), key=lambda kv: (ARMS.index(kv[0][0]), LEVELS["ax2"].index(kv[0][1])
                                                                if kv[0][0] == "ax2" else kv[0][1])):
        if arm == "all" or not by_d:
            continue
        fx = fixed.get((arm, k), {})
        base = {s: x["best"] for s, x in fx.items()}
        ds, seeds, g = level_curve(by_d, base)
        if len(ds) < 3 or not seeds:
            print(f"   {label(arm, k)}: {len(ds)} intervals, {len(seeds)} common seeds with a plan-once baseline; skipped")
            continue
        _, _, gm = level_curve(by_d, None, seeds)
        desc = pd.DataFrame([fx[s]["describe"] for s in seeds]).mean(numeric_only=True).to_dict()
        d = dstar(ds, g, X_PRIMARY)
        sens = {x: dstar(ds, g, x) for x in X_SENS}
        res[(arm, k)] = {"curve": (ds, seeds, g), "d": d, "desc": desc, "seeds": seeds, "base": base}
        scale = k if arm == "proc" else 1.0
        pbar = desc["p_mean_plan"]
        once = gain({s: {"real": base[s], "mdd_ptect": fx[s]["fixed"]["MDD-PTECT"]} for s in seeds}, seeds, None)
        pairs = pd.Series([fx[s]["best_pair"] for s in seeds]).value_counts()
        a6, a2 = arm in ("size", "fleet"), arm == "ax2"
        busy = (f"τ / p̄ {desc['transport_mean'] / pbar:.3f}, AGV busy on duty "
                + (f"{desc['agv_busy_on_duty']:.2f}" if desc.get("agv_busy_on_duty") is not None else "-")
                + f" (all bays {desc['agv_busy_share']:.2f}), jobs {desc['jobs_window']:.0f}") if a6 else \
            (f"machine busy {desc['machine_busy_share']:.2f}, AGV busy on duty "
             + (f"{desc['agv_busy_on_duty']:.2f}" if desc.get("agv_busy_on_duty") is not None else "-")
             + f" (all bays {desc['agv_busy_share']:.2f}), jobs {desc['jobs_window']:.0f}, plan-once tardiness "
             f"{sum(base[s] for s in seeds):.0f} (1000 job-s, summed)") if a2 else \
            f"AGV busy {desc['agv_busy_share']:.2f}"
        print(f"   {label(arm, k)} (n {len(seeds)}, p̄ {pbar:.0f} s, transport {desc['transport_mean']:.0f} s, {busy}; "
              f"plan-once {once:+.2f}% vs MDD-PTECT, best pairs {', '.join(f'{p} {n}' for p, n in pairs.items())}):")
        print("      " + "  ".join(f"{dd:g}: {gg:+.2f} [{hh:+.2f}]" for dd, gg, hh in zip(ds, g, gm)))
        for dd, gg, hh in zip(ds, g, gm):
            rows = by_d[dd]
            curve_rows.append({"arm": arm, "k": ax2_name(k) if a2 else k, "d1": dd, "delta_s": dd * scale, "delta_over_p": dd * scale / pbar,
                               "gain_over_plan_once_%": gg, "gain_vs_mdd_ptect_%": hh, "n": len(seeds),
                               "replans": float(np.mean([rows[s]["replans"] for s in seeds])),
                               "switches": float(np.mean([rows[s]["switches"] for s in seeds])),
                               "plan_seconds": float(np.mean([rows[s]["plan_seconds"] for s in seeds])),
                               "task_seconds": float(np.mean([rows[s]["seconds"] for s in seeds])),
                               **({"t_c_over_delta": float(np.mean([rows[s]["plan_seconds"] / rows[s]["replans"]
                                                                    for s in seeds])) / (dd * scale)} if a6 or a2 else {})})
        if d["flat"]:
            print(f"      flat: best re-planning gain {d['g_best']:+.2f} < {G_MIN} point")
            continue
        bt = boot_dstar(by_d, base, seeds, X_PRIMARY, rng2 if a2 else rng)
        lo, med, hi = q([b["dstar"] for b in bt])
        print(f"      Δ* {d['dstar']:.0f} s B2-eq = {d['dstar'] * scale:.0f} s = {d['dstar'] * scale / pbar:.2f} p̄ "
              f"[boot 90% {lo * scale / pbar:.2f}-{hi * scale / pbar:.2f} p̄]; best D1 {d['best_d']:g} ({d['g_best']:+.2f}),"
              f" plateau edge on the grid {d['right_grid']:g}, width {d['width']:.1f}x"
              f"{' (right-censored)' if d['cens_right'] else ''}{' (left-censored)' if d['cens_left'] else ''}; "
              + ", ".join(f"X {100 * x:.0f}%: " + (f"{sens[x]['dstar'] * scale / pbar:.2f} p̄" if sens[x]["dstar"] else "-")
                          for x in X_SENS))
        dstar_rows.append({"arm": arm, "k": ax2_name(k) if a2 else k, "n": len(seeds), "p_mean": pbar, "transport_s": desc["transport_mean"],
                           "agv_busy": desc["agv_busy_share"], "machine_busy": desc["machine_busy_share"],
                           "block_s": desc.get("block_seconds"), "plan_once_vs_mdd_ptect_%": once,
                           "g_best_%": d["g_best"], "best_d1": d["best_d"],
                           "dstar_d1": d["dstar"], "dstar_s": d["dstar"] * scale, "dstar_over_p": d["dstar"] * scale / pbar,
                           "boot_lo_over_p": lo * scale / pbar, "boot_hi_over_p": hi * scale / pbar,
                           "right_grid_d1": d["right_grid"], "width": d["width"], "cens_right": d["cens_right"],
                           "cens_left": d["cens_left"],
                           **{f"dstar_over_p_x{int(100 * x)}": (sens[x]["dstar"] or 0) * scale / pbar for x in X_SENS},
                           **({"tau_over_p": desc["transport_mean"] / pbar, "agv_busy_on_duty": desc.get("agv_busy_on_duty"),
                               "jobs_window": desc["jobs_window"]} if a6 or a2 else {}),
                           **({"ell": k[0], "c": k[1], "block": k[2], "mix": k[3]} if a2 else {})})
    print("\n-- readings (README §6): S = shape (slope), R = transfer regret of B2's interval")
    for arm in ("proc", "reg"):
        print("\n".join(reading(arm, data, res, rng)))
    if axis6(data):
        ok2, r2 = check_c2(data, fixed)
        print(f"\n-- axis 6 (README §13): C2 instances = the floor-size grid's: {'PASS' if ok2 else 'FAIL'}")
        print("\n".join(r2))
        print("\n-- axis 6 readings (README §13): S_m = slope on log2 m, R = B2's grid edge, R120 = D1 120 s; "
              "T_c / Δ = single-core planning time of one re-plan over the interval")
        for arm in ("size", "fleet"):
            print("\n".join(reading6(arm, data, res, rng)))
    if axis2(data):
        ok3, r3 = check_c3(data, fixed)
        print(f"\n-- axis 2 (README §15): C3 instances: {'PASS' if ok3 else 'FAIL'}")
        print("\n".join(r3))
        print("\n-- axis 2 readings (README §15): R_ℓ / R_c / R_mix = the centre cell's own grid edge along each line "
              "through the centre, per block length; R = B2's grid edge, R120 = D1 120 s (reported); slopes reported")
        print("\n".join(reading2(data, res, rng2)))
    pd.DataFrame(curve_rows).to_csv(root / "curve.csv", index=False)
    pd.DataFrame(dstar_rows).to_csv(root / "dstar.csv", index=False)


def preview(rng):
    """The estimator on rq2-replan's 1a curve (exact forecast, B2, seeds 0-39, 1-60 min): X sensitivity for both
    baselines, and the bootstrap spread of Δ* with 40 seeds and with 20 (resampling 20 of the 40). Plan-once there =
    exact-d300-ev2 (no event ever fires with the exact forecast: one plan at the warm-up, held)."""
    by_d, once = {}, {}
    for f in RQ2.glob("exact-d*/s*.json"):
        d = json.loads(f.read_text())
        if d.get("trigger") is None:
            by_d.setdefault(float(d["slot_seconds"]), {})[d["seed"]] = d
        elif d["trigger"] == "ev2":
            assert d["replans"] == 1, d
            once[d["seed"]] = d["real"]
    for bname, base in (("plan-once (primary)", once), ("MDD-PTECT (rq2-replan's reference)", None)):
        ds, seeds, g = level_curve(by_d, base)
        print(f"== preview: rq2-replan 1a (B2, exact forecast), {len(seeds)} seeds; gain over {bname}")
        print("   " + "  ".join(f"{d:g}: {x:+.2f}" for d, x in zip(ds, g)))
        sub = {d: by_d[d] for d in ds if d in D1_GRID}
        dsub, _, gsub = level_curve(sub, base, seeds)
        for name, (dd, gg, bd) in {"rq2-replan grid (with 60, 1200 s)": (ds, g, by_d),
                                   "this study's grid (no 60, 1200; 450 s not run there)": (dsub, gsub, sub)}.items():
            print(f"-- {name}")
            for x in (0.05, 0.10, 0.15, 0.20, 0.30):
                d = dstar(dd, gg, x)
                line = (f"   X {100 * x:2.0f}%: Δ* {d['dstar']:6.0f} s = {d['dstar'] / 355.4:.2f} p̄, grid edge "
                        f"{d['right_grid']:g}, width {d['width']:.1f}x")
                for n in (40, 20):
                    bt = boot_dstar(bd, base, seeds, x, rng, n=n)
                    lo, med, hi = q([b["dstar"] for b in bt])
                    cells = pd.Series([b["right_grid"] for b in bt]).value_counts(normalize=True).sort_index()
                    line += (f" | n {n}: boot 90% {lo:.0f}-{hi:.0f} s, edge " +
                             ", ".join(f"{k:g} {100 * v:.0f}%" for k, v in cells.items() if v >= 0.05))
                print(line)
        print()
    print("   (p̄ = 355 s: seed 0's planned mean operation time; the study measures it per level)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    root = (HERE / a.root).resolve()
    buf = io.StringIO()
    with redirect_stdout(buf):
        preview(rng) if a.preview else main(root, rng)
    (HERE / "preview.out" if a.preview else root / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
