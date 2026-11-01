"""
@file analyze.py
@brief rq3-transfer-map (2026-10-25, user: remaining work plan 10-23 §2a): one edge-retention measure across every kind
       of drift the final policy family (train-v5-cmix s0-s4, trained on B2) has been evaluated on. Analysis only: it
       reads the evaluation runs already made, through each source experiment's own loaders.
  edge       % window tardiness vs MDD-PTECT held all episode, pooled over test seeds (< 0 better); the family is the
             per-instance mean of its 5 seeds (the sources' convention); 95% seed-bootstrap CI
  retention  edge at the condition / edge on B2 (twin, family, seeds 0-39: -1.69%); paired seed bootstrap (the same
             seed indices drawn for both). Twin axes only: in Unity the B2 edge is not below 0 (s3 +0.5% on seeds
             0-39, +3.6% on 0-19), so a Unity ratio has no meaning.
  gap        family (or s3) vs the condition's best v5 pair, chosen in-sample on the same seeds
  shift      how far the condition moves the rule landscape from B2's: (1 - Kendall tau) / 2 between the 6 v5 pairs'
             pooled % vs MDD-PTECT there and on B2 (0 same order, 1 reversed), and the mean |change| in points
Axes (sources): floor size (../rq3-size-grid, ../rq3-size-v5; arms T50 / T100 / Tmix), due dates / load / fleet /
layout (../rq3-gen-v5, ../rq3-gen-v5-cmix, ../eval-v5-cmix), noise (../rq4-uncert-v5-policy, E3 / mix cells on the
screen's measure as there), AGV breakdowns (Unity, ../rq4-agvfail-v5-cmix), machine flexibility (Unity, s3 only,
../rq3-flex-v5-unity), and the Unity subset (s3, ../rq3-gen-v5-unity-cmix). Twin unless marked.
-> analysis.out, transfer_map.csv, transfer_map.png
@par Usage
@code{.sh}
python results/rq3-transfer-map/analyze.py
@endcode
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
R = HERE.parent
NBOOT = 10000
REF = "MDD-PTECT"
V5 = ["SRT-PECT", "SRT-PTECT", "MDD-PECT", "MDD-PTECT", "ATC-PECT", "ATC-PTECT"]
FAM = [f"cmix s{k}" for k in range(5)]
S3 = ["cmix s3"]
ARMS = {"T50": ("train-size-m50", 50), "T100": ("train-size-m100", 100), "Tmix": ("train-size-mix", None)}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


size = module("size_v5", R / "rq3-size-v5" / "analyze.py")          # name() / table() / floor_table()
sys.path.insert(0, str(R / "rq4-uncert-v5-policy"))
unc = module("unc_v5", R / "rq4-uncert-v5-policy" / "analyze.py")   # cell_table() / screen_measure()
T_unc = unc.T
agv = module("agv_cmix", R / "rq4-agvfail-v5-cmix" / "analyze.py")  # load_cmix(), rq4.load()
flex = module("flex_v5", R / "rq3-flex-v5-unity" / "analyze.py")    # no_flex(), u.table, rename()
usub = module("usub", R / "rq3-gen-v5-unity-cmix" / "analyze.py")   # rename(), u.table


def arm_name(p):
    p = size.name(p)
    for arm, (run, _) in ARMS.items():
        if p.startswith(run + "_s"):
            return f"{arm} s{p.rsplit('_s', 1)[1]}"
    return p


def grid_floor(m):
    """Every policy on floor m: T15 (cmix) and the pairs from the sources the grid uses, the arms from the grid."""
    if m in (15, 50, 100):
        t = size.floor_table(f"m{m}")
    else:
        d = R / "rq3-size-grid" / f"m{m}"
        t = size.table(sorted(d.glob("pdr_s*-*/episodes.csv")) + sorted(d.glob("cmix_s*-*/episodes.csv")))
    arms = sorted((R / "rq3-size-grid" / f"m{m}").glob("size*_s*-*/episodes.csv"))
    if arms:
        fr = [pd.read_csv(f, dtype=size.H) for f in arms if (f.parent / "DONE").exists()]
        e = pd.concat(fr, ignore_index=True)
        e["policy"] = e.policy.map(arm_name)
        e = e[e.seed.isin(size.SEEDS)].drop_duplicates(["policy", "seed"])
        a = e.pivot_table(index="seed", columns="policy", values="window_tardiness") / 1000.0
        assert (a[REF] - t[REF]).abs().max() < 1e-6, f"m{m}: the arms' MDD-PTECT differs from the pairs'"
        t = t.join(a.drop(columns=[REF]), how="left")
    return t


def gen_cond(c):
    if c == "base":
        return size.floor_table("m15")
    cm = (R / "eval-v5-cmix" / "test" / c / "cmix" / "episodes.csv") if c.startswith("due-") else \
        (R / "rq3-gen-v5-cmix" / c / "cmix" / "episodes.csv")
    return size.table([R / "rq3-gen-v5" / c / "pdr" / "episodes.csv", cm])


def unity_agv(level):
    d = agv.rq4.load()
    d = d[d.level == level].drop_duplicates(["policy", "seed"])
    t = d.pivot_table(index="seed", columns="policy", values="tard")
    c = agv.load_cmix()
    c = c[(c.level == level) & c.policy.isin(FAM)].drop_duplicates(["policy", "seed"])
    return t.join(c.pivot_table(index="seed", columns="policy", values="tard"), how="left")


def unity_flex(level):
    if level == "none":
        return flex.no_flex()
    fr = {g: flex.rename(flex.u.table(sorted((R / "rq3-flex-v5-unity" / "runs" / level / g).glob("s*/episodes.csv"))))
          for g in ("fa", "fb", "fc")}
    return fr["fa"].join(fr["fb"], how="outer").join(fr["fc"].drop(columns=[REF]), how="outer")


def unity_subset(c):
    if c == "base":
        return flex.no_flex()
    tu = usub.rename(usub.u.table(sorted((R / "rq3-gen-v5-unity" / "runs" / c).glob("p[ab]/s*/episodes.csv"))))
    pc = usub.rename(usub.u.table(sorted((R / "rq3-gen-v5-unity-cmix" / "runs" / c / "pc").glob("s*/episodes.csv"))))
    return tu.join(pc.drop(columns=[REF]), how="outer")


class Stat:
    """Edge / gap / retention of a policy (mean over `cols` per instance) on table t, with a bootstrap over seeds."""

    def __init__(self, rng):
        self.rng = rng

    def edge(self, t, cols, vs=REF, base=None, base_cols=None):
        t = t.dropna(subset=cols + [vs])
        f, r = t[cols].mean(axis=1), t[vs]
        out = {"n": len(t), "edge": 100 * (f.sum() / r.sum() - 1), "W": int((f < r - 1e-9).sum()),
               "L": int((f > r + 1e-9).sum())}
        d = (f - r)[(f - r).abs() > 1e-9]
        out["p"] = stats.wilcoxon(d).pvalue if len(d) >= 5 else float("nan")
        idx = self.rng.integers(0, len(t), size=(NBOOT, len(t)))
        bs = 100 * (f.values[idx].sum(1) / r.values[idx].sum(1) - 1)
        out["lo"], out["hi"] = np.percentile(bs, [2.5, 97.5])
        out["per_seed"] = [100 * (t[c].sum() / r.sum() - 1) for c in cols] if len(cols) > 1 else []
        if base is not None:
            b = base.dropna(subset=base_cols + [REF])
            common = sorted(set(t.index) & set(b.index))
            fc, rc = f.loc[common].values, r.loc[common].values
            fb, rb = b.loc[common, base_cols].mean(axis=1).values, b.loc[common, REF].values
            e_b = 100 * (fb.sum() / rb.sum() - 1)
            out["retention"] = (100 * (fc.sum() / rc.sum() - 1)) / e_b
            idx = self.rng.integers(0, len(common), size=(NBOOT, len(common)))
            num = 100 * (fc[idx].sum(1) / rc[idx].sum(1) - 1)
            den = 100 * (fb[idx].sum(1) / rb[idx].sum(1) - 1)
            ok = den < 0
            out["ret_lo"], out["ret_hi"] = np.percentile(num[ok] / den[ok], [2.5, 97.5])
            out["ret_den_neg"] = ok.mean()
        return out


def landscape(t):
    pairs = [p for p in V5 if p in t.columns]
    tt = t.dropna(subset=pairs)
    return pd.Series({p: 100 * (tt[p].sum() / tt[REF].sum() - 1) for p in pairs})


def best_pair(t):
    pairs = [p for p in V5 if p in t.columns]
    return t.dropna(subset=pairs)[pairs].sum().idxmin()


def fmt(v, lo, hi):
    star = "**" if (lo > 0 or hi < 0) else ""
    return f"{star}{v:+.2f}{star} [{lo:+.2f}, {hi:+.2f}]"


def main():
    st = Stat(np.random.default_rng(0))
    B = size.floor_table("m15")                       # twin B2, seeds 0-39: size m15 = rq3 base = uncertainty none
    B_land = landscape(B)
    rows = []

    def add(axis, cond, x, sim, t, policy, cols, retention, xlabel=None):
        e = st.edge(t, cols, base=B if retention else None, base_cols=FAM if retention else None)
        bp = best_pair(t)
        g = st.edge(t, cols, vs=bp)
        row = {"axis": axis, "condition": cond, "x": x, "x_label": xlabel or str(x), "sim": sim, "policy": policy,
               "n": e["n"], "edge": e["edge"], "edge_lo": e["lo"], "edge_hi": e["hi"], "W": e["W"], "L": e["L"],
               "p": e["p"], "per_seed": " / ".join(f"{v:+.1f}" for v in e["per_seed"]),
               "retention": e.get("retention", np.nan), "ret_lo": e.get("ret_lo", np.nan),
               "ret_hi": e.get("ret_hi", np.nan), "best_pair": bp, "gap": g["edge"], "gap_lo": g["lo"],
               "gap_hi": g["hi"], "mdd_ptect_mean": t[REF].mean()}
        if sim == "twin":
            land = landscape(t)
            common = [p for p in V5 if p in land.index]
            tau = stats.kendalltau(B_land[common], land[common])[0]
            row["shift_tau"] = (1 - tau) / 2
            row["shift_l1"] = float((land[common] - B_land[common]).abs().mean())
        pairs = [p for p in V5 if p in t.columns]
        if len(pairs) == 6:                                   # room: the per-instance best v5 pair in hindsight
            tt = t.dropna(subset=pairs + cols)
            row["room"] = 100 * (tt[pairs].min(axis=1).sum() / tt[REF].sum() - 1)
            row["capture"] = row["edge"] / row["room"] if row["room"] < 0 else np.nan
        rows.append(row)

    # floor size: T15 family + s3, the arms' families (twin, seeds 0-39)
    for m in (15, 20, 30, 50, 75, 100, 150, 200):
        t = grid_floor(m)
        add("floor size", f"m{m}", m, "twin", t, "T15 family", FAM, True)
        add("floor size", f"m{m}", m, "twin", t, "T15 s3", S3, False)
        for arm in ARMS:
            mem = [f"{arm} s{k}" for k in range(5)]
            if all(c in t.columns for c in mem) and t[mem].notna().all().all():
                add("floor size", f"m{m}", m, "twin", t, f"{arm} family", mem, False)
    # due dates, load, fleet, layout (twin, seeds 0-39)
    GEN = [("due dates", "due-tight", 1.5, "c 1.25-1.75"), ("due dates", "base", 2.125, "c 1.75-2.5 (B2)"),
           ("due dates", "due-loose", 3.0, "c 2.5-3.5"), ("load", "load-0.8", 0.8, "x 0.8"),
           ("load", "base", 1.0, "x 1.0 (B2)"), ("load", "load-1.2", 1.2, "x 1.2"),
           ("fleet", "agv5-bay9", 5, "5 AGVs, 9 bays"), ("fleet", "agv7-bay9", 7, "7 AGVs, 9 bays"),
           ("fleet", "agv9-bay9", 9, "9 AGVs, 9 bays"), ("layout", "lay-c", "C", "C"),
           ("layout", "base", "D", "D (B2)"), ("layout", "lay-g", "G", "G"), ("layout", "lay-j", "J", "J")]
    for axis, c, x, lab in GEN:
        t = gen_cond(c)
        add(axis, c, x, "twin", t, "T15 family", FAM, True, lab)
        add(axis, c, x, "twin", t, "T15 s3", S3, False, lab)
    # noise (twin, seeds 0-39): x = MDD-PTECT's degradation vs B2 on the same measure
    for cell in T_unc.CELLS:
        t, shifted = unc.screen_measure(cell, unc.cell_table(cell))
        deg = 100 * (t[REF].sum() / B.loc[t.index, REF].sum() - 1)
        kind, lv = cell.split(",")[0].split("=")
        lab = f"{kind} {lv}" + (" I1" if "I1" in cell else "") + (" (screen measure)" if shifted else "")
        add("noise", cell, deg, "twin", t, "T15 family", FAM, True, lab)
        add("noise", cell, deg, "twin", t, "T15 s3", S3, False, lab)
    # AGV breakdowns (Unity, seeds 0-39): x = failures per AGV-hour, 3600 / (lambda Gamma(1 + 1/1.5))
    for lv, lam in (("none", None), ("l3000", 3000.0), ("l1500", 1500.0)):
        t = unity_agv(lv)
        rate = 0.0 if lam is None else 3600.0 / (lam * math.gamma(1 + 1 / 1.5))
        lab = "none" if lam is None else f"lambda {lam:.0f} s"
        add("AGV breakdowns", lv, rate, "Unity", t, "T15 family", FAM, False, lab)
        add("AGV breakdowns", lv, rate, "Unity", t, "T15 s3", S3, False, lab)
    # machine flexibility (Unity, s3 only, seeds 0-19)
    for lv, p in (("none", 0.0), ("flex-p0.3", 0.3), ("flex-p0.5", 0.5)):
        add("flexibility", lv, p, "Unity", unity_flex(lv), "T15 s3", S3, False, f"p {p}")
    # Unity subset (s3, seeds 0-19), next to the twin on the same seeds
    for c in ("base", "load-1.2", "due-tight", "lay-g"):
        add("Unity subset", c, c, "Unity", unity_subset(c), "T15 s3", S3, False)
        tw = gen_cond(c)
        add("Unity subset", c, c, "twin (seeds 0-19)", tw.loc[tw.index < 20], "T15 s3", S3, False)

    D = pd.DataFrame(rows)
    D.to_csv(HERE / "transfer_map.csv", index=False)
    report(D)
    plot(D)


def report(D):
    print("rq3-transfer-map: the final policy family (train-v5-cmix s0-s4, trained on B2) under every measured drift.")
    print("edge = % window tardiness vs MDD-PTECT held, pooled (< 0 better; family = per-instance mean of 5 seeds); "
          "95% seed-bootstrap CI, ** = excludes 0; retention = edge / B2 edge (twin family, paired bootstrap); "
          "gap = vs the condition's best v5 pair (in-sample).\n")
    for axis, g in D.groupby("axis", sort=False):
        print(f"## {axis}")
        print("| condition | sim | policy | n | edge [95% CI] | W/L | retention [95% CI] | best pair | gap to it | "
              "training seeds | shift (tau / L1) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            ret = "" if np.isnan(r.retention) else f"{r.retention:.2f} [{r.ret_lo:.2f}, {r.ret_hi:.2f}]"
            sh = "" if np.isnan(r.get("shift_tau", np.nan)) else f"{r.shift_tau:.2f} / {r.shift_l1:.1f}"
            print(f"| {r.x_label} | {r.sim} | {r.policy} | {r.n} | {fmt(r.edge, r.edge_lo, r.edge_hi)} | "
                  f"{r.W}/{r.L} | {ret} | {r.best_pair} | {fmt(r.gap, r.gap_lo, r.gap_hi)} | {r.per_seed} | {sh} |")
        print()
    fam = D[(D.policy == "T15 family") & (D.sim == "twin")].drop_duplicates(["axis", "condition"])
    fam = fam[fam.condition != "base"].copy()
    print("## All twin conditions (T15 family, B2 excluded): does the edge fall with how far the rule landscape moved?")
    for k in ("shift_tau", "shift_l1"):
        rho, p = stats.spearmanr(fam[k], fam.edge)
        print(f"- edge vs {k}: Spearman rho {rho:+.2f} (p {p:.2g}, {len(fam)} conditions)")
    ns = fam[fam.axis != "floor size"]
    for k in ("shift_tau", "shift_l1"):
        rho, p = stats.spearmanr(ns[k], ns.edge)
        print(f"- without the floor-size axis, edge vs {k}: rho {rho:+.2f} (p {p:.2g}, {len(ns)} conditions)")
    print("\n## ... or with how much room the condition leaves (per-instance best v5 pair in hindsight vs MDD-PTECT; "
          "capture = edge / room)")
    for lab, g in (("all twin conditions", fam), ("without floor size", ns), ("floor size only", fam[fam.axis == "floor size"])):
        rho, p = stats.spearmanr(g.room, g.edge)
        print(f"- {lab}: edge vs room Spearman rho {rho:+.2f} (p {p:.2g}, {len(g)} conditions)")
    b2 = D[(D.axis == "floor size") & (D.condition == "m15") & (D.policy == "T15 family")].iloc[0]
    print(f"- B2: room {b2.room:+.2f}%, edge {b2.edge:+.2f}%, capture {b2.capture:.2f}")
    print("| condition | axis | room | edge | capture |")
    print("|---|---|---|---|---|")
    for _, r in fam.sort_values("room").iterrows():
        print(f"| {r.x_label} | {r.axis} | {r.room:+.2f} | {r.edge:+.2f} | {r.capture:+.2f} |")
    sig_better = fam[fam.edge_hi < 0]
    sig_worse = fam[fam.edge_lo > 0]
    print(f"- significantly better than MDD-PTECT: {len(sig_better)} / {len(fam)} conditions; significantly worse: "
          f"{len(sig_worse)} ({', '.join(sig_worse.condition)})")
    print(f"- retention >= 1 (edge at least B2's): {int((fam.retention >= 1).sum())} / {len(fam)}; "
          f"retention < 0 (edge lost): {int((fam.retention < 0).sum())} ({', '.join(fam[fam.retention < 0].condition)})")


AXIS_COLORS = {"floor size": "#1f77b4", "due dates": "#ff7f0e", "load": "#2ca02c", "fleet": "#d62728",
               "layout": "#9467bd", "noise": "#8c564b"}


def plot(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 3, figsize=(14, 12))
    b2 = D[(D.axis == "floor size") & (D.condition == "m15") & (D.policy == "T15 family")].edge.iloc[0]

    def frame(ax, title, xlabel, twin=True):
        ax.axhline(0, color="black", lw=0.8)
        if twin:
            ax.axhline(b2, color="grey", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("edge, % vs MDD-PTECT (< 0 better)", fontsize=8)
        ax.tick_params(labelsize=8)

    def series(ax, g, label, color, ls="-", marker="o", band=True, x=None):
        x = g.x.astype(float).values if x is None else x
        ax.plot(x, g.edge, marker=marker, ls=ls, color=color, label=label, ms=4)
        if band:
            ax.fill_between(x, g.edge_lo, g.edge_hi, color=color, alpha=0.12)

    # 1 floor size
    ax = axs[0, 0]
    g = D[D.axis == "floor size"]
    series(ax, g[g.policy == "T15 family"], "T15 family (trained on 15)", "#1f77b4")
    series(ax, g[g.policy == "T15 s3"], "T15 s3 (final policy)", "#1f77b4", ls=":", marker=".", band=False)
    for arm, col in (("T50", "#ff7f0e"), ("T100", "#2ca02c"), ("Tmix", "#9467bd")):
        series(ax, g[g.policy == f"{arm} family"], f"{arm} family", col, band=False)
    ax.set_xscale("log")
    ax.set_xticks([15, 20, 30, 50, 75, 100, 150, 200])
    ax.set_xticklabels(["15", "20", "30", "50", "75", "100", "150", "200"])
    frame(ax, "Floor size (twin)", "machines (log scale)")
    ax.legend(fontsize=7)
    # 2-4 due dates, load, fleet
    for ax, axis, title, xl in ((axs[0, 1], "due dates", "Due-date tightness (twin)", "allowance c, block midpoint"),
                                (axs[0, 2], "load", "Load (twin)", "load multiplier on B2"),
                                (axs[1, 0], "fleet", "Fleet on a 9-bay floor (twin)", "AGVs on duty")):
        g = D[D.axis == axis]
        series(ax, g[g.policy == "T15 family"], "T15 family", AXIS_COLORS[axis])
        series(ax, g[g.policy == "T15 s3"], "T15 s3", AXIS_COLORS[axis], ls=":", marker=".", band=False)
        frame(ax, title, xl)
        ax.legend(fontsize=7)
    axs[0, 1].axvspan(1.25, 3.5, color="grey", alpha=0.08)
    axs[0, 1].text(2.6, axs[0, 1].get_ylim()[0] + 0.4, "shaded: cmix training c range 1.25-3.5", fontsize=7,
                   color="grey")
    # 5 layout (categorical)
    ax = axs[1, 1]
    g = D[(D.axis == "layout")]
    for pol, off, col in (("T15 family", -0.1, AXIS_COLORS["layout"]), ("T15 s3", 0.1, "#c5b0d5")):
        h = g[g.policy == pol]
        xs = np.arange(len(h)) + off
        ax.errorbar(xs, h.edge, yerr=[h.edge - h.edge_lo, h.edge_hi - h.edge], fmt="o", color=col, capsize=3,
                    label=pol, ms=4)
    ax.set_xticks(range(len(g[g.policy == "T15 family"])))
    ax.set_xticklabels(g[g.policy == "T15 family"].x_label, fontsize=8)
    frame(ax, "Layout (twin)", "layout (B2 = D)")
    ax.legend(fontsize=7)
    # 6 noise
    ax = axs[1, 2]
    g = D[(D.axis == "noise") & (D.policy == "T15 family")]
    for _, r in g.iterrows():
        sev = "severe" in r.condition
        i1 = "I1" in r.condition
        ax.errorbar(r.x, r.edge, yerr=[[r.edge - r.edge_lo], [r.edge_hi - r.edge]], fmt="s" if sev else "o",
                    mfc="white" if i1 else AXIS_COLORS["noise"], color=AXIS_COLORS["noise"], capsize=2, ms=4,
                    alpha=0.8)
        ax.annotate(r.condition.split("=")[0], (r.x, r.edge), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=10)
    frame(ax, "Noise (twin, T15 family; o realistic, s severe, hollow I1)", "MDD-PTECT tardiness increase vs B2, %")
    # 7 AGV breakdowns (Unity)
    ax = axs[2, 0]
    g = D[D.axis == "AGV breakdowns"]
    series(ax, g[g.policy == "T15 family"], "T15 family", "#e377c2")
    series(ax, g[g.policy == "T15 s3"], "T15 s3", "#e377c2", ls=":", marker=".", band=False)
    frame(ax, "AGV breakdowns (Unity, seeds 0-39)", "failures per AGV-hour", twin=False)
    ax.legend(fontsize=7)
    # 8 flexibility + Unity subset (Unity, s3)
    ax = axs[2, 1]
    g = D[D.axis == "flexibility"]
    series(ax, g, "flexibility, T15 s3", "#7f7f7f")
    frame(ax, "Machine flexibility (Unity, s3, seeds 0-19)", "P(a machine also runs another type)", twin=False)
    ax.legend(fontsize=7)
    # 9 all twin conditions: edge vs the room the condition leaves
    ax = axs[2, 2]
    g = D[(D.policy == "T15 family") & (D.sim == "twin")].drop_duplicates(["axis", "condition"])
    for axis, h in g.groupby("axis"):
        ax.errorbar(h.room, h.edge, yerr=[h.edge - h.edge_lo, h.edge_hi - h.edge], fmt="o",
                    color=AXIS_COLORS.get(axis, "black"), capsize=2, ms=4, label=axis, alpha=0.8)
    lim = min(g.room.min(), g.edge_lo.min()) - 0.5
    ax.plot([lim, 0], [lim, 0], color="grey", lw=0.7, ls=":")
    ax.text(lim + 0.3, lim + 1.2, "edge = room", fontsize=7, color="grey", rotation=33)
    frame(ax, "All twin conditions (T15 family)", "room: per-instance best v5 pair in hindsight, % vs MDD-PTECT")
    ax.legend(fontsize=7)
    fig.suptitle("Transfer map: the final policy family's edge over MDD-PTECT as each condition drifts from B2 "
                 "(dashed: its B2 edge; bands / bars: 95% seed-bootstrap CI)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "transfer_map.png", dpi=150)


class _Tee:
    """stdout and analysis.out at once."""

    def __init__(self, path):
        self.f, self.out = open(path, "w"), sys.stdout

    def write(self, s):
        self.f.write(s)
        self.out.write(s)

    def flush(self):
        self.f.flush()
        self.out.flush()


if __name__ == "__main__":
    sys.stdout = _Tee(HERE / "analysis.out")
    main()
    sys.stdout.flush()
