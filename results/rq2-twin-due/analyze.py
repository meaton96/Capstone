"""
@file analyze.py
@brief rq2-twin-due tables: twin calibration against Unity, switching headroom by objective / pair set / allowance,
       fixed-pair rankings on tardiness, and what the oracle schedules use.

@details Gains are reductions (positive = better than the reference) of the oracle's own objective:
  static_%   best-on-average pair's mean gap to the per-seed best fixed pair
  switch_%   oracle vs the per-seed best fixed pair (switching value alone)
  total_%    oracle vs the best-on-average pair (what a policy could hope to beat the best fixed rule by)
Writes headroom.csv, fixed_gaps.csv, calibration.csv to analysis/ (analysis_<x>/ for tasks_<x>/) and prints
       the tables.

@par Usage
@code{.sh}
.venv/bin/python results/rq2-twin-due/analyze.py [tasks_oldsel]
@endcode
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TASKS = HERE / "tasks"          # main() takes another task folder as argv[1], e.g. tasks_oldsel
OUT = HERE / "analysis"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def load():
    tasks = [json.loads(p.read_text()) for p in sorted(TASKS.glob("*.json"))]
    stamps = {t.get("code", {}).get("id", "unstamped") for t in tasks}
    heads = {t.get("code", {}).get("git_head", "?") for t in tasks}
    print(f"code stamps: {sorted(stamps)} at {sorted(heads)}")
    if len(stamps) > 1:
        print("WARNING: tasks from more than one code version are mixed in this folder")
    meta, fixed = [], []
    for t in tasks:
        obj = t["objective"]
        final_rows = [r for r in t["rows"] if r["stage"] == len(t["schedule"])]
        final = min(final_rows, key=lambda r: r[obj])
        meta.append({k: t[k] for k in ("setting", "seed", "objective", "pairset", "allowance", "best_fixed_pair",
                                       "best_fixed", "oracle", "warmup", "jobs")}
                    | {"schedule": "|".join(t["schedule"]), "oracle_tis": final["tis"], "oracle_tard": final["tard"],
                       "oracle_jobs_exited": final["jobs_exited"]})
        for pair, v in t["fixed"].items():
            fixed.append({"setting": t["setting"], "seed": t["seed"], "objective": obj, "pairset": t["pairset"],
                          "allowance": t["allowance"], "pair": pair, **v})
    return pd.DataFrame(meta), pd.DataFrame(fixed)


def headroom(meta, fixed):
    out = []
    for key, g in meta.groupby(["setting", "objective", "pairset", "allowance"]):
        setting, obj, pairset, c = key
        f = fixed[(fixed.setting == setting) & (fixed.objective == obj) & (fixed.pairset == pairset)
                  & (fixed.allowance == c)].pivot(index="seed", columns="pair", values=obj)
        f = f.loc[g.seed.values]
        zero = f.min(axis=1) <= 1e-9            # no tardiness under some pair: relative gaps undefined
        n_zero = int(zero.sum())
        g = g[~zero.values]
        f = f[~zero]
        if f.empty:
            out.append({"setting": setting, "objective": obj, "pairset": pairset, "c": c, "seeds": 0,
                        "zero_seeds": n_zero})
            continue
        per_seed_best = f.min(axis=1)
        best_pair = f.mean().idxmin()
        orc = g.set_index("seed")["oracle"]
        static = (f[best_pair] / per_seed_best - 1) * 100
        switch = (1 - orc / per_seed_best) * 100
        total = (1 - orc / f[best_pair]) * 100
        spread = (f.max(axis=1) / per_seed_best - 1) * 100
        jobs_rules = g["schedule"].str.split("|").explode().str.split("-").str[0].value_counts(normalize=True)
        mach_rules = g["schedule"].str.split("|").explode().str.split("-").str[1].value_counts(normalize=True)
        late = fixed[(fixed.setting == setting) & (fixed.objective == obj) & (fixed.pairset == pairset)
                     & (fixed.allowance == c) & (fixed.pair == best_pair)]
        pct_tardy = 100 * late.late_exited.sum() / max(late.jobs_exited.sum(), 1)
        tis_best = fixed[(fixed.setting == setting) & (fixed.objective == "tis") & (fixed.pairset == "P12")]
        tis_best = tis_best.pivot(index="seed", columns="pair", values="tis").mean().min() if len(tis_best) else np.nan
        out.append({
            "setting": setting, "objective": obj, "pairset": pairset, "c": c, "seeds": len(g), "zero_seeds": n_zero,
            "best_pair": best_pair, "pct_tardy": pct_tardy if obj == "tard" else np.nan,
            "best_mean": f[best_pair].mean(), "static_%": static.mean(), "switch_%": switch.mean(),
            "switch_med_%": switch.median(), "total_%": total.mean(), "total_med_%": total.median(),
            "total_min_%": total.min(), "total_max_%": total.max(),
            "seeds>2%": int((total > 2).sum()), "seeds>3%": int((total > 3).sum()),
            "gain_abs": (f[best_pair] - orc).mean(), "spread_%": spread.mean(),
            "oracle_tis_vs_best_tis_%": (g["oracle_tis"].mean() / tis_best - 1) * 100 if obj == "tard" else np.nan,
            "job_rules": ", ".join(f"{k} {v:.0%}" for k, v in jobs_rules.head(4).items()),
            "machine_rules": ", ".join(f"{k} {v:.0%}" for k, v in mach_rules.head(3).items()),
        })
    return pd.DataFrame(out)


def fixed_gaps(fixed, setting="warm"):
    """Mean gap (%) of each pair to the per-seed best on tardiness, P30 tasks, by allowance; plus seed wins."""
    rows = []
    f = fixed[(fixed.setting == setting) & (fixed.objective == "tard") & (fixed.pairset == "P30")]
    for c, g in f.groupby("allowance"):
        piv = g.pivot(index="seed", columns="pair", values="tard")
        piv = piv[piv.min(axis=1) > 1e-9]       # seeds with no tardiness under some pair have no relative gap
        gap = (piv.div(piv.min(axis=1), axis=0) - 1) * 100
        wins = piv.idxmin(axis=1).value_counts()
        tis = fixed[(fixed.setting == setting) & (fixed.objective == "tard") & (fixed.pairset == "P30")
                    & (fixed.allowance == c)].pivot(index="seed", columns="pair", values="tis")
        tis = tis.loc[piv.index]
        tis_gap = (tis.div(tis.min(axis=1), axis=0) - 1) * 100
        for pair in piv.columns:
            rows.append({"c": c, "pair": pair, "tard_gap_%": gap[pair].mean(), "tard_mean": piv[pair].mean(),
                         "wins": int(wins.get(pair, 0)), "tis_gap_%": tis_gap[pair].mean()})
    return pd.DataFrame(rows)


def calibration(meta, fixed):
    """Twin (t0, tis-P12) vs Unity rq2-switch-oracle on its failure-free seeds, and warm vs Unity mf-off. Both Unity
    runs are pre-eb61a6c2 builds (old AGV selection, results/outdated/): like-for-like only for tasks_oldsel."""
    rows = []
    for seed in (0, 2, 4, 11, 15):
        p = REPO / f"results/outdated/rq2-switch-oracle/s{seed}/result.json"
        if not p.exists():
            continue
        u = json.loads(p.read_text())
        tw = fixed[(fixed.setting == "t0") & (fixed.seed == seed) & (fixed.objective == "tis") & (fixed.pairset == "P12")]
        m = meta[(meta.setting == "t0") & (meta.seed == seed) & (meta.objective == "tis") & (meta.pairset == "P12")]
        if tw.empty:
            continue
        twin = tw.set_index("pair")["tis"]
        unity = pd.Series({k: -v for k, v in u["fixed_returns"].items()})
        common = [k for k in unity.index if not k.startswith("FIFO") and k in twin.index]   # Unity run had the old FIFO
        d = (twin[common] / unity[common] - 1) * 100
        rows.append({"ref": "rq2-switch-oracle (t0)", "seed": seed, "pairs": len(common),
                     "mean_|diff|_%": d.abs().mean(), "max_|diff|_%": d.abs().max(),
                     "rank_corr": twin[common].rank().corr(unity[common].rank()),
                     "best_unity": unity[common].idxmin(), "best_twin": twin[common].idxmin(),
                     "oracle_gain_unity_%": (1 - (-u["oracle_return"]) / (-u["best_fixed_return"])) * 100,
                     "oracle_gain_twin_%": (1 - m["oracle"].iloc[0] / m["best_fixed"].iloc[0]) * 100})
    p = REPO / "results/outdated/rq2-oracle-screen-qfifo/mf-off/s0/result.json"
    tw = fixed[(fixed.setting == "warm") & (fixed.seed == 0) & (fixed.objective == "tis") & (fixed.pairset == "P12")]
    m = meta[(meta.setting == "warm") & (meta.seed == 0) & (meta.objective == "tis") & (meta.pairset == "P12")]
    if p.exists() and not tw.empty:
        u = json.loads(p.read_text())
        twin = tw.set_index("pair")["tis"]
        unity = pd.Series({k: -v for k, v in u["fixed_returns"].items()})
        common = [k for k in unity.index if k in twin.index]
        d = (twin[common] / unity[common] - 1) * 100
        rows.append({"ref": "oracle-screen-qfifo mf-off (warm)", "seed": 0, "pairs": len(common),
                     "mean_|diff|_%": d.abs().mean(), "max_|diff|_%": d.abs().max(),
                     "rank_corr": twin[common].rank().corr(unity[common].rank()),
                     "best_unity": unity[common].idxmin(), "best_twin": twin[common].idxmin(),
                     "oracle_gain_unity_%": (1 - (-u["oracle_return"]) / (-u["best_fixed_return"])) * 100,
                     "oracle_gain_twin_%": (1 - m["oracle"].iloc[0] / m["best_fixed"].iloc[0]) * 100})
    return pd.DataFrame(rows)


def per_seed(meta, fixed, setting, obj, pairset, c):
    g = meta[(meta.setting == setting) & (meta.objective == obj) & (meta.pairset == pairset) & (meta.allowance == c)]
    f = fixed[(fixed.setting == setting) & (fixed.objective == obj) & (fixed.pairset == pairset)
              & (fixed.allowance == c)].pivot(index="seed", columns="pair", values=obj)
    best_pair = f.mean().idxmin()
    g = g.set_index("seed").sort_index()
    return pd.DataFrame({"best_pair_value": f[best_pair], "per_seed_best": f.min(axis=1),
                         "per_seed_best_pair": f.idxmin(axis=1), "oracle": g["oracle"],
                         "total_%": (1 - g["oracle"] / f[best_pair]) * 100, "schedule": g["schedule"]}), best_pair


def choices_by_load(setting="warm", obj="tard", pairset="P30", c=2.0):
    """Which job rule the oracle picks per 900 s slot, by the load of the generator segment the slot falls in
    (the segment with the most overlap): lull, under 1 (utilization < 1) or overload (>= 1)."""
    import sys
    sys.path.insert(0, str(REPO / "env"))
    sys.path.insert(0, str(HERE))
    import run
    rows = []
    for p in sorted(TASKS.glob(f"{setting}_s*_{obj}_{pairset}_c{c:g}.json")):
        t = json.loads(p.read_text())
        t0 = t["rows"][0]["t0"]
        phases = run.scenario(setting, t["seed"])["_phases"]
        for k, pair in enumerate(t["schedule"]):
            a, b = t0 + k * run.SEGMENT, t0 + (k + 1) * run.SEGMENT
            ph = max(phases, key=lambda ph: max(0.0, min(b, ph["end"]) - max(a, ph["start"])))
            load = "lull" if ph["kind"] == "lull" else ("overload" if ph["utilization"] >= 1.0 else "under 1")
            rows.append({"seed": t["seed"], "slot": k, "kind": ph["kind"], "load": load, "job": pair.split("-")[0],
                         "machine": pair.split("-")[1]})
    return pd.DataFrame(rows)


def decision_table(meta, fixed, setting="warm", allowances=(1.5, 2.0), n_boot=10000, seed=0):
    """Go / no-go view: oracle gain over the best-on-average pair (total_%) per action set, with seed-bootstrap 90%
    intervals for the mean, the median and the share of seeds above 2% (best pair held at its full-sample choice)."""
    rng = np.random.default_rng(seed)
    rows = []
    sets = [("tis", "P12", 2.0)] + [("tard", ps, c) for c in allowances for ps in ("P12", "P30", "H15", "H10")]
    for obj, ps, c in sets:
        g = meta[(meta.setting == setting) & (meta.objective == obj) & (meta.pairset == ps) & (meta.allowance == c)]
        if g.empty:
            continue
        f = fixed[(fixed.setting == setting) & (fixed.objective == obj) & (fixed.pairset == ps)
                  & (fixed.allowance == c)].pivot(index="seed", columns="pair", values=obj)
        f = f[f.min(axis=1) > 1e-9]
        best = f.mean().idxmin()
        orc = g.set_index("seed")["oracle"].loc[f.index]
        tot = ((1 - orc / f[best]) * 100).values
        sw = ((1 - orc / f.min(axis=1)) * 100).values
        idx = rng.integers(0, len(tot), size=(n_boot, len(tot)))
        bt = tot[idx]
        ci = lambda a: f"[{np.percentile(a, 5):.1f}, {np.percentile(a, 95):.1f}]"
        rows.append({"objective": obj, "set": ps, "c": c, "seeds": len(tot), "best_pair": best,
                     "base": f[best].mean(), "gain_abs": (f[best] - orc).mean(),
                     "switch_med_%": np.median(sw), "total_mean_%": tot.mean(), "mean_CI90": ci(bt.mean(axis=1)),
                     "total_med_%": np.median(tot), "median_CI90": ci(np.median(bt, axis=1)),
                     "share>2%": (tot > 2).mean() * 100, "share_CI90": ci((bt > 2).mean(axis=1) * 100)})
    return pd.DataFrame(rows)


def main():
    import sys
    global TASKS, OUT
    if len(sys.argv) > 1:
        TASKS = (HERE / sys.argv[1]).resolve()
        OUT = HERE / f"analysis_{TASKS.name.removeprefix('tasks_')}"
    OUT.mkdir(exist_ok=True)
    meta, fixed = load()
    print(f"{len(meta)} tasks\n")
    cal = calibration(meta, fixed)
    cal.to_csv(OUT / "calibration.csv", index=False, float_format="%.3f")
    print("== Calibration: twin vs Unity, time in system, current 12 pairs ==")
    print(cal.round(2).to_string(index=False), "\n")
    dt = decision_table(meta, fixed)
    dt.to_csv(OUT / "decision.csv", index=False, float_format="%.3f")
    print("== Decision table (warm): oracle vs best-on-average pair, seed-bootstrap 90% intervals ==")
    print(dt.round(2).to_string(index=False), "\n")
    hr = headroom(meta, fixed)
    hr.to_csv(OUT / "headroom.csv", index=False, float_format="%.3f")
    cols = ["setting", "objective", "pairset", "c", "seeds", "zero_seeds", "best_pair", "pct_tardy", "static_%", "switch_%",
            "switch_med_%", "total_%", "total_med_%", "total_min_%", "total_max_%", "seeds>2%", "seeds>3%",
            "gain_abs", "spread_%", "oracle_tis_vs_best_tis_%"]
    print("== Headroom ==")
    print(hr[cols].round(2).to_string(index=False), "\n")
    print(hr[["setting", "objective", "pairset", "c", "job_rules", "machine_rules"]].to_string(index=False), "\n")
    fg = pd.concat([fixed_gaps(fixed, s).assign(setting=s) for s in ("warm", "t0")])
    fg.to_csv(OUT / "fixed_gaps.csv", index=False, float_format="%.3f")
    for s in ("warm", "t0"):
        print(f"== Fixed pairs on tardiness ({s}): mean gap to per-seed best, top 10 per c ==")
        for c, g in fg[fg.setting == s].groupby("c"):
            g = g.sort_values("tard_gap_%").head(10)
            print(f"-- c = {c:g}")
            print(g[["pair", "tard_gap_%", "wins", "tard_mean", "tis_gap_%"]].round(2).to_string(index=False))
        print()
    for c in sorted(meta[meta.objective == "tard"].allowance.unique()):
        ch = choices_by_load(c=c)
        if ch.empty:
            continue
        print(f"== Oracle job-rule choice by segment load (warm, tard, P30, c = {c:g}), share of slots ==")
        print((pd.crosstab(ch.load, ch.job, normalize="index") * 100).round(0).to_string(), "\n")
    for c in sorted(meta[meta.objective == "tard"].allowance.unique()):
        ps, bp = per_seed(meta, fixed, "warm", "tard", "P30", c)
        print(f"== Per seed (warm, tard, P30, c = {c:g}; best-on-average pair {bp}) ==")
        print(ps.round(3).to_string(), "\n")


if __name__ == "__main__":
    main()
