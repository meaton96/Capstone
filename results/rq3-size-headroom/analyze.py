"""
@file analyze.py
@brief rq3-size-headroom analysis (pre-declared 10-25 in README.md): per cell (floor size x AGVs per machine), the v5
       and v5 + MDD-TECT switching ceilings vs MDD-PTECT, what MDD-TECT adds, the oracle's switching, and the load /
       transport descriptors under MDD-PTECT; then the two pre-declared tests (averaging, transport scarcity).
       Checks first: the job stream is the same at every ratio; MDD-PTECT / MDD-TECT held in the desc runs equal the
       oracles' slot-0 tails (reused r 0.52 outputs of rq3-size-oracle / rq3-size-portfolio included).
       Writes analysis.out (stdout), summary.csv (one row per cell) and instances.csv (one row per cell x seed).
       Works on a partial run (cells and seeds with missing files are reported and left out).
@par Usage
@code{.sh}
python results/rq3-size-headroom/analyze.py          # also writes analysis.out
@endcode
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("hr_run", HERE / "run.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)              # constants and paths only (the frozen code loads in run.load, not here)
SIZES = sorted(run.SIZES)
RATIOS = list(run.RATIOS)
RNG = np.random.default_rng(0)
NBOOT = 10000


def oracle_file(size, ratio, name, seed):
    f = run.HERE / "out" / f"m{size}" / ratio / name / f"s{seed}.json"
    return f if f.exists() else run.reused(size, ratio, name, seed)


def load_cell(size, ratio):
    rows = []
    for s in run.SEEDS:
        fd = run.HERE / "out" / f"m{size}" / ratio / "desc" / f"s{s}.json"
        fo = {n: oracle_file(size, ratio, n, s) for n in run.SETS}
        if not fd.exists() or not all(fo.values()):
            continue
        d = json.loads(fd.read_text())
        o = {n: json.loads(f.read_text()) for n, f in fo.items()}
        rows.append({"seed": s, "desc": d, "oracle": o})
    return rows


def pooled(a, b):
    """100 (sum a / sum b - 1), with a paired seed-bootstrap 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pt = 100.0 * (a.sum() / b.sum() - 1.0)
    idx = RNG.integers(0, len(a), size=(NBOOT, len(a)))
    bs = 100.0 * (a[idx].sum(1) / b[idx].sum(1) - 1.0)
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fmt(p):
    pt, lo, hi = p
    star = "**" if (lo > 0 or hi < 0) else ""
    return f"{star}{pt:+.2f}{star} [{lo:+.2f}, {hi:+.2f}]"


def wilcoxon(x):
    x = np.asarray(x, float)
    x = x[x != 0]
    return stats.wilcoxon(x).pvalue if len(x) >= 5 else float("nan")


def slot_stats(slots):
    """Per-instance descriptors from the per-slot rows of one held-rule run."""
    L = np.array([r["load_per_machine"] for r in slots])
    sdm = np.array([r["load_sd_machines"] / r["load_per_machine"] if r["load_per_machine"] > 0 else 0.0
                    for r in slots])
    reg = np.array([r["regime"] for r in slots])
    rated = [r for r in slots if r.get("agv_util") is not None]
    w = np.array([rr["t"] for rr in slots[1:]]) - np.array([rr["t"] for rr in slots[:-1]])
    au = np.array([r["agv_util"] for r in rated])
    mu = np.array([r["machine_util"] for r in rated])
    out = {"load_mean": L.mean(), "load_cv_time": L.std() / L.mean() if L.mean() > 0 else float("nan"),
           "load_cv_machines": sdm.mean(), "agv_util": float((au * w).sum() / w.sum()),
           "machine_util": float((mu * w).sum() / w.sum()),
           "wait_per_agv": float(np.mean([r["wait_per_agv"] for r in slots])),
           "path_per_trip": float(np.nanmean([r["path_per_trip"] or np.nan for r in rated])),
           "trips_per_h_per_machine": float(np.nanmean([r["trips_per_h"] or np.nan for r in rated])) / slots[0]["n_agv"],
           "short_share": float((reg == "short").mean())}
    for g in ("surge", "short"):
        if (reg == g).any() and (reg == "normal").any():
            out[f"contrast_{g}"] = (L[reg == g].mean() - L[reg == "normal"].mean()) / L.mean()
        else:
            out[f"contrast_{g}"] = float("nan")
    return out


def instances():
    rows, bad, missing = [], 0, []
    hashes = {}
    for m in SIZES:
        for r in RATIOS:
            cell = load_cell(m, r)
            missing.append((m, r, len(cell)))
            for e in cell:
                d, o, s = e["desc"], e["oracle"], e["seed"]
                hashes.setdefault((m, s), {})[r] = d["jobs_hash"]
                for n in run.SETS:
                    if "jobs_hash" in o[n] and o[n]["jobs_hash"] != d["jobs_hash"]:
                        bad += 1
                        print(f"CHECK FAIL m{m} {r} s{s}: {n} oracle instance hash differs from desc")
                pt = d["rules"]["MDD-PTECT"]["tard"]
                tt = d["rules"]["MDD-TECT"]["tard"]
                for n, rule, v in (("v5", "MDD-PTECT", pt), ("v5tect", "MDD-PTECT", pt), ("v5tect", "MDD-TECT", tt)):
                    ref = o[n]["fixed"][rule]
                    if abs(ref - v) > 1e-6 * max(1.0, abs(v)) + 1e-4:
                        bad += 1
                        print(f"CHECK FAIL m{m} {r} s{s}: {rule} desc {v:.6f} vs {n} oracle slot-0 tail {ref:.6f}")
                row = {"size": m, "ratio": r, "r": run.RATIOS[r], "seed": s, "mddptect": pt, "mddtect": tt}
                for n in run.SETS:
                    sch = o[n]["schedule"]
                    tails = np.array(o[n]["tails"])
                    row[f"oracle_{n}"] = o[n]["oracle"]
                    row[f"hindsight_{n}"] = min(o[n]["fixed"].values())
                    row[f"switches_{n}"] = sum(a != b for a, b in zip(sch, sch[1:]))
                    row[f"spread_{n}"] = float(np.mean((tails.max(1) - tails.min(1)) / tails.min(1)))
                row["tect_share"] = float(np.mean([p == "MDD-TECT" for p in o["v5tect"]["schedule"]]))
                for k, v in slot_stats(d["rules"]["MDD-PTECT"]["slots"]).items():
                    row[k] = v
                for k, v in slot_stats(d["rules"]["MDD-TECT"]["slots"]).items():
                    if k in ("agv_util", "load_cv_time", "wait_per_agv"):
                        row[f"tect_{k}"] = v
                rows.append(row)
    for (m, s), h in hashes.items():
        if len(set(h.values())) > 1:
            bad += 1
            print(f"CHECK FAIL m{m} s{s}: job stream differs across ratios {h}")
    return pd.DataFrame(rows), bad, missing


def main():
    df, bad, missing = instances()
    print("rq3-size-headroom analysis (B2 test seeds; window tardiness, k job-s; % vs MDD-PTECT held, pooled, < 0 better;"
          " 95% paired seed-bootstrap CI, ** = excludes 0)\n")
    print("seeds per cell (desc + both oracles present): " +
          ", ".join(f"m{m}/{r} {n}" for m, r, n in missing))
    print(f"checks: {'all passed' if not bad else f'{bad} FAILED'} (job stream equal across ratios; held rules = "
          "oracle slot-0 tails)\n")
    if df.empty:
        return
    cells = []
    for (m, r), g in df.groupby(["size", "ratio"], sort=False):
        on, short = run.fleet(m, r)
        c = {"size": m, "ratio": r, "r": run.RATIOS[r], "agvs": on, "short_agvs": short, "n": len(g),
             "mddptect_mean": g.mddptect.mean(), "mddptect_per_machine": g.mddptect.mean() / m}
        for n in run.SETS:
            p = pooled(g[f"oracle_{n}"], g.mddptect)
            c.update({f"oracle_{n}": p[0], f"oracle_{n}_lo": p[1], f"oracle_{n}_hi": p[2]})
            c[f"hindsight_{n}"] = pooled(g[f"hindsight_{n}"], g.mddptect)[0]
            c[f"saving_{n}"] = (g.mddptect - g[f"oracle_{n}"]).mean()
            c[f"switches_{n}"] = g[f"switches_{n}"].mean()
            c[f"spread_{n}"] = 100 * g[f"spread_{n}"].mean()
        p = pooled(g.oracle_v5tect, g.oracle_v5)
        c.update({"tect_adds": p[0], "tect_adds_lo": p[1], "tect_adds_hi": p[2]})
        c["mddtect_vs_mddptect"] = pooled(g.mddtect, g.mddptect)[0]
        c["tect_share"] = 100 * g.tect_share.mean()
        for k in ("agv_util", "machine_util", "load_mean", "load_cv_time", "load_cv_machines", "contrast_surge",
                  "contrast_short", "wait_per_agv", "path_per_trip", "short_share", "tect_agv_util"):
            c[k] = g[k].mean()
        cells.append(c)
    S = pd.DataFrame(cells).sort_values(["size", "r"])
    S.to_csv(HERE / "summary.csv", index=False)
    df.to_csv(HERE / "instances.csv", index=False)

    print("## Ceilings per cell")
    print("| floor | AGVs on duty (short) | n | MDD-PTECT mean | v5 oracle | v5 + MDD-TECT oracle | MDD-TECT adds "
          "(vs v5 oracle) | MDD-TECT held | hindsight best v5 / v5+TECT | saving per instance v5 / v5+TECT |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for _, c in S.iterrows():
        print(f"| m{c['size']} r{c.r:.2f} | {c.agvs} ({c.short_agvs}) | {c.n} | {c.mddptect_mean:,.1f} | "
              f"{fmt((c.oracle_v5, c.oracle_v5_lo, c.oracle_v5_hi))} | "
              f"{fmt((c.oracle_v5tect, c.oracle_v5tect_lo, c.oracle_v5tect_hi))} | "
              f"{fmt((c.tect_adds, c.tect_adds_lo, c.tect_adds_hi))} | {c.mddtect_vs_mddptect:+.2f} | "
              f"{c.hindsight_v5:+.2f} / {c.hindsight_v5tect:+.2f} | {c.saving_v5:.1f} / {c.saving_v5tect:.1f} |")
    print("\n## Oracle behaviour and descriptors (MDD-PTECT held; means over instances)")
    print("| floor | switches v5 / v5+TECT | per-slot tail spread v5 % | TECT slot share % | AGV util | AGV util "
          "under MDD-TECT | machine util | backlog / machine (s) | backlog CV over slots | backlog CV over machines | "
          "surge contrast | short contrast | waiting per AGV | path per trip (m) | short-fleet slots |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, c in S.iterrows():
        print(f"| m{c['size']} r{c.r:.2f} | {c.switches_v5:.1f} / {c.switches_v5tect:.1f} | {c.spread_v5:.2f} | "
              f"{c.tect_share:.0f} | {c.agv_util:.2f} | {c.tect_agv_util:.2f} | {c.machine_util:.2f} | "
              f"{c.load_mean:.0f} | {c.load_cv_time:.2f} | {c.load_cv_machines:.2f} | {c.contrast_surge:+.2f} | "
              f"{c.contrast_short:+.2f} | {c.wait_per_agv:.2f} | {c.path_per_trip:.0f} | {100 * c.short_share:.0f}% |")

    def per_inst(g, n):
        return 100.0 * (g[f"oracle_{n}"] / g.mddptect - 1.0)

    print("\n## Test A, averaging (pre-declared): at each ratio, (a) backlog CV over slots falls from m50 to m200 "
          "(Mann-Whitney, one-sided) and (b) the v5 + MDD-TECT ceiling still shrinks from m50 to m200 (per-instance "
          "%, Mann-Whitney, one-sided: m200 less negative). Supported if (a) and (b) hold at r 0.80.")
    verdict = {}
    for r in RATIOS:
        a, b = df[(df["size"] == SIZES[0]) & (df.ratio == r)], df[(df["size"] == SIZES[-1]) & (df.ratio == r)]
        if len(a) < 5 or len(b) < 5:
            print(f"- r{run.RATIOS[r]:.2f}: too few instances ({len(a)} / {len(b)})")
            continue
        pa = stats.mannwhitneyu(a.load_cv_time, b.load_cv_time, alternative="greater").pvalue
        pb = stats.mannwhitneyu(per_inst(b, "v5tect"), per_inst(a, "v5tect"), alternative="greater").pvalue
        verdict[r] = (pa < 0.05, pb < 0.05)
        print(f"- r{run.RATIOS[r]:.2f}: backlog CV m{SIZES[0]} {a.load_cv_time.mean():.2f} -> m{SIZES[-1]} "
              f"{b.load_cv_time.mean():.2f} (p {pa:.3g}); v5+TECT ceiling per instance m{SIZES[0]} "
              f"{per_inst(a, 'v5tect').median():+.2f}% -> m{SIZES[-1]} {per_inst(b, 'v5tect').median():+.2f}% median "
              f"(p {pb:.3g})")
    if "r080" in verdict:
        print(f"  -> averaging {'SUPPORTED' if all(verdict['r080']) else 'NOT supported'} at r 0.80 "
              f"(a {verdict['r080'][0]}, b {verdict['r080'][1]})")

    print("\n## Test B, transport scarcity (pre-declared): on m100 and m200, (c) the v5 ceiling at r 0.80 is larger than "
          "at r 0.52 (same instances, per-instance %, Wilcoxon) and (d) what MDD-TECT adds over the v5 oracle is "
          "larger at r 0.35 than at r 0.80 (paired, Wilcoxon). Supported if (c) and (d) hold on both floors.")
    tb = []
    for m in SIZES[1:]:
        g = df[df["size"] == m].pivot(index="seed", columns="ratio")
        need = [("oracle_v5", x) for x in RATIOS] + [("mddptect", x) for x in RATIOS] + [("oracle_v5tect", x)
                                                                                         for x in RATIOS]
        if any(k not in g.columns for k in need):
            print(f"- m{m}: ratios incomplete")
            continue
        g = g.dropna(subset=need)
        if len(g) < 5:
            print(f"- m{m}: too few paired instances ({len(g)})")
            continue
        v5 = {x: 100 * (g[("oracle_v5", x)] / g[("mddptect", x)] - 1) for x in RATIOS}
        adds = {x: 100 * (g[("oracle_v5tect", x)] / g[("oracle_v5", x)] - 1) for x in RATIOS}
        dc = v5["r080"] - v5["r052"]
        dd = adds["r035"] - adds["r080"]
        pc, pd_ = wilcoxon(dc), wilcoxon(dd)
        c_ok, d_ok = dc.median() < 0 and pc < 0.05, dd.median() < 0 and pd_ < 0.05
        tb.append(c_ok and d_ok)
        print(f"- m{m} (n {len(g)}): v5 ceiling per instance, median r0.35 / 0.52 / 0.80: "
              f"{v5['r035'].median():+.2f} / {v5['r052'].median():+.2f} / {v5['r080'].median():+.2f}%; "
              f"(c) r0.80 - r0.52 median {dc.median():+.2f} pts, p {pc:.3g} -> {c_ok}. MDD-TECT adds, median "
              f"r0.35 / 0.52 / 0.80: {adds['r035'].median():+.2f} / {adds['r052'].median():+.2f} / "
              f"{adds['r080'].median():+.2f}%; (d) r0.35 - r0.80 median {dd.median():+.2f} pts, p {pd_:.3g} -> {d_ok}")
    if len(tb) == len(SIZES) - 1:
        print(f"  -> transport scarcity {'SUPPORTED' if all(tb) else 'NOT supported'} (m100 {tb[0]}, m200 {tb[1]})")

    print("\n## How much of the size drop does a larger fleet recover? (pooled v5 ceiling, pts)")
    for m in SIZES[1:]:
        try:
            base = S[(S["size"] == SIZES[0]) & (S.ratio == "r052")].oracle_v5.iloc[0]
            at = {x: S[(S["size"] == m) & (S.ratio == x)].oracle_v5.iloc[0] for x in RATIOS}
        except IndexError:
            continue
        drop = at["r052"] - base
        rec = (at["r052"] - at["r080"]) / drop if drop > 0 else float("nan")
        print(f"- m{m}: m{SIZES[0]} r0.52 {base:+.2f}% -> m{m} r0.52 {at['r052']:+.2f}% (drop {drop:+.2f}); "
              f"r0.80 {at['r080']:+.2f}%: recovers {100 * rec:.0f}% of the drop; r0.35 {at['r035']:+.2f}%")

    print("\n## Instance level: Spearman of the v5 oracle gain (per-instance %, < 0 better) with the descriptors, "
          "within each floor size (all ratios pooled)")
    for m in SIZES:
        g = df[df["size"] == m]
        if len(g) < 8:
            continue
        gain = per_inst(g, "v5")
        parts = []
        for k in ("agv_util", "load_cv_time", "load_cv_machines", "wait_per_agv", "spread_v5", "switches_v5"):
            rho, p = stats.spearmanr(gain, g[k], nan_policy="omit")
            parts.append(f"{k} {rho:+.2f} (p {p:.2g})")
        print(f"- m{m} (n {len(g)}): " + "; ".join(parts))


class _Tee:
    """stdout and analysis.out at once (the lab shows stdout in its finished event)."""

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
