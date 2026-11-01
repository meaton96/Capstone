"""
@file analyze.py
@brief rq3-size-grid (2026-10-20): the floor-size cross-evaluation grid, twin, test seeds 0-39, window tardiness.
       Arms (5 training seeds each): T15 = the final-policy family train-v5-cmix (trained on 15 machines), T50 / T100 =
       train-size-m50 / -m100, Tmix = train-size-mix (15 + 50 + 100). Per test floor (15-200 machines):
       - the v5 pairs vs MDD-PTECT and the floor's best pair;
       - each arm's family mean of 5 vs MDD-PTECT and vs the best pair (pooled over test seeds, seed-bootstrap 95% CI,
         Wilcoxon), and the spread over training seeds (each seed's pooled %, one-sample t-test of the 5 against 0:
         the training-seed-level test);
       - Tmix vs each single-size arm, paired per test seed.
       Then the transfer curve: single-size arms' gap vs |log10(test / train size)| (Spearman).
       Run-to-run check: MDD-PTECT in every policy group = the pdr group's on the same floor and seeds.
       m15 / m50 / m100 pdr and T15 come from ../rq3-gen-v5 base, ../eval-v5-cmix/test base and ../rq3-size-v5.
       -> analysis.out, summary.csv (long), grid.csv (arm x floor family % vs MDD-PTECT)
"""
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

HERE = Path(__file__).resolve().parent
R = HERE.parent
_s = importlib.util.spec_from_file_location("size_an", R / "rq3-size-v5" / "analyze.py")
size = importlib.util.module_from_spec(_s)
_s.loader.exec_module(size)
rq3 = size.rq3
REF = rq3.REF
H = {"config_hash": str, "instance_hash": str}
FLOORS = {"m15": 15, "m20": 20, "m30": 30, "m50": 50, "m75": 75, "m100": 100, "m150": 150, "m200": 200}
ARMS = {"T15": 15, "T50": 50, "T100": 100, "Tmix": None}       # training size (None = mixed 15 / 50 / 100)
_base_name = size.name


def name(p):
    p = str(p).replace("ckpt:", "").replace("/checkpoint", "").replace("results/", "")
    for pre, lab in (("train-size-m50_s", "T50 s"), ("train-size-m100_s", "T100 s"), ("train-size-mix_s", "Tmix s"),
                     ("train-v5-cmix_s", "T15 s")):
        if p.startswith(pre):
            return lab + p.rsplit("_s", 1)[1]
    return _base_name(p)


size.name = name                                                    # size.table() maps policy names through this


def group_files(cond, group):
    """episodes.csv files of one group on one floor (finished tasks only), wherever that cell was run."""
    if cond == "m15" and group == "pdr":
        fs = [R / "rq3-gen-v5" / "base" / "pdr" / "episodes.csv"]
    elif cond == "m15" and group == "cmix":
        fs = [R / "eval-v5-cmix" / "test" / "base" / "cmix" / "episodes.csv"]
    elif cond in ("m50", "m100") and group in ("pdr", "cmix"):
        d = R / "rq3-size-v5" / cond
        fs = [d / group / "episodes.csv"] + sorted(d.glob(f"{group}_s*-*/episodes.csv"))
    else:
        fs = sorted((HERE / cond).glob(f"{group}_s*-*/episodes.csv"))
    return [f for f in fs if f.exists() and (f.parent / "DONE").exists()]


def ref_check(cond, pdr):
    """(matching, compared) MDD-PTECT seeds between each policy group's file and the pdr group's."""
    ok = n = 0
    for g in ("cmix", "size50", "size100", "sizemix"):
        for f in group_files(cond, g):
            e = pd.read_csv(f, dtype=H)
            e = e[(e.policy == REF) & e.seed.isin(pdr.index)]
            for s, v in zip(e.seed, e.window_tardiness / 1000.0):
                n += 1
                ok += abs(v - pdr.loc[s, REF]) < 1e-6
    return ok, n


def fam(t, arm):
    mem = [f"{arm} s{k}" for k in range(5)]
    if not all(m in t.columns for m in mem) or t[mem].notna().all(axis=1).sum() < 40:
        return None
    return mem


def line(lab, c):
    return (f"   {lab:26s} {c['pooled_%']:+6.2f} [{c['ci_lo']:+6.2f}, {c['ci_hi']:+6.2f}] "
            f"{c['better']:2d}/{c['worse']:<2d} p={c['wilcoxon_p']:.2g}")


def main():
    rng = np.random.default_rng(0)
    print("== rq3-size-grid: floor-size cross-evaluation, twin, test seeds 0-39, window tardiness; < 0 better")
    print("   arms: T15 = final-policy family (train-v5-cmix), T50 / T100 / Tmix = train-size-m50 / -m100 / -mix")
    rows, grid, curve = [], {}, []
    for cond, m in FLOORS.items():
        pdr = size.table(group_files(cond, "pdr"))
        pairs = [p for p in rq3.V5 if p in pdr.columns]
        if len(pairs) < 6 or len(pdr) < 40:
            print(f"\n-- {cond}: pairs not complete yet ({len(pairs)} / 6, {len(pdr)} seeds)")
            continue
        files = group_files(cond, "pdr") + [f for g in ("cmix", "size50", "size100", "sizemix")
                                            for f in group_files(cond, g)]
        t = size.table(files)
        tot = t[pairs].sum().sort_values()
        best = tot.index[0]
        ok, n = ref_check(cond, pdr)
        print(f"\n-- {cond} ({m} machines): MDD-PTECT mean {t[REF].mean():.1f} k job-s; best pair {best}; pairs vs "
              f"MDD-PTECT: " + ", ".join(f"{p} {100 * (tot[p] / tot[REF] - 1):+.1f}%" for p in tot.index))
        print(f"   run-to-run: MDD-PTECT in the policy groups = the pdr group's on {ok} / {n} seed-runs"
              + ("" if ok == n else "   ** MISMATCH **"))
        t["hindsight best v5 pair"] = t[pairs].min(axis=1)
        c = rq3.compare(t["hindsight best v5 pair"].to_dict(), t[REF].to_dict(), rng)
        print(line("hindsight best v5 pair", c) + "   (vs MDD-PTECT)")
        fams = {}
        for arm, tr in ARMS.items():
            mem = fam(t, arm)
            if mem is None:
                print(f"   {arm:26s} not complete yet")
                continue
            t[f"{arm} mean of 5"] = t[mem].mean(axis=1, skipna=False)
            fams[arm] = f"{arm} mean of 5"
            per = [100 * (t[k].sum() / t[REF].sum() - 1) for k in mem]
            tt = ttest_1samp(per, 0.0)
            cr = rq3.compare(t[fams[arm]].to_dict(), t[REF].to_dict(), rng)
            cb = rq3.compare(t[fams[arm]].to_dict(), t[best].to_dict(), rng)
            grid[(arm, cond)] = cr["pooled_%"]
            print(line(f"{arm} mean of 5", cr) + f"   | vs {best} {cb['pooled_%']:+6.2f}"
                  f"   | training seeds " + " ".join(f"{v:+.1f}" for v in per)
                  + f" (mean {np.mean(per):+.2f}, sd {np.std(per, ddof=1):.2f}, t-test p={tt.pvalue:.2g})")
            for vs, b, cc in (("MDD-PTECT", REF, cr), ("best pair", best, cb)):
                rows.append({"floor": cond, "machines": m, "arm": arm, "train_size": tr, "policy": "mean of 5",
                             "vs": vs, "best_pair": best, **cc,
                             **({"seed_mean_%": np.mean(per), "seed_sd_%": np.std(per, ddof=1),
                                 "seed_ttest_p": tt.pvalue} if vs == "MDD-PTECT" else {})})
            for k, v in zip(mem, per):
                rows.append({"floor": cond, "machines": m, "arm": arm, "train_size": tr, "policy": k,
                             "vs": "MDD-PTECT", "best_pair": best, "pooled_%": v})
            if tr is not None:
                curve.append({"arm": arm, "floor": cond, "d": abs(np.log10(m / tr)), "vs_ref": cr["pooled_%"],
                              "vs_best": cb["pooled_%"]})
        if "Tmix" in fams:
            for arm in ("T15", "T50", "T100"):
                if arm in fams:
                    c = rq3.compare(t[fams["Tmix"]].to_dict(), t[fams[arm]].to_dict(), rng)
                    rows.append({"floor": cond, "machines": m, "arm": "Tmix", "policy": "mean of 5",
                                 "vs": f"{arm} mean of 5", **c})
                    print(line(f"Tmix vs {arm} (paired)", c))
    if grid:
        g = pd.Series(grid).unstack().reindex(index=list(ARMS), columns=list(FLOORS))
        g.to_csv(HERE / "grid.csv")
        print("\n== family mean of 5 vs MDD-PTECT, pooled % (rows: training size, columns: test floor)")
        print(g.round(2).to_string(na_rep="."))
    if len(curve) >= 4:
        cv = pd.DataFrame(curve)
        print("\n== transfer curve (single-size arms): gap vs |log10(test / train)|")
        for col in ("vs_ref", "vs_best"):
            r, p = spearmanr(cv.d, cv[col])
            print(f"   {col:7s} Spearman rho {r:+.2f} (p {p:.2g}, {len(cv)} arm-floor cells)")
        print(cv.sort_values(["arm", "d"]).round(3).to_string(index=False))
    pd.DataFrame(rows).to_csv(HERE / "summary.csv", index=False)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
