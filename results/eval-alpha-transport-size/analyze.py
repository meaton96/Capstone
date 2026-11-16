"""
@file analyze.py
@brief eval-alpha-transport-size (2026-10-28): stage 2's pre-declared analysis (README.md has the design and the
       criteria, written before any result). train-alpha-off-het with the transport law of ../rq3-alpha-transport
       (law.json) and the size gate, on the six test cells, twin, test seeds 0-39.
Sections:
  0. checks: seeds complete; lg's MDD-PTECT equals the simulator's alpha 3 seed for seed; the episode rows carry the
     law's alpha (lg / lf) or 3 (v7); gated lg decisions are all MDD / PTECT; B2's h is 1.
  1. per cell, window tardiness % vs MDD-PTECT pooled over test seeds [95% instance-bootstrap CI], better / worse,
     Wilcoxon: the fixed alphas (simulator), the cell's best alpha chosen on fitting seeds (alpha*_fit), the best single
     alpha on the test seeds (hindsight), the law alone, the v7 arm (full; alpha-only where it exists), lf, lg, and the
     final policy T15 where the grid has it. Family = per-instance mean of the 5 training seeds; each seed's own
     pooled % and a one-sample t-test of the 5.
  2. criteria P1-P6, H and the decision rule N.
  3. what the policies play: mean played alpha and offset shares (lg, lf, v7), pair shares (lf, v7).
Outputs: analysis.out (stdout), summary.csv, played.csv.
@par Usage (from the repo root)
@code{.sh}
.venv/bin/python results/eval-alpha-transport-size/analyze.py [--root DIR] [--law FILE] [--transport-root DIR]
    [--seeds 0-39] [--min-n 5]
@endcode
"""
import argparse
import csv
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
R = HERE.parent
_spec = importlib.util.spec_from_file_location("ofs_analyze", R / "eval-alpha-off-size" / "analyze.py")
OFS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OFS)                     # read, ckpts, family, compare, fmt, worse_sig, regime_of
compare, fmt, worse_sig = OFS.compare, OFS.fmt, OFS.worse_sig

A = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
OFFSETS = [1.0, 0.5, 0.25, 1.5, 2.0]
CELLS = ["m50_r052", "m75_r052", "m100_r052", "m200_r052", "m100_r035", "m100_r080"]
HELD_OUT = ["m75_r035", "m150_r080"]
GATE_MACHINES = 50
## @brief Test-seed alpha curves: ../rq3-alpha-size's s0 cells for 50-100 machines at r052, else ../rq3-alpha-transport.
SIM_CURVE = {"m50_r052": "rq3-alpha-size/out/m50_s0", "m75_r052": "rq3-alpha-size/out/m75_s0",
             "m100_r052": "rq3-alpha-size/out/m100_s0"}
## @brief v7 references on the grid cells (full arm / alpha-only), and T15, from earlier runs on the same seeds.
V7_FULL = {"m50_r052": "eval-alpha-off-size/out/m50/off-het_s*", "m100_r052": "eval-alpha-off-size/out/m100/off-het_s*",
           "m200_r052": "eval-alpha-off-size/out/m200/off-het_s*"}
V7_AO = {"m50_r052": "eval-alpha-only-size/out/m50/ao-het_s*", "m100_r052": "eval-alpha-only-size/out/m100/ao-het_s*",
         "m200_r052": "eval-alpha-only-size/out/m200/ao-het_s*"}
T15 = {"m50_r052": OFS.SOURCES["m50"]["T15"], "m100_r052": OFS.SOURCES["m100"]["T15"],
       "m200_r052": OFS.SOURCES["m200"]["T15"], "m75_r052": "rq3-size-grid/m75/cmix_s*"}


TRANSPORT = R / "rq3-alpha-transport"


def sim_runs(cell, seeds, sub=""):
    """{seed: {config: total}} of the simulator's runs of a cell on test seeds (sub "law": the law runs)."""
    base = R / SIM_CURVE[cell] if cell in SIM_CURVE and not sub else TRANSPORT / "out" / cell / sub
    out = {}
    for s in seeds:
        f = base / f"s{s}.json"
        if f.exists():
            d = json.loads(f.read_text())
            out[s] = {k: v["total"] for k, v in d.items() if not k.startswith("_")}
    return out


def by_rule(runs, name):
    return {s: v[name] for s, v in runs.items() if name in v}


def read_group(root, cell, group, seeds):
    return OFS.read(glob.glob(str(root / "out" / cell / f"{group}_s*")), seeds)


def seed_line(src, ref, rng, min_n):
    vals = [e["pct"] for e in (compare(v, ref, rng, min_n) for v in OFS.ckpts(src).values()) if e]
    if len(vals) < 2:
        return ""
    return f" | seeds {' '.join(f'{v:+.1f}' for v in vals)} (t-test p={stats.ttest_1samp(vals, 0.0).pvalue:.2g})"


def played(dec, law_alpha):
    """Policy slots' offset shares, mean log2 offset, mean played alpha, PECT share, non-MDD share."""
    rows = [r for r in dec if r["kind"] != "pdr"]
    if not rows:
        return None
    offs, pect, nonmdd = [], 0, 0
    for r in rows:
        base, _, suf = r["rule"].partition("@")
        offs.append(float(suf[1:]) if suf.startswith("x") else 1.0)
        pect += base.endswith("-PECT")
        nonmdd += not base.startswith("MDD-")
    offs = np.array(offs)
    alpha = np.minimum(4.0, law_alpha * offs)
    return {"slots": len(rows), **{f"share_x{o:g}": float(np.mean(offs == o)) for o in OFFSETS},
            "mean_log2_offset": float(np.mean(np.log2(offs))), "mean_alpha": float(alpha.mean()),
            "pect_share": pect / len(rows), "non_mdd_share": nonmdd / len(rows)}


def main(argv=None):
    global TRANSPORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--law", default=str(R / "rq3-alpha-transport" / "law.json"))
    ap.add_argument("--seeds", default="0-39")
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--transport-root", default=str(TRANSPORT), help="stage 1's folder (its out/; dry runs)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    TRANSPORT = Path(a.transport_root)
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = set(range(lo, hi + 1))
    rng = np.random.default_rng(0)
    law = json.loads(Path(a.law).read_text())
    print("== eval-alpha-transport-size: train-alpha-off-het with the transport law and the size gate (twin, test seeds "
          f"{a.seeds})")
    print(f"   law {law['law']}: alpha = max(0, 3 - 6 sigma) x clip(({law['x1']:.4f} - x) / ({law['x1']:.4f} - "
          f"{law['x0']:.4f}), 0, 1), x = {law['label']}; gate F1 {law['F1']}")
    print("   window tardiness, % vs MDD-PTECT pooled over instances [95% CI] better/worse Wilcoxon p")

    D, summary, played_rows = {}, [], []
    print("\n-- 0. checks")
    print(f"   B2 (m15_b2): h {law['cells']['m15_b2']['h']} (the law is v7's on B2, so B2 needs no rerun)")
    for c in CELLS:
        la = law["cells"][c]["alpha"]
        sim = sim_runs(c, seeds)
        d = {"sim": sim, "law_rule": by_rule(sim_runs(c, seeds, "law"), f"MDD-PTECT@{la:g}"),
             "ref": by_rule(sim, "MDD-PTECT@3")}
        lines = []
        for g in ("lg", "lf", "v7"):
            pol, dec, _ = read_group(root, c, g, seeds)
            if not pol:
                continue
            d[g], d[g + "_dec"] = pol, dec
            ck = OFS.ckpts(pol)
            n = sorted(len(v) for v in ck.values())
            lines.append(f"{g}: {len(ck)} policies, seeds per policy {n}")
            want = 3.0 if g == "v7" else la
            labels = set()
            for f in glob.glob(str(root / "out" / c / f"{g}_s*" / "episodes.csv")):
                labels |= {float(r["alpha_law"]) for r in csv.DictReader(open(f))}
            lines.append(f"alpha label {sorted(labels)} (want {want})")
            if g == "lg":
                h = pol.get("MDD-PTECT", {})
                same = sum(abs(h[s] - d["ref"][s]) <= 1e-6 * max(1.0, abs(d["ref"][s])) for s in h if s in d["ref"])
                lines.append(f"MDD-PTECT = simulator alpha 3 on {same} / {sum(s in d['ref'] for s in h)} common seeds "
                             f"({len(h)} run)")
                if int(c.split("_")[0][1:]) > GATE_MACHINES:
                    bad = sum(1 for r in dec if r["kind"] != "pdr" and r["rule"].partition("@")[0] != "MDD-PTECT")
                    lines.append(f"gated, decisions not MDD-PTECT: {bad} of {sum(r['kind'] != 'pdr' for r in dec)}")
        lines.append(f"simulator: grid {len(sim)} seeds, law alone (alpha {la:g}) {len(d['law_rule'])}")
        for name, pat in (("v7_full", V7_FULL), ("v7_ao", V7_AO), ("T15", T15)):
            if c in pat:
                pol, dec, _ = OFS.read(glob.glob(str(R / pat[c])), seeds)
                d[name] = OFS.family(pol) if OFS.ckpts(pol) else {}
                d[name + "_pol"] = pol
        D[c] = d
        print(f"   {c:10s} " + "; ".join(lines))
    for c in HELD_OUT:
        la = law["cells"][c]["alpha"]
        sim = sim_runs(c, seeds)
        D[c] = {"sim": sim, "law_rule": by_rule(sim_runs(c, seeds, "law"), f"MDD-PTECT@{la:g}"),
                "ref": by_rule(sim, "MDD-PTECT@3")}
        print(f"   {c:10s} held out: grid {len(sim)} seeds, law alone (alpha {la:g}) {len(D[c]['law_rule'])}")

    print("\n-- 1. per cell, % vs MDD-PTECT (alpha 3)")
    R_ = {}
    for c in CELLS + HELD_OUT:
        d, la = D[c], law["cells"][c]["alpha"]
        ref = d["ref"]
        if len(ref) < a.min_n:
            print(f"   {c}: no reference yet")
            continue
        pooled = {x: sum(by_rule(d["sim"], f"MDD-PTECT@{x:g}").values()) for x in A}
        hind = min(pooled, key=pooled.get)
        fit_best = law["cells"][c]["best_alpha_fit"]
        print(f"   {c} (MDD-PTECT mean {np.mean(list(ref.values())):,.1f} k job-s, n {len(ref)}; law alpha {la:g}, "
              f"alpha*_fit {fit_best}, hindsight best single alpha {hind:g})")
        rows = [(f"MDD-PTECT@{x:g}", by_rule(d["sim"], f"MDD-PTECT@{x:g}"), "") for x in A if x != 3.0]
        rows.append((f"law alone, MDD-PTECT@{la:g}", d["law_rule"], ""))
        res = {"law": compare(d["law_rule"], ref, rng, a.min_n),
               "fit": compare(by_rule(d["sim"], f"MDD-PTECT@{fit_best:g}"), ref, rng, a.min_n) if fit_best is not None else None,
               "hind": compare(by_rule(d["sim"], f"MDD-PTECT@{hind:g}"), ref, rng, a.min_n), "hind_alpha": hind}
        for name, label in (("T15", "final policy T15, mean of 5"), ("v7_ao", "v7 off-het alpha-only, mean of 5"),
                            ("v7_full", "v7 off-het full, mean of 5")):
            if d.get(name):
                rows.append((label, d[name], seed_line(d[name + "_pol"], ref, rng, a.min_n)))
        for g, label in (("v7", "v7 off-het full (here), mean of 5"), ("lf", "lf: transport law, full, mean of 5"),
                         ("lg", "lg: transport law + gate, mean of 5")):
            if d.get(g):
                fam = OFS.family(d[g])
                d[g + "_fam"] = fam
                rows.append((label, fam, seed_line(d[g], ref, rng, a.min_n)))
        for label, x, extra in rows:
            r = compare(x, ref, rng, a.min_n)
            print(f"     {label:44s} {fmt(r)}{extra}")
            if r:
                summary.append({"cell": c, "row": label, **{k: r[k] for k in ("n", "pct", "lo", "hi", "better", "worse",
                                                                              "p")}})
        R_[c] = res
        for g in ("lg", "lf", "v7"):
            if d.get(g + "_dec"):
                p = played(d[g + "_dec"], 3.0 if g == "v7" else la)
                if p:
                    played_rows.append({"cell": c, "group": g, **p})

    print("\n-- 2. pre-declared criteria (family of 5, paired over test seeds)")

    def fam(c, g):
        return D.get(c, {}).get(g + "_fam") or D.get(c, {}).get(g) or {}

    def cmp(c, g, other):
        return compare(fam(c, g), other, rng, a.min_n)

    def show(label, r):
        print(f"     {label:52s} {fmt(r)}")
        return r

    verdict = {}
    r = show("P1 m200_r052 lg vs v7 alpha-only", cmp("m200_r052", "lg", D["m200_r052"].get("v7_ao", {})))
    verdict["P1"] = r is not None and r["pct"] < 0 and r["p"] < 0.05
    ok = True
    for c in ("m100_r052", "m200_r052"):
        fb = law["cells"][c]["best_alpha_fit"]
        r = show(f"P2 {c} lg vs alpha*_fit (MDD-PTECT@{fb:g})", cmp(c, "lg", by_rule(D[c]["sim"], f"MDD-PTECT@{fb:g}")))
        ok &= r is not None and r["pct"] <= 1.0 and not worse_sig(r)
    verdict["P2"] = ok
    ok = True
    for c in CELLS:
        r = show(f"P3 {c} lg vs MDD-PTECT", cmp(c, "lg", D[c]["ref"]))
        ok &= r is not None and not worse_sig(r)
    r = show("P3 m50_r052 lg vs v7 off-het full", cmp("m50_r052", "lg", D["m50_r052"].get("v7_full", {})))
    verdict["P3"] = ok and r is not None and not worse_sig(r)
    r = show("P4 m200_r052 lg vs lf", cmp("m200_r052", "lg", fam("m200_r052", "lf")))
    ok = r is not None and r["pct"] < 0 and r["p"] < 0.05
    for c in ("m75_r052", "m100_r052", "m100_r035", "m100_r080"):
        r = show(f"P4 {c} lg vs lf", cmp(c, "lg", fam(c, "lf")))
        ok &= r is not None and not worse_sig(r)
    verdict["P4"] = ok
    ok = True
    for c in CELLS:
        if int(c.split("_")[0][1:]) > GATE_MACHINES:
            r = show(f"P5 {c} lg vs the law alone", cmp(c, "lg", D[c]["law_rule"]))
            ok &= r is not None and not worse_sig(r)
    verdict["P5"] = ok
    r1 = show("P6 m100_r035 lg vs v7 off-het full", cmp("m100_r035", "lg", fam("m100_r035", "v7")))
    r2 = show("P6 m100_r080 lg vs v7 off-het full", cmp("m100_r080", "lg", fam("m100_r080", "v7")))
    verdict["P6"] = (r1 is not None and r1["pct"] < 0 and r1["p"] < 0.05) and (r2 is not None and not worse_sig(r2))
    ok = True
    for c in HELD_OUT:
        d = D[c]
        hind = R_.get(c, {}).get("hind_alpha")
        if hind is None:
            ok = False
            continue
        r = show(f"H  {c} law alone vs best single alpha ({hind:g})",
                 compare(d["law_rule"], by_rule(d["sim"], f"MDD-PTECT@{hind:g}"), rng, a.min_n))
        ok &= r is not None and r["pct"] <= 2.0
        best_vs3 = R_[c]["hind"]
        if best_vs3 is not None and best_vs3["pct"] <= -2.0:
            r = show(f"H  {c} law alone vs alpha 3 (room {best_vs3['pct']:+.1f}%)", compare(d["law_rule"], d["ref"], rng,
                                                                                         a.min_n))
            ok &= r is not None and r["pct"] < 0 and r["p"] < 0.05
    verdict["H"] = ok
    print("   " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in verdict.items()))
    law_fit_ok = all((lambda r: r is not None and r["pct"] <= 1.0 and not worse_sig(r))(
        compare(D[c]["law_rule"], by_rule(D[c]["sim"], f"MDD-PTECT@{law['cells'][c]['best_alpha_fit']:g}"), rng,
                a.min_n)) for c in ("m100_r052", "m200_r052"))
    if verdict["P1"] and verdict["P2"] and verdict["P3"]:
        n = "the transport law + gate is paper B's alpha policy across sizes and fleets (carried forward)"
    elif verdict["P1"]:
        n = ("the offset head costs on large floors: next, the law alone above the gate" if law_fit_ok else
             "the law form stops short of the room: next, refit (another variable or form)")
    else:
        n = "the transport term does not reach the policy: report the alpha curves and v7 + gate as the large-floor result"
    print(f"   N -> {n}")
    if not verdict["P4"]:
        print("   N (P4 FAIL) -> the pair heads no longer cost once alpha is right: drop the gate")
    if not verdict["H"]:
        print("   N (H FAIL) -> the law is a fit to these floors, not a rule that transfers to new size / fleet pairs")

    print("\n-- 3. what the policies play (policy slots): offset shares x1 / x0.5 / x0.25 / x1.5 / x2, mean played alpha, "
          "PECT share, non-MDD share")
    for p in played_rows:
        print(f"   {p['cell']:10s} {p['group']:3s} " + " / ".join(f"{p[f'share_x{o:g}']:.2f}" for o in OFFSETS) +
              f" | alpha {p['mean_alpha']:.2f} | PECT {p['pect_share']:.2f} | non-MDD {p['non_mdd_share']:.2f}")
    for name, rows in (("summary.csv", summary), ("played.csv", played_rows)):
        if rows:
            with open(root / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
