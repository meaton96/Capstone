"""
@file analyze.py
@brief eval-alpha-only-size (2026-10-28): pre-declared analysis of the alpha-head-only policies (train-alpha-off-het /
       -b2 with the job and machine heads fixed to MDD-PTECT, only the offset head choosing) on B2 (m15) and the
       floor-size grid's 50 / 100 / 200-machine floors (twin, test seeds 0-39; README.md has the design and the criteria,
       written before any result). The floors are homogeneous, so the law is alpha 3 and every move of alpha is an offset.
Sections:
  0. checks: ao-het's MDD-PTECT equals the reference pdr run seed for seed; the full arms on m15 equal
     ../eval-alpha-offset's B2 runs; every alpha-only decision played MDD / PTECT; seeds complete.
  1. per floor, window tardiness % vs MDD-PTECT pooled over test seeds [95% instance-bootstrap CI], better / worse,
     Wilcoxon: the fixed alphas the offsets reach and their per-instance best, the final policy (T15), the full arms
     (../eval-alpha-off-size; m15 here) and the alpha-only arms (family = per-instance mean of 5 training seeds; each
     seed's own pooled % and a one-sample t-test of the 5).
  2. paired: each alpha-only arm vs its full arm, vs MDD-PTECT@0.75, vs T15; ao-het vs ao-b2; capture of the reachable
     fixed-alpha room by the full and the alpha-only arms.
  3. offsets chosen per slot, alpha-only vs full: shares, mean log2 offset, mean played alpha, by regime; per seed.
  4. criteria O1 (gain at 200 machines), O2 (the pair heads cost at 200), O3 (no loss with size), O4 (reaches the
     static room at 200), B (B2 cost vs the full arm), N (next step, ao-het).
Outputs: analysis.out (stdout), summary.csv, offset_choices.csv.
@par Usage (from the repo root)
@code{.sh}
.venv/bin/python results/eval-alpha-only-size/analyze.py [--root DIR] [--seeds 0-39] [--min-n 5]
@endcode
"""
import argparse
import csv
import glob
import importlib.util
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
R = HERE.parent
_spec = importlib.util.spec_from_file_location("ofs_analyze", R / "eval-alpha-off-size" / "analyze.py")
OFS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OFS)                     # read, ckpts, family, worse_sig, verdict (+ compare, fmt, regime_of)
compare, fmt, regime_of, read, ckpts, family = OFS.compare, OFS.fmt, OFS.regime_of, OFS.read, OFS.ckpts, OFS.family
worse_sig, verdict = OFS.worse_sig, OFS.verdict

FLOORS = ("m15", "m50", "m100", "m200")
ARMS = ("het", "b2")                              # ao-<arm> here, off-<arm> the full arm
LABEL = {"het": "train-alpha-off-het", "b2": "train-alpha-off-b2"}
REF = "MDD-PTECT"
ALPHAS = {"MDD-PTECT": 3.0, "MDD-PTECT@1.5": 1.5, "MDD-PTECT@0.75": 0.75}
OFFSETS = OFS.OFFSETS
## @brief References on the same floors and seeds: the pdr run holding MDD-PTECT, the final policy T15.
SOURCES = {"m15": {"pdr": "rq3-gen-v5/base/pdr", "T15": "eval-v5-cmix/test/base/cmix"},
           **{f: {"pdr": OFS.SOURCES[f]["pdr"], "T15": OFS.SOURCES[f]["T15"]} for f in ("m50", "m100", "m200")}}


def full_dirs(root, f, arm):
    """The full arm's output dirs: m15 here (harness check), m50-m200 ../eval-alpha-off-size."""
    base = root if f == "m15" else R / "eval-alpha-off-size"
    return glob.glob(str(base / "out" / f / f"off-{arm}_s*"))


def alpha_dirs(root, f):
    """Fixed-alpha runs: m15 in ao-het here, m50-m200 ../eval-alpha-off-size's alpha group."""
    return glob.glob(str(root / "out" / f / "ao-het_s*")) if f == "m15" else \
        glob.glob(str(R / "eval-alpha-off-size" / "out" / f / "alpha_s*"))


def seed_line(src, ref, rng, min_n):
    vals = [e["pct"] for e in (compare(v, ref, rng, min_n) for v in ckpts(src).values()) if e]
    if len(vals) < 2:
        return vals, ""
    return vals, f" | seeds {' '.join(f'{v:+.1f}' for v in vals)} (t-test p={stats.ttest_1samp(vals, 0.0).pvalue:.2g})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--seeds", default="0-39")
    ap.add_argument("--min-n", type=int, default=5, help="fewest paired instances for a comparison (dry runs: 1)")
    a = ap.parse_args()
    root = Path(a.root)
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    rng = np.random.default_rng(20261028)
    print(f"== eval-alpha-only-size: alpha-head-only policies (job / machine heads fixed to MDD-PTECT) on B2 and the "
          f"size grid (law alpha 3); seeds {a.seeds}\n   window tardiness, % vs MDD-PTECT pooled over instances [95% CI] "
          f"better/worse Wilcoxon p")

    AO, FULL, DA, DF, B, G, AL = {}, {}, {}, {}, {}, {}, {}
    for f in FLOORS:
        for arm in ARMS:
            AO[f, arm], DA[f, arm], blk = read(glob.glob(str(root / "out" / f / f"ao-{arm}_s*")), seeds)
            B.setdefault(f, {}).update(blk)
            FULL[f, arm], DF[f, arm], blk = read(full_dirs(root, f, arm), seeds)
            B[f].update(blk)
        G[f] = {k: read(glob.glob(str(R / pat)), seeds)[0] for k, pat in SOURCES[f].items()}
        al = read(alpha_dirs(root, f), seeds)[0]
        AL[f] = {k: (G[f]["pdr"].get(REF, {}) if k == REF else al.get(k, {})) for k in ALPHAS}
    ref = {f: G[f]["pdr"].get(REF, {}) for f in FLOORS}

    # 0. checks
    print("\n-- 0. checks")
    for f in FLOORS:
        for arm in ARMS:
            mine = AO[f, arm].get(REF, {})
            same = sum(1 for s in mine if s in ref[f] and abs(mine[s] - ref[f][s]) <= 1e-9 * max(1.0, abs(ref[f][s])))
            n_seeds = sorted({len(v) for k, v in AO[f, arm].items() if k.startswith("ckpt:")}) or "-"
            bad = sum(1 for r in DA[f, arm] if r["job_head"] != "1" or r["machine_head"] != "1")
            print(f"   {f:4s} ao-{arm:3s}: {len(ckpts(AO[f, arm]))} alpha-only policies, seeds per policy {n_seeds}; "
                  f"MDD-PTECT = reference on {same} / {len(mine)} seeds; decisions not MDD / PTECT: {bad} of "
                  f"{len(DA[f, arm])}")
        print(f"   {f:4s} references: pdr MDD-PTECT {len(ref[f])} seeds, T15 {len(family(G[f]['T15']))}, fixed alpha "
              f"{' / '.join(str(len(AL[f][k])) for k in ALPHAS)}; full arms {len(family(FULL[f, 'het']))} / "
              f"{len(family(FULL[f, 'b2']))}")
    for arm in ARMS:
        old = read([R / "eval-alpha-offset" / "out" / "0" / f"off-{arm}"], seeds, need_done=False)[0]
        here = ckpts(FULL["m15", arm])
        pairs = [(p, s) for p in here for s in here[p] if s in old.get(p, {})]
        same = sum(1 for p, s in pairs if abs(here[p][s] - old[p][s]) <= 1e-9 * max(1.0, abs(old[p][s])))
        print(f"   m15 off-{arm} here = ../eval-alpha-offset B2 on {same} / {len(pairs)} seed-runs")

    # 1. per floor
    print("\n-- 1. per floor, % vs MDD-PTECT (alpha 3 = the law on a homogeneous floor)")
    summary, fam = [], {}
    for f in FLOORS:
        if not ref[f]:
            continue
        al = AL[f]
        ss = sorted(s for s in ref[f] if all(s in al[k] for k in ALPHAS))
        rows = [(f"{k} (offset x{ALPHAS[k] / 3:g})", al[k]) for k in ("MDD-PTECT@1.5", "MDD-PTECT@0.75")]
        if ss:
            rows.append(("per-instance best of alpha 3 / 1.5 / 0.75 (hindsight)",
                         {s: min(al[k][s] for k in ALPHAS) for s in ss}))
        srcs = [("T15", "final policy T15 (train-v5-cmix)", G[f]["T15"])]
        srcs += [(f"off-{arm}", f"{LABEL[arm]} full", FULL[f, arm]) for arm in ARMS]
        srcs += [(f"ao-{arm}", f"{LABEL[arm]} alpha-only", AO[f, arm]) for arm in ARMS]
        for key, lab, src in srcs:
            if not ckpts(src):
                continue
            fam[f, key] = family(src)
            _, tag = seed_line(src, ref[f], rng, a.min_n)
            rows.append((f"{lab}, mean of {len(ckpts(src))}", fam[f, key], tag))
        print(f"   {f} (MDD-PTECT mean {np.mean(list(ref[f].values())):,.1f} k job-s, n {len(ref[f])})")
        for row in rows:
            lab, x, tag = (row + ("",))[:3]
            r = compare(x, ref[f], rng, a.min_n) if x else None
            print(f"     {lab:58s} {fmt(r)}{tag}")
            if r:
                summary.append({"floor": f, "row": lab, **{k: round(v, 4) if isinstance(v, float) else v
                                                         for k, v in r.items()}})

    # 2. paired
    print("\n-- 2. paired comparisons (pre-declared)")
    P = {}
    for f in FLOORS:
        parts = []
        for arm in ARMS:
            ao = fam.get((f, f"ao-{arm}"))
            if not ao:
                continue
            for key, other in ((f"off-{arm}", fam.get((f, f"off-{arm}"))), ("@0.75", AL[f]["MDD-PTECT@0.75"]),
                               ("T15", fam.get((f, "T15")))):
                if other:
                    P[f, arm, key] = compare(ao, other, rng, a.min_n)
                    parts.append(f"ao-{arm} vs {key} {fmt(P[f, arm, key])}")
        if fam.get((f, "ao-het")) and fam.get((f, "ao-b2")):
            P[f, "het-b2"] = compare(fam[f, "ao-het"], fam[f, "ao-b2"], rng, a.min_n)
            parts.append(f"ao-het vs ao-b2 {fmt(P[f, 'het-b2'])}")
        if parts:
            print(f"   {f}:\n     " + "\n     ".join(parts))
    print("   capture of the reachable fixed-alpha room (arm - MDD-PTECT) / (per-instance best alpha - MDD-PTECT), pooled:")
    for f in FLOORS:
        al = AL[f]
        for key in ("off-het", "ao-het", "off-b2", "ao-b2"):
            x = fam.get((f, key), {})
            ss = [s for s in x if s in ref[f] and all(s in al[k] for k in ALPHAS)]
            if len(ss) < a.min_n:
                continue
            tot = sum(ref[f][s] for s in ss)
            room = sum(min(al[k][s] for k in ALPHAS) - ref[f][s] for s in ss)
            gain = sum(x[s] - ref[f][s] for s in ss)
            print(f"     {f:4s} {key:7s}: room {100 * room / tot:+.2f}%, arm {100 * gain / tot:+.2f}%, capture "
                  f"{gain / room:+.2f}" if room < 0 else f"     {f:4s} {key:7s}: no reachable room")

    # 3. offsets
    print("\n-- 3. offsets chosen per slot (policy rows; law alpha 3): shares x1 / x0.5 / x0.25 / x1.5 / x2, mean log2 "
          "offset, mean played alpha, by regime")
    choices, by_seed = [], {}
    for arm in ARMS:
        for kind, D in (("full", DF), ("ao", DA)):
            for f in FLOORS:
                rows = D[f, arm]
                if not rows:
                    continue
                off = np.array([OFFSETS[int(r["alpha_head"])] for r in rows])
                played = np.minimum(4.0, 3.0 * off)
                share = [float((off == v).mean()) for v in OFFSETS]
                reg_a = {}
                for r, p in zip(rows, played):
                    bl = B[f].get(int(r["seed"]))
                    reg_a.setdefault(regime_of(bl, float(r["sim_time"])) if bl else None, []).append(p)
                    by_seed.setdefault((arm, kind, r["policy"].split("~")[0], f), []).append(p)
                regs = " ".join(f"{k} a{np.mean(v):.2f}" for k, v in sorted(reg_a.items(), key=lambda kv: str(kv[0])))
                print(f"   {kind:4s} {arm:3s} {f:4s} shares {' / '.join(f'{x:.2f}' for x in share)} | log2 off "
                      f"{np.log2(off).mean():+.2f} | alpha {played.mean():.2f} | {regs}")
                choices.append({"arm": arm, "kind": kind, "floor": f, "slots": len(rows),
                                "mean_log2_offset": round(float(np.log2(off).mean()), 4),
                                "mean_alpha": round(float(played.mean()), 4),
                                **{f"share_x{v:g}": round(x, 4) for v, x in zip(OFFSETS, share)},
                                **{f"alpha_{k}": round(float(np.mean(v)), 4) for k, v in reg_a.items()}})
    print("   per training seed, mean played alpha full -> alpha-only, m15 / m50 / m100 / m200:")
    for arm in ARMS:
        for k in range(5):
            pol = f"ckpt:{LABEL[arm]}_s{k}/checkpoint"
            cells = []
            for f in FLOORS:
                x, y = by_seed.get((arm, "full", pol, f)), by_seed.get((arm, "ao", pol, f))
                cells.append(f"{np.mean(x):.2f}->{np.mean(y):.2f}" if x and y else "n/a")
            print(f"     {arm} s{k}: " + " / ".join(cells))

    # 4. criteria
    print("\n-- 4. criteria (pre-declared)")
    res = {}
    for arm in ARMS:
        ao = {f: fam.get((f, f"ao-{arm}")) for f in FLOORS}
        vs_ref = {f: compare(ao[f], ref[f], rng, a.min_n) if ao[f] else None for f in FLOORS}
        o1r = vs_ref["m200"]
        o1 = None if o1r is None else (o1r["pct"] < 0 and o1r["p"] < 0.05)
        o2r = P.get(("m200", arm, f"off-{arm}"))
        o2 = None if o2r is None else (o2r["pct"] < 0 and o2r["p"] < 0.05)
        o3r = [vs_ref[f] for f in ("m50", "m100", "m200")]
        o3 = None if any(r is None for r in o3r) else not any(worse_sig(r) for r in o3r)
        o4r = P.get(("m200", arm, "@0.75"))
        o4 = None if o4r is None else (o4r["pct"] <= 1.0 and not worse_sig(o4r))
        br = P.get(("m15", arm, f"off-{arm}"))
        b = None if br is None else (br["pct"] <= 1.0 and not worse_sig(br))
        print(f"   ao-{arm}:")
        print(f"     O1 gain at m200, better than MDD-PTECT (p < 0.05): {fmt(o1r)} -> {verdict(o1)}")
        print(f"     O2 the pair heads cost at m200, better than the full arm (p < 0.05): {fmt(o2r)} -> {verdict(o2)}")
        print(f"     O3 no loss with size, not significantly worse than MDD-PTECT on m50 / m100 / m200: "
              f"{' / '.join(fmt(r) for r in o3r)} -> {verdict(o3)}")
        print(f"     O4 reaches the static room at m200, within +1% of MDD-PTECT@0.75 and not significantly worse: "
              f"{fmt(o4r)} -> {verdict(o4)}")
        print(f"     B  B2 cost, within +1 point of the full arm on m15 and not significantly worse: {fmt(br)} -> "
              f"{verdict(b)}   (m15 vs MDD-PTECT: {fmt(vs_ref['m15'])})")
        res[arm] = (o1, o2, o3, o4, b)
    o1, _, _, o4, _ = res["het"]
    if o1 is None or o4 is None:
        print("   N  n/a (m200 incomplete)")
    else:
        nxt = ("the pair heads: the offset head picks alpha well once the pair is fixed" if o1 and o4 else
               "both: the pair heads cost part of the room and the offset head stops short (size / transport term)"
               if o1 else "the law: the offset head does not take the room even with the pair fixed (size / transport "
               "term in the law)")
        print(f"   N  (ao-het, O1 {verdict(o1)}, O4 {verdict(o4)}) -> next step: {nxt}")

    with open(root / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]) if summary else ["floor"])
        w.writeheader()
        w.writerows(summary)
    if choices:
        keys = sorted({k for c in choices for k in c}, key=lambda k: list(choices[0]).index(k) if k in choices[0] else 99)
        with open(root / "offset_choices.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(choices)


if __name__ == "__main__":
    main()
