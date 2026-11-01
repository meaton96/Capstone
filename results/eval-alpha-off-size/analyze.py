"""
@file analyze.py
@brief eval-alpha-off-size (2026-10-26): pre-declared analysis of the law + offset arms (train-alpha-off-b2 / -het) on
       the floor-size grid's 50 / 100 / 200-machine floors (twin, test seeds 0-39; README.md has the design and the
       criteria, written before any result). The floors are homogeneous, so the law is alpha 3 and every move of alpha
       is the head's offset.
Sections:
  0. checks: every group's MDD-PTECT equals the grid's pdr MDD-PTECT seed for seed; seeds complete.
  1. per floor, window tardiness % vs MDD-PTECT pooled over test seeds [95% instance-bootstrap CI], better / worse,
     Wilcoxon: the fixed alphas the offsets reach (MDD-PTECT@1.5 / @0.75), their hindsight bests, MDD-TECT (alpha 0,
     outside the head's reach), the best and hindsight-best v5 pair, the final policy (T15, train-v5-cmix), T100
     (train-size-m100, m50 / m100 only), and the two arms (family = per-instance mean of 5 training seeds; each seed's
     own pooled % and a one-sample t-test of the 5).
  2. paired (pre-declared): each arm vs T15, vs T100 (m50 / m100), off-het vs off-b2; capture of the reachable
     fixed-alpha room.
  3. offsets chosen per slot: shares, mean log2 offset, mean played alpha, by regime (normal / surge / short fleet),
     per floor, with B2 (m15, ../eval-alpha-offset s 0) as the first row; per training seed.
  4. criteria S1 (no loss with size), S2 (gain at 200 machines), S3 (the head lowers alpha with size), D (which arm).
Outputs: analysis.out (stdout), summary.csv, offset_choices.csv.
@par Usage (from the repo root)
@code{.sh}
.venv/bin/python results/eval-alpha-off-size/analyze.py [--root DIR] [--seeds 0-39] [--min-n 5]
@endcode
"""
import argparse
import csv
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
R = HERE.parent
_spec = importlib.util.spec_from_file_location("eas_analyze", R / "eval-alpha-spread" / "analyze.py")
EAS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EAS)                     # compare, fmt, load_grid, regime_of
compare, fmt, regime_of = EAS.compare, EAS.fmt, EAS.regime_of

FLOORS = ("m50", "m100", "m200")
ARMS = ("off-b2", "off-het")
LABEL = {"off-b2": "train-alpha-off-b2", "off-het": "train-alpha-off-het", "T15": "final policy T15 (train-v5-cmix)",
         "T100": "T100 (train-size-m100)"}
REF = "MDD-PTECT"
ALPHAS = {"MDD-PTECT": 3.0, "MDD-PTECT@1.5": 1.5, "MDD-PTECT@0.75": 0.75}
OFFSETS = [1.0, 0.5, 0.25, 1.5, 2.0]              # v7 ALPHA_OFFSET_VALUES (index order)
V5 = ["SRT-PECT", "SRT-PTECT", "MDD-PECT", "MDD-PTECT", "ATC-PECT", "ATC-PTECT"]
## @brief The grid's outputs on the same floors and seeds: pdr (6 v5 pairs + MDD-TECT), T15, T100 (finished dirs only).
SOURCES = {
    "m50": {"pdr": "rq3-size-v5/m50/pdr", "T15": "rq3-size-v5/m50/cmix", "T100": "rq3-size-grid/m50/size100_s*"},
    "m100": {"pdr": "rq3-size-v5/m100/pdr_s*", "T15": "rq3-size-v5/m100/cmix_s*", "T100": "rq3-size-grid/m100/size100_s*"},
    "m200": {"pdr": "rq3-size-grid/m200/pdr_s*", "T15": "rq3-size-grid/m200/cmix_s*"},
}


def read(dirs, seeds, need_done=True):
    """{policy: {seed: window tardiness (k job-s)}}, decision rows (policy slots), {seed: blocks} of finished dirs
    (need_done False: ../eval-alpha-offset, whose DONE files sit one level up)."""
    pol, dec, blocks = {}, [], {}
    for d in sorted(Path(p) for p in dirs):
        if not (d / ("DONE" if need_done else "episodes.csv")).exists():
            continue
        for r in csv.DictReader(open(d / "episodes.csv")):
            if int(r["seed"]) in seeds:
                pol.setdefault(r["policy"], {})[int(r["seed"])] = float(r["window_tardiness"]) / 1000.0
        if (d / "decisions.csv").exists():
            dec += [r for r in csv.DictReader(open(d / "decisions.csv")) if r["kind"] != "pdr" and int(r["seed"]) in seeds]
        if (d / "blocks.json").exists():
            blocks.update({int(k): v for k, v in json.loads((d / "blocks.json").read_text()).items()})
    return pol, dec, blocks


def ckpts(pol):
    return {k: v for k, v in pol.items() if k.startswith("ckpt:")}


def family(pol):
    """Per-instance mean over the checkpoints present on every seed."""
    ck = ckpts(pol)
    if not ck:
        return {}
    common = sorted(set.intersection(*(set(v) for v in ck.values())))
    return {s: float(np.mean([v[s] for v in ck.values()])) for s in common}


def holm(results):
    """results: [compare-dicts or None]; all better and Holm-significant at 0.05."""
    if any(r is None for r in results):
        return None
    ps = sorted(results, key=lambda r: r["p"])
    return all(r["pct"] < 0 and r["p"] <= 0.05 / (len(ps) - i) for i, r in enumerate(ps))


def holm_better(results):
    """Holm over the list; the ones that are better and Holm-significant (indices)."""
    order = sorted(range(len(results)), key=lambda i: results[i]["p"])
    sig, m = set(), len(results)
    for rank, i in enumerate(order):
        if results[i]["p"] > 0.05 / (m - rank):
            break
        sig.add(i)
    return [i for i in sig if results[i]["pct"] < 0], [i for i in sig if results[i]["pct"] > 0]


def worse_sig(r):
    return r is not None and r["pct"] > 0 and r["p"] < 0.05


def verdict(x):
    return "n/a" if x is None else ("PASS" if x else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--seeds", default="0-39")
    ap.add_argument("--min-n", type=int, default=5, help="fewest paired instances for a comparison (dry runs: 1)")
    a = ap.parse_args()
    root = Path(a.root)
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    rng = np.random.default_rng(20261026)
    print(f"== eval-alpha-off-size: law + offset arms on the floor-size grid (homogeneous floors: law alpha 3); seeds "
          f"{a.seeds}\n   window tardiness, % vs MDD-PTECT pooled over instances [95% CI] better/worse Wilcoxon p")

    E, D, B, G = {}, {}, {}, {}
    for f in FLOORS:
        for grp in ARMS + ("alpha",):
            E[f, grp], D[f, grp], blk = read(glob.glob(str(root / "out" / f / f"{grp}_s*")), seeds)
            B.setdefault(f, {}).update(blk)
        for k, pat in SOURCES[f].items():
            G[f, k] = read(glob.glob(str(R / pat)), seeds)[0]
    m15 = {grp: read([R / "eval-alpha-offset" / "out" / "0" / grp], seeds, need_done=False) for grp in ARMS}
    m15_blocks = {s: v["blocks"] for s, v in EAS.load_grid(R / "eval-alpha-spread", "0", seeds).items()}

    # 0. checks
    print("\n-- 0. checks")
    for f in FLOORS:
        ref = G[f, "pdr"].get(REF, {})
        for grp in ARMS + ("alpha",):
            mine = E[f, grp].get(REF, {})
            same = sum(1 for s in mine if s in ref and abs(mine[s] - ref[s]) <= 1e-9 * max(1.0, abs(ref[s])))
            n_pol = len(E[f, grp])
            print(f"   {f} {grp:7s}: MDD-PTECT = grid pdr on {same} / {len(mine)} seeds; {n_pol} policies, "
                  f"seeds per policy {sorted({len(v) for v in E[f, grp].values()}) or '-'}")
        print(f"   {f} references: pdr {len(ref)} seeds, T15 {len(family(G[f, 'T15']))}, "
              f"T100 {len(family(G.get((f, 'T100'), {})))}")

    # 1. per floor
    print("\n-- 1. per floor, % vs MDD-PTECT (alpha 3 = the law on a homogeneous floor)")
    summary, fam, seed_pct = [], {}, {}
    for f in FLOORS:
        pdr = G[f, "pdr"]
        ref = pdr.get(REF, {})
        if not ref:
            continue
        al = {k: E[f, "alpha"].get(k, {}) for k in ALPHAS}
        al[REF] = ref
        ss = sorted(s for s in ref if all(s in al[k] for k in ALPHAS))
        rows = []
        for k in ("MDD-PTECT@1.5", "MDD-PTECT@0.75"):
            rows.append((f"{k} (offset x{ALPHAS[k] / 3:g})", al[k]))
        if ss:
            tot = {k: sum(al[k][s] for s in ss) for k in ALPHAS}
            kbest = min(tot, key=tot.get)
            rows.append((f"best fixed reachable alpha (hindsight): {kbest}", {s: al[kbest][s] for s in ss}))
            rows.append(("per-instance best of alpha 3 / 1.5 / 0.75 (hindsight)", {s: min(al[k][s] for k in ALPHAS) for s in ss}))
        rows.append(("MDD-TECT (alpha 0, outside the head's reach)", pdr.get("MDD-TECT", {})))
        sv = sorted(s for s in ref if all(s in pdr.get(p, {}) for p in V5))
        if sv:
            tot = {p: sum(pdr[p][s] for s in sv) for p in V5}
            pbest = min(tot, key=tot.get)
            rows.append((f"best v5 pair (in sample): {pbest}", pdr[pbest]))
            rows.append(("per-instance best v5 pair (hindsight)", {s: min(pdr[p][s] for p in V5) for s in sv}))
        for k, src in (("T15", G[f, "T15"]), ("T100", G.get((f, "T100"), {})), *((g, E[f, g]) for g in ARMS)):
            if not ckpts(src):
                continue
            fam[f, k] = family(src)
            each = [compare(v, ref, rng, a.min_n) for v in ckpts(src).values()]
            seed_pct[f, k] = [e["pct"] for e in each if e]
            vals = seed_pct[f, k]
            tag = ""
            if len(vals) >= 2:
                t = stats.ttest_1samp(vals, 0.0)
                tag = f" | seeds {' '.join(f'{v:+.1f}' for v in vals)} (t-test p={t.pvalue:.2g})"
            rows.append((f"{LABEL[k]}, mean of {len(ckpts(src))}", fam[f, k], tag))
        print(f"   {f} (MDD-PTECT mean {np.mean(list(ref.values())):,.1f} k job-s, n {len(ref)})")
        for row in rows:
            lab, x, tag = (row + ("",))[:3]
            r = compare(x, ref, rng, a.min_n) if x else None
            print(f"     {lab:58s} {fmt(r)}{tag}")
            if r:
                summary.append({"floor": f, "row": lab, **{k: round(v, 4) if isinstance(v, float) else v
                                                         for k, v in r.items()}})

    # 2. paired
    print("\n-- 2. paired comparisons (pre-declared)")
    P = {}
    for f in FLOORS:
        parts = []
        for g in ARMS:
            for other in ("T15", "T100"):
                if (f, g) in fam and (f, other) in fam:
                    P[f, g, other] = compare(fam[f, g], fam[f, other], rng, a.min_n)
                    parts.append(f"{g} vs {other} {fmt(P[f, g, other])}")
        if (f, "off-het") in fam and (f, "off-b2") in fam:
            P[f, "het-b2"] = compare(fam[f, "off-het"], fam[f, "off-b2"], rng, a.min_n)
            parts.append(f"off-het vs off-b2 {fmt(P[f, 'het-b2'])}")
        if parts:
            print(f"   {f:5s}" + " | ".join(parts))
    print("   capture of the reachable fixed-alpha room (arm - MDD-PTECT) / (per-instance best alpha - MDD-PTECT), pooled:")
    for f in FLOORS:
        ref = G[f, "pdr"].get(REF, {})
        al = {k: (ref if k == REF else E[f, "alpha"].get(k, {})) for k in ALPHAS}
        for g in ARMS:
            x = fam.get((f, g), {})
            ss = [s for s in x if s in ref and all(s in al[k] for k in ALPHAS)]
            if len(ss) < a.min_n:
                continue
            room = sum(min(al[k][s] for k in ALPHAS) - ref[s] for s in ss)
            gain = sum(x[s] - ref[s] for s in ss)
            print(f"     {f} {g}: room {100 * room / sum(ref[s] for s in ss):+.2f}%, arm {100 * gain / sum(ref[s] for s in ss):+.2f}%"
                  f", capture {gain / room:+.2f}" if room < 0 else f"     {f} {g}: no reachable room (MDD-PTECT best on every instance)")

    # 3. offsets
    print("\n-- 3. offsets chosen per slot (policy rows; law alpha 3): shares x1 / x0.5 / x0.25 / x1.5 / x2, mean log2 "
          "offset, mean played alpha; by regime")
    choices, by_seed = [], {}
    for g in ARMS:
        for f, rows, blocks in (("m15", m15[g][1], m15_blocks), *((f, D[f, g], B.get(f, {})) for f in FLOORS)):
            if not rows:
                continue
            off = np.array([OFFSETS[int(r["alpha_head"])] for r in rows])
            played = np.minimum(4.0, 3.0 * off)
            share = [float((off == v).mean()) for v in OFFSETS]
            reg_l2, reg_a = {}, {}
            for r, o, p in zip(rows, off, played):
                bl = blocks.get(int(r["seed"]))
                reg = regime_of(bl, float(r["sim_time"])) if bl else None
                reg_l2.setdefault(reg, []).append(np.log2(o))
                reg_a.setdefault(reg, []).append(p)
                by_seed.setdefault((g, r["policy"], f), []).append(np.log2(o))
            job = [np.mean([int(r["job_head"]) == k for r in rows]) for k in range(3)]
            mach = np.mean([int(r["machine_head"]) for r in rows])
            regs = " ".join(f"{k} {np.mean(v):+.2f}/a{np.mean(reg_a[k]):.2f}"
                            for k, v in sorted(reg_l2.items(), key=lambda kv: str(kv[0])))
            print(f"   {g:7s} {f:4s} shares {' / '.join(f'{x:.2f}' for x in share)} | log2 off {np.log2(off).mean():+.2f} "
                  f"| alpha {played.mean():.2f} | {regs} | PTECT {mach:.2f} | job SRT/MDD/ATC "
                  f"{' / '.join(f'{x:.2f}' for x in job)}")
            choices.append({"arm": g, "floor": f, "slots": len(rows), "mean_log2_offset": round(float(np.log2(off).mean()), 4),
                            "mean_alpha": round(float(played.mean()), 4),
                            **{f"share_x{v:g}": round(x, 4) for v, x in zip(OFFSETS, share)},
                            **{f"log2_offset_{k}": round(float(np.mean(v)), 4) for k, v in reg_l2.items()},
                            **{f"alpha_{k}": round(float(np.mean(v)), 4) for k, v in reg_a.items()},
                            "ptect_share": round(float(mach), 4)})
    print("   per training seed, mean log2 offset m15 / m50 / m100 / m200:")
    for g in ARMS:
        for k in range(5):
            pol = f"ckpt:{LABEL[g]}_s{k}/checkpoint"
            vals = [by_seed.get((g, pol, f)) for f in ("m15",) + FLOORS]
            print(f"     {g} s{k}: " + " / ".join(f"{np.mean(v):+.2f}" if v else "n/a" for v in vals))

    # 4. criteria
    print("\n-- 4. criteria (pre-declared)")
    res = {}
    for g in ARMS:
        vs15 = [P.get((f, g, "T15")) for f in ("m100", "m200")]
        h = holm(vs15)
        vs_ref = [compare(fam[f, g], G[f, "pdr"][REF], rng, a.min_n) if (f, g) in fam else None for f in FLOORS]
        no_loss = None if any(r is None for r in vs_ref) else not any(worse_sig(r) for r in vs_ref)
        s1 = None if h is None or no_loss is None else (h and no_loss)
        print(f"   S1 {g}: beats T15 at m100 / m200 (Holm): {verdict(h)} "
              f"({' / '.join(fmt(r) for r in vs15)}); not significantly worse than MDD-PTECT on m50 / m100 / m200: "
              f"{verdict(no_loss)} -> {verdict(s1)}")
        s2r = vs_ref[2]
        s2 = None if s2r is None else (s2r["pct"] < 0 and s2r["p"] < 0.05)
        print(f"   S2 {g}: better than MDD-PTECT at m200: {fmt(s2r)} -> {verdict(s2)}")
        pooled15 = [x for k in range(5) for x in by_seed.get((g, f"ckpt:{LABEL[g]}_s{k}/checkpoint", "m15"), [])]
        pooled200 = [x for k in range(5) for x in by_seed.get((g, f"ckpt:{LABEL[g]}_s{k}/checkpoint", "m200"), [])]
        lower = [np.mean(by_seed[g, p, "m200"]) < np.mean(by_seed[g, p, "m15"])
                 for p in (f"ckpt:{LABEL[g]}_s{k}/checkpoint" for k in range(5))
                 if (g, p, "m200") in by_seed and (g, p, "m15") in by_seed]
        s3 = None if not (pooled15 and pooled200 and lower) else (np.mean(pooled200) < np.mean(pooled15) and sum(lower) >= 4)
        print(f"   S3 {g}: mean log2 offset m200 {np.mean(pooled200) if pooled200 else float('nan'):+.2f} vs m15 "
              f"{np.mean(pooled15) if pooled15 else float('nan'):+.2f}; lower for {sum(lower)} / {len(lower)} training "
              f"seeds (need pooled lower and >= 4 of 5) -> {verdict(s3)}")
        res[g] = (s1, s2, s3)
    hb = [P.get((f, "het-b2")) for f in FLOORS]
    if all(r is not None for r in hb):
        better, worse = holm_better(hb)
        pick = "off-het" if len(better) >= 2 and not worse else "off-b2"
        print(f"   D  off-het vs off-b2 (Holm over m50 / m100 / m200): {' / '.join(fmt(r) for r in hb)}; off-het "
              f"significantly better on {len(better)}, worse on {len(worse)} -> carry forward {pick}")
    else:
        print("   D  n/a (floors incomplete)")

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
