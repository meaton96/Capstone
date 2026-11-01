"""
@file analyze.py
@brief eval-alpha-offset (2026-10-25): pre-declared analysis of the offset arms (train-alpha-off-het / -b2: the spread
       law sets alpha per floor, the policy's third head picks an offset x{1, 0.5, 0.25, 1.5, 2} on it) on 15-machine B2
       floors with machine-speed spread s in {0, 0.25, 0.5, 0.75, 1} (task.py), test seeds 0-39. The references are
       ../eval-alpha-spread's outputs on the same cells and seeds (final policy, train-alpha-het / -b2, the fixed-rule
       grid), and the statistics are its own (compare(): pooled %, 95% instance-bootstrap CI, better / worse, Wilcoxon),
       imported from its analyze.py. Window tardiness; % vs MDD-PTECT (alpha 3).
Steps:
  0. completeness; cross-checks: each new group's MDD-PTECT = eval-alpha-spread's het-group MDD-PTECT bit for bit; the
     law alone (MDD-PTECT@x1) is the same in both new groups and = MDD-PTECT at s 0; episodes.csv speed_sigma = the
     grid's sigma and alpha_law = max(0, 3 - 6 sigma).
  1. per cell: references (the law alone, v7: MDD-PTECT at alpha = max(0, 3 - 6 sigma), continuous; eval-alpha-spread's
     refit law max(0, 1.8 - 4 sigma) on the alpha grid; the hindsight best MDD-PTECT alpha; the per-instance best v6
     combination) and the policy stacks (final, b2, het, off-b2, off-het: per instance the mean over 5 training seeds;
     each training seed's pooled % shown).
  2. paired, pre-declared: off-het vs final, vs het and vs the law alone per cell; off-b2 vs final per cell.
  3. the offsets the off-het / off-b2 policies choose (slot actions): shares, mean log2 offset, mean played alpha
     (law x offset, capped at 4) vs the law, per cell and per regime block (normal / surge / short fleet).
Criteria (pre-declared 10-25, README):
  H1' off-het keeps the het arm's gain on spread floors: at s 0.5, 0.75 and 1 it beats the final policy significantly
      (Wilcoxon, Holm over the 3 cells), and pooled over those 120 instances it is at most 1.0% worse than the het arm.
  H2' off-het costs nothing on B2: at s 0 it is within +1% of the final policy and not significantly worse (a positive
      difference with Wilcoxon p < 0.05 fails).
  Secondary, reported (not part of the verdict): off-b2 under H2''s rule at s 0; H3' the head adds to the law: off-het
  vs the law alone pooled over all 5 cells (better, Wilcoxon p < 0.05); H4' the offset is lower in short-fleet blocks
  than in normal blocks (mean log2 offset, off-het, pooled over cells and per training seed).
Outputs: analysis.out (stdout), summary.csv, offset_choices.csv.
@par Usage
@code{.sh}
.venv/bin/python results/eval-alpha-offset/analyze.py [--root DIR] [--seeds 0-39]
@endcode
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parent
OLD = R / "eval-alpha-spread"
_spec = importlib.util.spec_from_file_location("eas_analyze", OLD / "analyze.py")
EAS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EAS)                     # compare, fmt, load_grid, load_eval, regime_of, GRID_A, MID, V6
compare, fmt, load_grid, load_eval, regime_of = EAS.compare, EAS.fmt, EAS.load_grid, EAS.load_eval, EAS.regime_of

CELLS = EAS.CELLS
SPREAD_CELLS = EAS.SPREAD_CELLS
OFFSETS = [1.0, 0.5, 0.25, 1.5, 2.0]               # v7 ALPHA_OFFSET_VALUES (index order)
REF = EAS.REF                                       # the grid's MDD-PTECT@3
LAW = "MDD-PTECT@x1"
OLD_GROUPS = ("final", "b2", "het")
NEW_GROUPS = ("off-b2", "off-het")
LABEL = {"final": "final policy (v5, cmix)", "b2": "train-alpha-b2", "het": "train-alpha-het",
         "off-b2": "train-alpha-off-b2", "off-het": "train-alpha-off-het"}


def law_alpha(sigma):
    return max(0.0, 3.0 - 6.0 * sigma)


def raw_rows(root, cell, group):
    f = root / "out" / cell / group / "episodes.csv"
    return list(csv.DictReader(open(f))) if f.exists() else []


def holm(results):
    """results: [(name, compare-dict)]; all better and Holm-significant at 0.05."""
    ps = sorted(results, key=lambda t: t[1]["p"])
    return all(r["pct"] < 0 and r["p"] <= 0.05 / (len(ps) - i) for i, (_, r) in enumerate(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--old-root", default=str(OLD), help="eval-alpha-spread's outputs (the references)")
    ap.add_argument("--seeds", default="0-39")
    ap.add_argument("--min-n", type=int, default=5, help="fewest paired instances for a comparison (dry runs: 2)")
    a = ap.parse_args()
    root, old = Path(a.root), Path(a.old_root)
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    rng = np.random.default_rng(0)
    print(f"== eval-alpha-offset: law + offset arms on 15-machine B2 floors with machine-speed spread; seeds {a.seeds}")
    print("   window tardiness, % vs MDD-PTECT (alpha 3) pooled over instances [95% CI] better/worse Wilcoxon p\n")

    # 0. completeness and cross-checks
    G, E, D, problems = {}, {}, {}, []
    for c in CELLS:
        G[c] = load_grid(old, c, seeds)
        for g in OLD_GROUPS:
            E[c, g], D[c, g] = load_eval(old, c, g, seeds)
        for g in NEW_GROUPS:
            E[c, g], D[c, g] = load_eval(root, c, g, seeds)
        miss = [f"grid {len(G[c])}/{len(seeds)}"] if len(G[c]) < len(seeds) else []
        for g in OLD_GROUPS + NEW_GROUPS:
            n = len(E[c, g].get("MDD-PTECT", {}))
            if n < len(seeds):
                miss.append(f"{g} {n}/{len(seeds)}")
        if miss:
            problems.append(f"s {c}: incomplete ({', '.join(miss)})")
    same, n_same, law_same, n_law, sig_ok, n_sig = 0, 0, 0, 0, 0, 0
    for c in CELLS:
        het_old = {r["seed"]: r for r in raw_rows(old, c, "het") if r["policy"] == "MDD-PTECT"}
        laws = {}
        for g in NEW_GROUPS:
            rows = raw_rows(root, c, g)
            for r in rows:
                if int(r["seed"]) not in seeds:
                    continue
                if r["policy"] == "MDD-PTECT" and r["seed"] in het_old:
                    n_same += 1
                    same += all(r[k] == het_old[r["seed"]][k] for k in ("window_tardiness", "return", "decisions"))
                if r["policy"] == LAW:
                    laws.setdefault(r["seed"], []).append(r["window_tardiness"])
                gs = G[c].get(int(r["seed"]))
                if gs is not None:
                    n_sig += 1
                    sig_ok += (float(r["speed_sigma"]) == gs["sigma"]
                               and float(r["alpha_law"]) == law_alpha(gs["sigma"]))
            if c == "0":
                base = {r["seed"]: r["window_tardiness"] for r in rows if r["policy"] == "MDD-PTECT"}
                for r in rows:
                    if r["policy"] == LAW and int(r["seed"]) in seeds:
                        n_law += 1
                        law_same += r["window_tardiness"] == base.get(r["seed"])
        for v in laws.values():
            n_law += 1
            law_same += len(v) == len(NEW_GROUPS) and len(set(v)) == 1
    print(f"-- 0. checks: new groups' MDD-PTECT = eval-alpha-spread het MDD-PTECT bit for bit: {same} / {n_same}")
    print(f"   the law alone: same in both groups, and = MDD-PTECT at s 0: {law_same} / {n_law}")
    print(f"   speed_sigma = grid sigma and alpha_law = max(0, 3 - 6 sigma): {sig_ok} / {n_sig} rows")
    if same != n_same or law_same != n_law or sig_ok != n_sig:
        problems.append("cross-check failed")
    for p in problems:
        print(f"   NOTE {p}")

    # 1. per cell
    summary, stacks, laws, seeds_pct = [], {}, {}, {}
    print("\n-- 1. per cell, % vs MDD-PTECT (alpha 3)")
    for c in CELLS:
        g = G[c]
        if not g:
            continue
        ss = sorted(g)
        ref = {s: g[s]["T"][REF] for s in ss}
        sig = np.array([g[s]["sigma"] for s in ss])
        law = E[c, "off-het"].get(LAW) or E[c, "off-b2"].get(LAW) or {}
        laws[c] = law
        refit_a = EAS.GRID_A[np.searchsorted(EAS.MID, np.maximum(0.0, 1.8 - 4.0 * sig))]
        refit = {s: g[s]["T"][f"MDD-PTECT@{al:g}"] for s, al in zip(ss, refit_a)}
        pt = np.array([[g[s]["T"][f"MDD-PTECT@{al:g}"] for al in EAS.GRID_A] for s in ss])
        bi = int(np.argmin(pt.sum(0)))
        best_a = {s: pt[i, bi] for i, s in enumerate(ss)}
        v6 = np.array([[g[s]["T"][k] for k in EAS.V6] for s in ss])
        inst_v6 = {s: v6[i].min() for i, s in enumerate(ss)}
        rows = [(f"law alone (v7): MDD-PTECT at alpha 3 - 6 sigma, mean {np.mean([law_alpha(x) for x in sig]):.2f}", law),
                (f"refit law 1.8 - 4 sigma on the grid (eval-alpha-spread), mean {refit_a.mean():.2f}", refit),
                (f"best fixed alpha (hindsight): MDD-PTECT@{EAS.GRID_A[bi]:g}", best_a),
                ("per-instance best v6 combo (hindsight)", inst_v6)]
        for grp in OLD_GROUPS + NEW_GROUPS:
            ck = {k: v for k, v in E[c, grp].items() if k.startswith("ckpt:")}
            if not ck:
                continue
            common = sorted(set.intersection(*(set(v) for v in ck.values())))
            stacks[c, grp] = {s: float(np.mean([v[s] for v in ck.values()])) for s in common}
            each = {k: compare(v, ref, rng, a.min_n) for k, v in ck.items()}
            seeds_pct[c, grp] = {k: e["pct"] for k, e in each.items() if e}
            vals = list(seeds_pct[c, grp].values())
            tag = f" (seeds {min(vals):+.1f} .. {max(vals):+.1f})" if vals else ""
            rows.append((f"{LABEL[grp]}, mean of {len(ck)}{tag}", stacks[c, grp]))
        print(f"   s {c} (sigma {sig.mean():.3f}, n {len(ss)})")
        for lab, x in rows:
            r = compare(x, ref, rng, a.min_n) if x else None
            print(f"     {lab:66s} {fmt(r)}")
            if r:
                summary.append({"cell": c, "row": lab, **{k: (round(v, 4) if isinstance(v, float) else v)
                                                          for k, v in r.items()}})

    # 2. paired, pre-declared
    print("\n-- 2. paired comparisons (pre-declared)")
    paired = {}
    for c in CELLS:
        oh = stacks.get((c, "off-het"))
        for name, other in (("final", stacks.get((c, "final"))), ("het", stacks.get((c, "het"))), ("law", laws.get(c))):
            if oh and other:
                paired[c, "off-het", name] = compare(oh, other, rng, a.min_n)
        ob, fin = stacks.get((c, "off-b2")), stacks.get((c, "final"))
        if ob and fin:
            paired[c, "off-b2", "final"] = compare(ob, fin, rng, a.min_n)
        line = " | ".join(f"{g} vs {o} {fmt(r)}" for (cc, g, o), r in paired.items() if cc == c)
        print(f"   s {c:4s} {line}")
    for (c, g, o), r in paired.items():
        if r:
            summary.append({"cell": c, "row": f"PAIRED {g} vs {o}", **r})

    # 3. offsets chosen
    print("\n-- 3. offsets chosen per slot (policy rows): shares x1 / x0.5 / x0.25 / x1.5 / x2, mean log2 offset, "
          "mean played alpha vs the law; by regime")
    choices, by_seed_reg = [], {}
    for grp in ("off-het", "off-b2"):
        for c in CELLS:
            rows = D[c, grp]
            if not rows:
                continue
            off = np.array([OFFSETS[int(r["alpha_head"])] for r in rows])
            law_r = np.array([law_alpha(G[c][int(r["seed"])]["sigma"]) for r in rows])
            played = np.minimum(4.0, law_r * off)
            share = [float((off == v).mean()) for v in OFFSETS]
            reg_l2, reg_a = {}, {}
            for r, o, p in zip(rows, off, played):
                gs = G[c].get(int(r["seed"]))
                reg = regime_of(gs["blocks"], float(r["sim_time"])) if gs else None
                reg_l2.setdefault(reg, []).append(np.log2(o))
                reg_a.setdefault(reg, []).append(p)
                if grp == "off-het":
                    by_seed_reg.setdefault((r["policy"], reg), []).append(np.log2(o))
            mach = np.mean([int(r["machine_head"]) for r in rows])
            job = [np.mean([int(r["job_head"]) == k for r in rows]) for k in range(3)]
            regs = " ".join(f"{k} {np.mean(v):+.2f}/a{np.mean(reg_a[k]):.2f}"
                            for k, v in sorted(reg_l2.items(), key=lambda kv: str(kv[0])))
            print(f"   {grp:7s} s {c:4s} shares {' / '.join(f'{x:.2f}' for x in share)} | log2 off {np.log2(off).mean():+.2f} "
                  f"| alpha {played.mean():.2f} (law {law_r.mean():.2f}) | {regs} | PTECT {mach:.2f} | "
                  f"job SRT/MDD/ATC {' / '.join(f'{x:.2f}' for x in job)}")
            choices.append({"group": grp, "cell": c, "slots": len(rows), "mean_log2_offset": round(float(np.log2(off).mean()), 4),
                            "mean_alpha": round(float(played.mean()), 4), "mean_law_alpha": round(float(law_r.mean()), 4),
                            **{f"share_x{v:g}": round(x, 4) for v, x in zip(OFFSETS, share)},
                            **{f"log2_offset_{k}": round(float(np.mean(v)), 4) for k, v in reg_l2.items()},
                            **{f"alpha_{k}": round(float(np.mean(v)), 4) for k, v in reg_a.items()},
                            "ptect_share": round(float(mach), 4)})

    # criteria
    print("\n-- criteria (pre-declared)")
    h1 = [(c, paired.get((c, "off-het", "final"))) for c in SPREAD_CELLS]
    if all(r for _, r in h1) and all((c, "het") in stacks for c in SPREAD_CELLS):
        ok_holm = holm(h1)
        oh = {(c, s): v for c in SPREAD_CELLS for s, v in stacks[c, "off-het"].items()}
        ht = {(c, s): v for c in SPREAD_CELLS for s, v in stacks[c, "het"].items()}
        rp = compare(oh, ht, rng, a.min_n)
        print(f"   H1' off-het beats final at s 0.5 / 0.75 / 1 (Holm): {'yes' if ok_holm else 'no'}; pooled vs het "
              f"{fmt(rp)} (<= +1.0%: {'yes' if rp['pct'] <= 1.0 else 'no'}) -> "
              f"{'PASS' if ok_holm and rp['pct'] <= 1.0 else 'FAIL'}")
    else:
        print("   H1' n/a (cells missing)")
    r = paired.get(("0", "off-het", "final"))
    if r:
        ok = r["pct"] <= 1.0 and not (r["pct"] > 0 and r["p"] < 0.05)
        print(f"   H2' off-het vs final at s 0: {fmt(r)} -> {'PASS' if ok else 'FAIL'}")
    else:
        print("   H2' n/a")
    print("   secondary (reported):")
    r = paired.get(("0", "off-b2", "final"))
    if r:
        ok = r["pct"] <= 1.0 and not (r["pct"] > 0 and r["p"] < 0.05)
        print(f"     off-b2 vs final at s 0 (H2' rule): {fmt(r)} -> {'ok' if ok else 'fails'}")
    if all((c, "off-het") in stacks and laws.get(c) for c in CELLS):
        oh = {(c, s): v for c in CELLS for s, v in stacks[c, "off-het"].items()}
        lw = {(c, s): v for c in CELLS for s, v in laws[c].items()}
        r = compare(oh, lw, rng, a.min_n)
        print(f"     H3' off-het vs the law alone, all 5 cells pooled: {fmt(r)} -> "
              f"{'yes' if r['pct'] < 0 and r['p'] < 0.05 else 'no'}")
    seeds_short = sorted({p for p, _ in by_seed_reg})
    if seeds_short:
        allv = {reg: [x for (p, rg), v in by_seed_reg.items() if rg == reg for x in v] for reg in ("normal", "short")}
        if all(allv.values()):
            per = [(p, np.mean(by_seed_reg.get((p, "short"), [np.nan])) - np.mean(by_seed_reg.get((p, "normal"), [np.nan])))
                   for p in seeds_short]
            print(f"     H4' mean log2 offset short fleet {np.mean(allv['short']):+.2f} vs normal {np.mean(allv['normal']):+.2f}"
                  f" ({'lower' if np.mean(allv['short']) < np.mean(allv['normal']) else 'not lower'}); per training seed "
                  f"short - normal: {' '.join(f'{d:+.2f}' for _, d in per)}")

    if summary:
        with open(root / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cell", "row", "n", "pct", "lo", "hi", "better", "worse", "p"])
            w.writeheader()
            w.writerows(summary)
    if choices:
        keys = sorted({k for r in choices for k in r}, key=lambda k: (k not in ("group", "cell", "slots"), k))
        with open(root / "offset_choices.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(choices)
    return 0


if __name__ == "__main__":
    sys.exit(main())
