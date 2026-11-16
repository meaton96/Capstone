"""
@file analyze.py
@brief eval-alpha-online stage 2 analysis (pre-declared 10-29, README "Analysis"): train-alpha-off-het with the online
       alpha law of ../rq3-alpha-online and the size gate, against the static law + gate on the same instances, the
       final policy, MDD-PTECT and the laws alone; criteria Q1-Q4 and H', and rule N.
Sources (test seeds 0-39, window tardiness in k job-s):
  og        out/<cell>/og_s*/ (this stage; only the cells where the online law changes the pair)
  static    ../eval-alpha-transport-size lg (size-grid cells), ../eval-alpha-offset off-het (15-machine cells)
  T15       the final policy: ../rq3-size-v5, ../rq3-size-grid (size cells), ../eval-alpha-spread final (15 machines)
  laws      ../rq3-alpha-online/out/<cell>/law/ (simulator: static, online, map law alone)
  surfaces  ../rq3-alpha-online/out/<cell>/s<seed>.json on test seeds (held-out cells: the best grid pair)
Outputs: analysis.out (stdout), summary.csv, played.csv.
"""
import argparse
import csv
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parent
_spec = importlib.util.spec_from_file_location("ofs_analyze", R / "eval-alpha-off-size" / "analyze.py")
OFS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OFS)                     # read, ckpts, family, compare, fmt, worse_sig
compare, fmt, worse_sig = OFS.compare, OFS.fmt, OFS.worse_sig
_spec = importlib.util.spec_from_file_location("ao_task", HERE / "task.py")
TASK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TASK)

CELLS = TASK.ORDER[::-1]                          # m15_b2, m15_s1, m50, m75, m100 x3, m200
SMALL = {"m15_b2": "0", "m15_s1": "1"}
STATIC = {c: f"eval-alpha-transport-size/out/{c}/lg_s*" for c in CELLS if c not in SMALL}
STATIC.update({c: f"eval-alpha-offset/out/{s}/off-het" for c, s in SMALL.items()})
T15 = {"m50_r052": OFS.SOURCES["m50"]["T15"], "m100_r052": OFS.SOURCES["m100"]["T15"],
       "m200_r052": OFS.SOURCES["m200"]["T15"], "m75_r052": "rq3-size-grid/m75/cmix_s*",
       "m15_b2": "eval-alpha-spread/out/0/final", "m15_s1": "eval-alpha-spread/out/1/final"}
ONLINE = R / "rq3-alpha-online"
## @brief Stage 1's held-out cells (never fitted on): criterion H' (m15_s1 is also a policy cell).
HELD = ["m75_r035", "m150_r080", "m15_s1", "m50_s1", "m100_s0.5", "m100_s1"]
OFFSETS = [1.0, 0.5, 0.25, 1.5, 2.0]
A = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0)


def read(pattern, seeds, small):
    return OFS.read(glob.glob(str(R / pattern)), seeds, need_done=not small)


def law_runs(cell, seeds):
    """{law: {seed: tardiness}} of the simulator's law runs, and {law: (alpha_full, alpha_short)} of seed rows."""
    out, pairs = {}, {}
    for s in seeds:
        f = ONLINE / "out" / cell / "law" / f"s{s}.json"
        if f.exists():
            d = json.loads(f.read_text())
            for k, v in d.items():
                if not k.startswith("_"):
                    out.setdefault(k, {})[s] = v["total"]
                    pairs.setdefault(k, set()).add((v["alpha_full"], v["alpha_short"]))
    return out, pairs


def best_pair(cell, seeds):
    """Pooled test-seed surface of a held-out cell: ({seed: tardiness at the best pair}, the pair)."""
    per = {}
    for s in seeds:
        f = ONLINE / "out" / cell / f"s{s}.json"
        if f.exists():
            d = json.loads(f.read_text())
            if d.get("_kind") == "surf":
                per[s] = {k: v["total"] for k, v in d.items() if not k.startswith("_")}
    if not per:
        return {}, None
    keys = list(next(iter(per.values())))
    pooled = {k: sum(v[k] for v in per.values()) for k in keys}
    k = min(pooled, key=pooled.get)
    return {s: v[k] for s, v in per.items()}, k


def regime_at(blocks, t):
    for b in blocks:
        if b["start"] <= t < b["end"]:
            return "short" if b["load_profile"] == "short" else "full"
    return "full"


def played(dec, blocks, pair):
    """Policy slots by fleet state: offset shares, mean played alpha (law alpha of the state x offset)."""
    out = {}
    for state in ("full", "short"):
        offs = []
        for r in dec:
            if r["kind"] == "pdr" or int(r["seed"]) not in blocks:
                continue
            if regime_at(blocks[int(r["seed"])], float(r["sim_time"])) != state:
                continue
            base, _, suf = r["rule"].partition("@")
            offs.append(float(suf[1:]) if suf.startswith("x") else 1.0)
        if offs:
            offs = np.array(offs)
            la = pair[0] if state == "full" else pair[1]
            out[state] = {"slots": len(offs), **{f"share_x{o:g}": float(np.mean(offs == o)) for o in OFFSETS},
                          "law_alpha": la, "mean_alpha": float(np.minimum(4.0, la * offs).mean())}
    return out


def holm_p(ps):
    """Holm-adjusted p-values (same order)."""
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main(argv=None):
    global ONLINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--law", default=None, help="law_online.json (default: stage 1's)")
    ap.add_argument("--seeds", default="0-39")
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--online-root", default=None, help="stage 1's folder (its out/; dry runs)")
    a = ap.parse_args(argv)
    ONLINE = Path(a.online_root) if a.online_root else ONLINE
    root = Path(a.root)
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = set(range(lo, hi + 1))
    rng = np.random.default_rng(0)
    spec = json.loads(Path(a.law or ONLINE / "law_online.json").read_text())
    sel, law = spec["selected"], spec["laws"]["online"]
    print("== eval-alpha-online: train-alpha-off-het with the online alpha law and the size gate (twin, test seeds "
          f"{a.seeds})")
    print(f"   online law ({sel}): alpha = min(4, max(0, 3 - 6 sigma) x h(tau / a_t)), full-fleet knots "
          f"{law['knots']['full']}, short-fleet knots {law['knots']['short']}; gate G1 {spec['G1']}")
    print("   window tardiness, % vs MDD-PTECT pooled over instances [95% CI] better/worse Wilcoxon p")

    D, summary, played_rows = {}, [], []
    print("\n-- 0. checks")
    for c in CELLS + [x for x in spec["law_cells"] if x not in CELLS]:
        if c not in spec["cells"]:
            continue
        pair = spec["cells"][c]["pair"]
        d = {"pair_on": pair[sel], "pair_st": pair["S0"], "changed": c in CELLS and TASK.changed(spec, c)}
        d["laws"], lpairs = law_runs(c, seeds)
        lines = [f"pair static {tuple(pair['S0'])} -> online {tuple(round(x, 3) for x in pair[sel])}"
                 f"{'' if d['changed'] else ' (unchanged)'}",
                 "law runs " + ", ".join(f"{k} {len(v)}" for k, v in d["laws"].items())]
        if c in CELLS:
            small = c in SMALL
            st_pol, _, _ = read(STATIC[c], seeds, small)
            d["static_pol"], d["static"] = st_pol, OFS.family(st_pol)
            d["ref"] = st_pol.get("MDD-PTECT", {})
            if c in T15:
                t_pol, _, _ = read(T15[c], seeds, small)
                d["T15_pol"], d["T15"] = t_pol, OFS.family(t_pol)
            if d["changed"]:
                pol, dec, blocks = OFS.read(glob.glob(str(root / "out" / c / "og_s*")), seeds)
                d["og_pol"], d["og"], d["og_dec"], d["og_blocks"] = pol, OFS.family(pol), dec, blocks
                d["og_law"] = pol.get("MDD-PTECT@x1", {})
                h = pol.get("MDD-PTECT", {})
                same = sum(abs(h[s] - d["ref"][s]) <= 1e-6 * max(1.0, abs(d["ref"][s])) for s in h if s in d["ref"])
                lines.append(f"og: {len(OFS.ckpts(pol))} policies, seeds {sorted(len(v) for v in OFS.ckpts(pol).values())}")
                lines.append(f"MDD-PTECT = reference on {same} / {sum(s in d['ref'] for s in h)}")
                sim_on = d["laws"].get("online", {})
                same = sum(abs(d["og_law"][s] - sim_on[s]) <= 1e-6 * max(1.0, abs(sim_on[s]))
                           for s in d["og_law"] if s in sim_on)
                lines.append(f"MDD-PTECT@x1 = simulator online law on {same} / "
                             f"{sum(s in sim_on for s in d['og_law'])}")
                if int(c.split("_")[0][1:]) > TASK.GATE_MACHINES:
                    bad = sum(1 for r in dec if r["kind"] != "pdr" and r["rule"].partition("@")[0] != "MDD-PTECT")
                    lines.append(f"gated, decisions not MDD-PTECT: {bad} of {sum(r['kind'] != 'pdr' for r in dec)}")
                p = played(dec, blocks, d["pair_on"])
                for state, v in p.items():
                    played_rows.append({"cell": c, "state": state, **v})
        else:
            d["ref"] = d["laws"].get("static", {})
        if c in HELD:
            d["best_pair"], d["best_key"] = best_pair(c, seeds)
            lines.append(f"held out: best test-seed pair {d['best_key']} ({len(d['best_pair'])} seeds)")
        D[c] = d
        print(f"   {c:10s} " + "; ".join(lines))

    print("\n-- 1. per cell, % vs MDD-PTECT (alpha 3; held-out cells: vs the static law alone)")
    for c, d in D.items():
        ref = d["ref"]
        if len(ref) < a.min_n:
            print(f"   {c}: no reference yet")
            continue
        print(f"   {c} (reference mean {np.mean(list(ref.values())):,.1f} k job-s, n {len(ref)}; pairs static "
              f"{tuple(d['pair_st'])} online {tuple(round(x, 3) for x in d['pair_on'])})")
        rows = [(f"{k} law alone", v) for k, v in d["laws"].items()]
        if "best_pair" in d and d["best_pair"]:
            rows.append((f"best test-seed pair {d['best_key']} (hindsight)", d["best_pair"]))
        for key, label in (("T15", "final policy T15, mean of 5"), ("static", "static law + gate, mean of 5"),
                           ("og", "og: online law + gate, mean of 5")):
            if d.get(key):
                rows.append((label, d[key]))
        if d.get("og_law"):
            rows.append(("og harness: MDD-PTECT@x1 (online law alone)", d["og_law"]))
        for label, x in rows:
            r = compare(x, ref, rng, a.min_n)
            print(f"     {label:52s} {fmt(r)}")
            if r:
                summary.append({"cell": c, "row": label, **{k: r[k] for k in ("n", "pct", "lo", "hi", "better",
                                                                              "worse", "p")}})

    print("\n-- 2. pre-declared criteria (family of 5, paired over test seeds)")
    verdict = {}

    def show(label, r):
        print(f"     {label:58s} {fmt(r)}")
        return r

    ch = [c for c in CELLS if D.get(c, {}).get("changed")]
    print(f"   changed policy cells: {ch}")
    q1 = [show(f"Q1 {c} og vs static law + gate", compare(D[c]["og"], D[c]["static"], rng, a.min_n)) for c in ch]
    ok = [r for r in q1 if r is not None]
    if ok and len(ok) == len(q1):
        adj = holm_p([r["p"] for r in ok])
        better = sum(1 for r, p in zip(ok, adj) if r["pct"] < 0 and p < 0.05)
        verdict["Q1"] = better * 2 >= len(ok) and not any(worse_sig(r) for r in ok)
        print(f"     Q1: better after Holm on {better} of {len(ok)} changed cells; none worse: "
              f"{not any(worse_sig(r) for r in ok)}")
    else:
        verdict["Q1"] = False
    if D.get("m15_b2", {}).get("changed"):
        r1 = show("Q2 m15_b2 og vs static (off-het)", compare(D["m15_b2"]["og"], D["m15_b2"]["static"], rng, a.min_n))
        r2 = show("Q2 m15_b2 og vs final T15", compare(D["m15_b2"]["og"], D["m15_b2"].get("T15", {}), rng, a.min_n))
        verdict["Q2"] = r1 is not None and r2 is not None and not worse_sig(r1) and not worse_sig(r2)
    else:
        print("     Q2 B2: the online law plays the static pair on B2 (unchanged), PASS by construction")
        verdict["Q2"] = True
    ok = True
    for c in ch:
        r = show(f"Q3 {c} og vs MDD-PTECT", compare(D[c]["og"], D[c]["ref"], rng, a.min_n))
        ok &= r is not None and not worse_sig(r)
    verdict["Q3"] = ok
    ok = True
    for c in ch:
        r = show(f"Q4 {c} og vs the online law alone", compare(D[c]["og"], D[c]["og_law"], rng, a.min_n))
        ok &= r is not None and not worse_sig(r)
    verdict["Q4"] = ok
    ok = True
    held = [c for c in HELD if c in D]
    for c in held:
        d = D[c]
        on, st = d["laws"].get("online", {}), d["laws"].get("static", {})
        r = show(f"H' {c} online law vs best pair {d['best_key']}", compare(on, d["best_pair"], rng, a.min_n))
        ok &= r is not None and r["pct"] <= 2.0
        r2 = show(f"H' {c} online law vs static law", compare(on, st, rng, a.min_n))
        ok &= r2 is not None and not worse_sig(r2)
        rs = compare(st, d["best_pair"], rng, a.min_n)
        if rs is not None and rs["pct"] >= 2.0:
            ok &= r2["pct"] < 0 and r2["p"] < 0.05
            print(f"       (static law is {rs['pct']:+.2f}% vs the best pair there, so the online law must beat it)")
    verdict["H'"] = ok and bool(held)
    print("   reported: the online law alone vs the static law alone on every law cell")
    for c, d in D.items():
        if d["laws"].get("online") and d["laws"].get("static"):
            show(f"   L {c} online vs static law alone", compare(d["laws"]["online"], d["laws"]["static"], rng, a.min_n))
    print("   " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in verdict.items()))
    law_better = None
    pooled = [compare(D[c]["laws"].get("online", {}), D[c]["laws"].get("static", {}), rng, a.min_n) for c in ch]
    if pooled and all(r is not None for r in pooled):
        law_better = all(r["pct"] <= 0 for r in pooled) and any(r["pct"] < 0 and r["p"] < 0.05 for r in pooled)
    if verdict["Q1"] and verdict["Q2"] and verdict["Q3"]:
        n = ("the online law + gate replaces the static law as paper B's alpha policy; next the Unity check of the "
             "online law (Python rule mode) on B2 and one large floor")
    elif not verdict["Q1"] and law_better:
        n = ("the online law alone beats the static law but the policy does not carry it: deploy the law alone above "
             "the gate, or retrain the arm with the online law in the loop")
    elif not verdict["Q1"]:
        n = "the online law adds nothing in the policy: keep the static law + gate"
    else:
        n = "the online law costs on B2 or against MDD-PTECT: apply it above the gate only and report the cost"
    print(f"   N -> {n}")
    if not verdict["H'"]:
        print("   N (H' FAIL) -> the online law is a fit to the fitted floors, not a shown transfer rule")

    print("\n-- 3. what the og policies play, by fleet state: offset shares x1 / x0.5 / x0.25 / x1.5 / x2, law alpha, "
          "mean played alpha")
    for p in played_rows:
        sh = " / ".join(f"{p[f'share_x{o:g}']:.2f}" for o in OFFSETS)
        print(f"   {p['cell']:10s} {p['state']:5s} {p['slots']:6d} slots  {sh} | law {p['law_alpha']:.2f} | "
              f"played {p['mean_alpha']:.2f}")
    for name, rows in (("summary.csv", summary), ("played.csv", played_rows)):
        if rows:
            with open(root / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
