"""
@file analyze.py
@brief eval-alpha-unity analysis (pre-declared 10-30, README "Analysis"): paper B's alpha policy (train-alpha-off-het
       with the static transport law and the size gate) and the law alone in Unity (Python rule mode), next to the
       twin on the same instances; checks K1-K5, criteria U1-U4 and rule N.
Sources (test seeds 0-39, window tardiness in k job-s):
  Unity     out/<cell>/<policy>_s*/ (this experiment; finished tasks only)
  twin b2   ../eval-alpha-offset/out/0/off-het (MDD-PTECT, MDD-PTECT@x1 = alpha 3, the 5 checkpoints)
  twin m100 ../eval-alpha-transport-size/out/m100_r052/lg_s* (MDD-PTECT, the 5 gated checkpoints);
            the law alone: ../rq3-alpha-online/out/m100_r052/law/s<seed>.json "static" (MDD-PTECT at alpha_law)
  K2        ../eval-v5-cmix-unity/runs/base/pc/s<seed>/ (Unity MDD-PTECT on B2, player 5e6e0f87) and its cmix s3 rows
            (the final policy in Unity, reported)
Outputs: analysis.out (stdout), summary.csv, played.csv.
"""
import argparse
import csv
import glob
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parent
_spec = importlib.util.spec_from_file_location("eas_analyze", R / "eval-alpha-spread" / "analyze.py")
EAS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EAS)                     # compare, fmt (as every alpha-study analysis)
compare, fmt = EAS.compare, EAS.fmt
OFFSETS = [1.0, 0.5, 0.25, 1.5, 2.0]
CKS = [f"ck{k}" for k in range(5)]
TWIN_B2 = R / "eval-alpha-offset" / "out" / "0" / "off-het"
TWIN_M100 = sorted(glob.glob(str(R / "eval-alpha-transport-size" / "out" / "m100_r052" / "lg_s*")))
LAWRUNS = R / "rq3-alpha-online" / "out" / "m100_r052" / "law"
CMIX_UNITY = R / "eval-v5-cmix-unity" / "runs" / "base" / "pc"
## @brief Law alpha per cell when no finished task's cond.json gives it: round(3 h, 3) from law.json (task.py's rule).
LAW_ALPHA = {"b2": 3.0, "m100_r052": round(3.0 * json.loads((R / "rq3-alpha-transport" / "law.json").read_text())
                                           ["cells"]["m100_r052"]["h"], 3)}


def worse_sig(r):
    return r is not None and r["pct"] > 0 and r["p"] < 0.05


def better_sig(r):
    return r is not None and r["pct"] < 0 and r["p"] < 0.05


def ck_key(policy):
    m = re.search(r"train-alpha-off-het_s(\d)", policy)
    return f"ck{m.group(1)}" if m else None


def read_unity(root, cell, seeds):
    """{key: {seed: row}} of finished tasks, decision rows {key: [rows]}, blocks {seed: blocks}, conds [cond.json]."""
    rows, dec, blocks, conds = {}, {}, {}, []
    for d in sorted((root / "out" / cell).glob("*_s*")):
        if not (d / "DONE").exists():
            continue
        key = d.name.rsplit("_s", 1)[0]
        for r in csv.DictReader(open(d / "episodes.csv")):
            if int(r["seed"]) in seeds:
                rows.setdefault(key, {})[int(r["seed"])] = r
        if (d / "decisions.csv").exists() and key.startswith("ck"):
            dec.setdefault(key, []).extend(r for r in csv.DictReader(open(d / "decisions.csv"))
                                           if r["kind"] != "pdr" and int(r["seed"]) in seeds)
        if (d / "blocks.json").exists():
            blocks.update({int(k): v for k, v in json.loads((d / "blocks.json").read_text()).items()})
        c = json.loads((d / "cond.json").read_text())
        c["_decisions"] = sum(int(r["decisions"]) for r in csv.DictReader(open(d / "episodes.csv")))
        conds.append(c)
    return rows, dec, blocks, conds


def tard(rows):
    return {k: {s: float(r["window_tardiness"]) / 1000.0 for s, r in v.items()} for k, v in rows.items()}


def read_twin(dirs, seeds):
    """{key: {seed: tardiness}} (keys ck0-4, pdr, law) and decision rows {key: [rows]} of twin evaluate.py dirs."""
    out, dec = {}, {}
    for d in dirs:
        d = Path(d)
        if not (d / "episodes.csv").exists():
            continue
        for r in csv.DictReader(open(d / "episodes.csv")):
            s = int(r["seed"])
            if s not in seeds:
                continue
            p = r["policy"]
            key = ck_key(p) or {"MDD-PTECT": "pdr", "MDD-PTECT@x1": "law"}.get(p)
            if key:
                out.setdefault(key, {})[s] = float(r["window_tardiness"]) / 1000.0
        if (d / "decisions.csv").exists():
            for r in csv.DictReader(open(d / "decisions.csv")):
                k = ck_key(r["policy"])
                if k and r["kind"] != "pdr" and int(r["seed"]) in seeds:
                    dec.setdefault(k, []).append(r)
    return out, dec


def family(pol):
    """Per-instance mean over ck0-ck4 on the seeds all five have."""
    ck = [pol[k] for k in CKS if k in pol]
    if len(ck) < 5:
        return {}
    common = sorted(set.intersection(*(set(v) for v in ck)))
    return {s: float(np.mean([v[s] for v in ck])) for s in common}


def regime_at(blocks, t):
    for b in blocks:
        if b["start"] <= t < b["end"]:
            return "short" if b["load_profile"] == "short" else "full"
    return "full"


def played(dec, blocks, law_alpha):
    """Policy slots by fleet state: offset shares and the mean played alpha (law alpha x offset, capped at 4)."""
    out = {}
    for state in ("full", "short"):
        offs = []
        for r in dec:
            s = int(r["seed"])
            if s in blocks and regime_at(blocks[s], float(r["sim_time"])) == state:
                _, _, suf = r["rule"].partition("@")
                offs.append(float(suf[1:]) if suf.startswith("x") else 1.0)
        if offs:
            offs = np.array(offs)
            out[state] = {"slots": len(offs), **{f"share_x{o:g}": float(np.mean(offs == o)) for o in OFFSETS},
                          "mean_alpha": float(np.minimum(4.0, law_alpha * offs).mean())}
    return out


def gap(u, t, ub, tb, rng):
    """Unity % minus twin % (each vs its own MDD-PTECT), seeds resampled jointly: (gap, lo, hi, n)."""
    s = sorted(set(u) & set(t) & set(ub) & set(tb))
    if len(s) < 5:
        return None
    U, T, UB, TB = (np.array([d[k] for k in s]) for d in (u, t, ub, tb))
    idx = rng.integers(0, len(s), (5000, len(s)))
    boot = 100 * (U[idx].sum(1) / UB[idx].sum(1) - T[idx].sum(1) / TB[idx].sum(1))
    g = 100 * (U.sum() / UB.sum() - T.sum() / TB.sum())
    return g, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--seeds", default="0-39")
    ap.add_argument("--min-n", type=int, default=5, help="fewest paired instances for a comparison (dry runs: 1)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    lo, _, hi = a.seeds.partition("-")
    seeds = set(range(int(lo), int(hi or lo) + 1))
    rng = np.random.default_rng(20261030)
    summary, played_rows = [], []

    def row(cell, label, r):
        print(f"     {label:56s} {fmt(r)}")
        if r is not None:
            summary.append({"cell": cell, "row": label, **{k: r[k] for k in ("n", "pct", "lo", "hi", "better", "worse",
                                                                              "p")}})
        return r

    print("== eval-alpha-unity: the static law + gate (train-alpha-off-het, v7) and the law alone in Unity (Python rule "
          f"mode), next to the twin; seeds {a.seeds}")
    print("   window tardiness, % vs MDD-PTECT in the same simulator, pooled over instances [95% CI] better/worse "
          "Wilcoxon p; family = per-instance mean of s0-s4")

    U, T, K, DEC, BLK, LAWA = {}, {}, {}, {}, {}, {}
    for cell in ("b2", "m100_r052"):
        rows, dec, blocks, conds = read_unity(root, cell, seeds)
        U[cell], DEC[cell, "unity"], BLK[cell] = tard(rows), dec, blocks
        K[cell] = (rows, conds)
        las = {c["law_alpha"] for c in conds}
        LAWA[cell] = las.pop() if len(las) == 1 else LAW_ALPHA[cell]
    T["b2"], DEC["b2", "twin"] = read_twin([TWIN_B2], seeds)
    T["m100_r052"], DEC["m100_r052", "twin"] = read_twin(TWIN_M100, seeds)
    for s in seeds:
        f = LAWRUNS / f"s{s}.json"
        if f.exists():
            T["m100_r052"].setdefault("law", {})[s] = json.loads(f.read_text())["static"]["total"]

    # ---------------------------------------------------------------- checks
    print("\n-- 0. checks")
    ok = {}
    b2rows = K["b2"][0]
    same = [s for s in seeds if s in b2rows.get("pdr", {}) and s in b2rows.get("py3", {})]
    eq = [s for s in same if all(b2rows["pdr"][s][f] == b2rows["py3"][s][f]
                                 for f in ("window_tardiness", "jobs_exited", "decisions", "makespan", "instance_hash"))]
    ok["K1"] = bool(same) and len(eq) == len(same)
    print(f"   K1 b2: Python MDD-PTECT@3 equals C# MDD-PTECT on {len(eq)} / {len(same)} seeds (tardiness, jobs exited, "
          f"decisions, makespan, instance hash)")
    ref = {}
    for d in sorted(CMIX_UNITY.glob("s*")):
        if (d / "episodes.csv").exists():
            for r in csv.DictReader(open(d / "episodes.csv")):
                ref.setdefault(r["policy"], {})[int(r["seed"])] = r
    pd_ = b2rows.get("pdr", {})
    k2 = [s for s in pd_ if s in ref.get("MDD-PTECT", {})]
    k2eq = [s for s in k2 if pd_[s]["window_tardiness"] == ref["MDD-PTECT"][s]["window_tardiness"]
            and pd_[s]["instance_hash"] == ref["MDD-PTECT"][s]["instance_hash"]]
    ok["K2"] = bool(k2) and len(k2eq) == len(k2)
    print(f"   K2 b2: C# MDD-PTECT equals eval-v5-cmix-unity's base rows (player 5e6e0f87) on {len(k2eq)} / {len(k2)} "
          "seeds (tardiness and instance hash)")
    mrows, mconds = K["m100_r052"]
    gate_rows = [r for k, v in DEC["m100_r052", "unity"].items() for r in v]
    off_gate = sum(1 for r in gate_rows if not r["rule"].startswith("MDD-PTECT"))
    ok["K3"] = bool(gate_rows) and off_gate == 0
    print(f"   K3 m100_r052: gated checkpoint slots off MDD-PTECT on the pair heads: {off_gate} of {len(gate_rows)}")
    short = []                                           # tasks with fewer Python picks than measured decisions
    for cell in ("b2", "m100_r052"):
        for c in K[cell][1]:
            if c["policy"] != "pdr" and sum(c.get("python_picks", {}).values()) < c["_decisions"]:
                short.append((cell, c["policy"], c["seeds"]))
    n_py = sum(1 for cell in ("b2", "m100_r052") for c in K[cell][1] if c["policy"] != "pdr")
    ok["K4"] = n_py > 0 and not short
    print(f"   K4 every measured decision picked in Python (picks >= decisions; the extra picks are Unity's discarded "
          f"startup episode): {n_py - len(short)} / {n_py} policy and Python-rule tasks" + (f" short: {short[:5]}" if short else ""))
    bad = [(c, k, s) for c in ("b2", "m100_r052") for k, v in K[c][0].items() for s, r in v.items()
           if r["deadlock"] == "True" or r["timed_out"] == "True" or r["tick_error"] == "True"]
    ok["K5"] = not bad
    print(f"   K5 deadlock / time-out / tick error rows: {len(bad)}" + (f" {bad[:10]}" if bad else ""))
    for cell in ("b2", "m100_r052"):
        print(f"   {cell}: law alpha {LAWA[cell]}; Unity episodes done {sum(len(v) for v in K[cell][0].values())}")
    checks_pass = all(ok.values())
    print("   " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in ok.items()))
    if not checks_pass:
        print("   checks fail: the criteria below are printed for diagnosis only, not read (README)")

    # ---------------------------------------------------------------- per cell
    print("\n-- 1. per cell, % vs MDD-PTECT")
    FAM = {}
    for cell, extra in (("b2", ("py3",)), ("m100_r052", ("law",))):
        for sim, D in (("Unity", U[cell]), ("twin", T[cell])):
            ref_ = D.get("pdr", {})
            print(f"   {cell} {sim} (MDD-PTECT mean {np.mean(list(ref_.values())) if ref_ else float('nan'):,.1f} k job-s, "
                  f"n {len(ref_)})")
            for k in CKS:
                row(f"{cell}/{sim}", f"{k}", compare(D.get(k, {}), ref_, rng, a.min_n))
            FAM[cell, sim] = family(D)
            row(f"{cell}/{sim}", "family s0-s4 (the static law + gate)", compare(FAM[cell, sim], ref_, rng, a.min_n))
            for k in extra:
                lab = {"py3": "MDD-PTECT@3 (Python; = MDD-PTECT)", "law": f"the law alone, MDD-PTECT@{LAWA[cell]}"}[k]
                if sim == "twin" and k == "py3":
                    lab, k = "MDD-PTECT@x1 (= alpha 3)", "law"
                row(f"{cell}/{sim}", lab, compare(D.get(k, {}), ref_, rng, a.min_n))
        prem = compare(U[cell].get("pdr", {}), T[cell].get("pdr", {}), rng, a.min_n)
        row(cell, "MDD-PTECT Unity vs twin (the Unity premium)", prem)
        g = gap(FAM[cell, "Unity"], FAM[cell, "twin"], U[cell].get("pdr", {}), T[cell].get("pdr", {}), rng)
        if g:
            print(f"     family: Unity % minus twin %                             {g[0]:+.2f} pp [{g[1]:+.2f}, {g[2]:+.2f}] "
                  f"n {g[3]}")
        if cell == "m100_r052":
            g = gap(U[cell].get("law", {}), T[cell].get("law", {}), U[cell].get("pdr", {}), T[cell].get("pdr", {}), rng)
            if g:
                print(f"     law alone: Unity % minus twin %                          {g[0]:+.2f} pp [{g[1]:+.2f}, "
                      f"{g[2]:+.2f}] n {g[3]}")
    fin = {s: float(r["window_tardiness"]) / 1000.0 for s, r in ref.get("ckpt:train-v5-cmix_s3/checkpoint", {}).items()
           if s in seeds}
    print("   b2 Unity, reported: the final policy (train-v5-cmix s3, eval-v5-cmix-unity base rows)")
    row("b2/Unity", "final policy cmix s3 vs MDD-PTECT", compare(fin, U["b2"].get("pdr", {}), rng, a.min_n))
    row("b2/Unity", "family vs the final policy", compare(FAM["b2", "Unity"], fin, rng, a.min_n))

    # ---------------------------------------------------------------- criteria
    print("\n-- 2. pre-declared criteria (Unity, family of 5, paired over test seeds)")
    v = {}
    r1 = row("b2", "U1 b2 family vs MDD-PTECT (not significantly worse)",
             compare(FAM["b2", "Unity"], U["b2"].get("pdr", {}), rng, a.min_n))
    v["U1"] = r1 is not None and not worse_sig(r1)
    r2 = row("m100_r052", "U2 m100 family vs MDD-PTECT (significantly better)",
             compare(FAM["m100_r052", "Unity"], U["m100_r052"].get("pdr", {}), rng, a.min_n))
    v["U2"] = better_sig(r2)
    r3 = row("m100_r052", "U3 m100 the law alone vs MDD-PTECT (significantly better)",
             compare(U["m100_r052"].get("law", {}), U["m100_r052"].get("pdr", {}), rng, a.min_n))
    v["U3"] = better_sig(r3)
    rt_ = compare(FAM["m100_r052", "twin"], T["m100_r052"].get("pdr", {}), rng, a.min_n)
    if r2 is not None and rt_ is not None and rt_["pct"] < 0:
        v["U4"] = r2["pct"] <= 0.5 * rt_["pct"]
        print(f"     U4 m100 family keeps at least half its twin gain: Unity {r2['pct']:+.2f}% vs half of twin "
              f"{0.5 * rt_['pct']:+.2f}%")
    else:
        v["U4"] = False
        print("     U4 m100: twin or Unity family missing, or the twin shows no gain on these seeds")
    print("   " + ", ".join(f"{k} {'PASS' if x else 'FAIL'}" for k, x in v.items()))
    if not checks_pass:
        n = "checks fail: fix and rerun before reading the criteria"
    elif v["U1"] and v["U2"] and v["U3"] and v["U4"]:
        n = ("the static law + gate and the law alone hold in Unity on B2 and at 100 machines: paper B states both as "
             "Unity results; no further Unity work for paper B")
    elif v["U3"] and not (v["U2"] and v["U4"]):
        n = ("the law alone holds in Unity at 100 machines but the offset head's extra does not: paper B's Unity claim "
             "is the law alone there, the policy is a twin result at 100 machines")
    elif not v["U3"]:
        n = ("the transport term does not carry to Unity at 100 machines: report the law as a twin result and diagnose "
             "with the MDD-PTECT Unity premium (congestion) before any further alpha work at scale")
    else:
        n = "(no case)"
    print(f"   N -> {n}")
    if checks_pass and not v["U1"]:
        print("   N (U1 FAIL) -> the policy costs on B2 in Unity: paper B reports it as a twin result on B2")

    # ---------------------------------------------------------------- what the policies play
    print("\n-- 3. what the checkpoints play, by fleet state: offset shares x1 / x0.5 / x0.25 / x1.5 / x2, mean played "
          "alpha (law alpha x offset, capped at 4)")
    for cell in ("b2", "m100_r052"):
        for sim in ("Unity", "twin"):
            dec = [r for k, rows_ in DEC[cell, sim.lower()].items() for r in rows_]
            for state, p in played(dec, BLK[cell], LAWA[cell]).items():
                sh = " / ".join(f"{p[f'share_x{o:g}']:.2f}" for o in OFFSETS)
                print(f"   {cell:10s} {sim:5s} {state:5s} {p['slots']:6d} slots  {sh} | played {p['mean_alpha']:.2f}")
                played_rows.append({"cell": cell, "sim": sim, "state": state, **p})
    for name, rows_ in (("summary.csv", summary), ("played.csv", played_rows)):
        if rows_:
            with open(root / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_[0]))
                w.writeheader()
                w.writerows(rows_)


if __name__ == "__main__":
    main()
