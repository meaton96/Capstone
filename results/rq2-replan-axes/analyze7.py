"""
@file analyze7.py
@brief rq2-replan-axes, axis 7 (README §19, approved by the user 11-10 as proposed): F7a, the rate limit, from the
       stored outputs and the dry run's timings; and for 7b, the horizon arm (out7/): C7, H1, H2 and the reported items.
       `extension` names the floors where H 3,600 s fails H1 (launch7.sh then adds H 7,200 s there).

Pre-declared (§19):
- F7a: on one core, the real-time interval Δ_rt keeps >= 85% of the best gain at all 35 conditions. Δ_rt = 2 x the mean
  serial wall time per re-plan of the condition's D1 60 s tasks (cluster cores), in D1; the kept share as fit.py's
  (gain linear in log D1; below 60 s read at 60 s). With 6 forks Δ_rt is divided by the floor's measured speed-up
  (dryrun7/time.out at the window start; m20 and m30, not measured, take B2's, the lowest).
- C7: every 7b task's MDD-PTECT equals its stored full-horizon task's (same level, D1, seed).
- H1: K_H = g(D1 60, H) / g(D1 60, ∞) >= 0.85 at all three floors (point estimate; 90% interval reported). H_min is the
  shortest H that passes. Extension: H 7,200 s at a floor where H 3,600 s fails; a floor not extended counts as passing
  at 7,200 s (its 3,600 s passed).
- H2 at H 1,800 s: d = g(D1 60, 1,800) − g(Δ_eq, ∞), Δ_eq the full-horizon interval of equal rollout volume, read on the
  stored curve linearly in log D1. PASS: the 90% interval of d above 0 at >= 2 of 3 floors; FAIL: below 0 at >= 2;
  indeterminate otherwise.
g = 100 (1 − Σ_s T_s / Σ_s T_s(plan-once)) pooled over seeds 0-39; bootstrap 2,000 joint resamples of seeds across the
floors (stream 1910, this analysis's own).
@par Usage
@code{.sh}
.venv/bin/python results/rq2-replan-axes/analyze7.py [--root DIR]            # everything -> <root>/analysis7.out + stdout
.venv/bin/python results/rq2-replan-axes/analyze7.py frontier                # F7a only (stdout)
.venv/bin/python results/rq2-replan-axes/analyze7.py extension [--root DIR]  # floors needing H 7,200 s (stdout)
@endcode
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit as F                                                    # noqa: E402
import run7 as R7                                                  # noqa: E402

GRID = (60, 120, 180, 300, 450, 600, 900, 1800, 3600)   # the stored full-horizon grid (D1) on all three floors
SEEDS = R7.SEEDS7
FLOORS = tuple(R7.FLOORS7)
K_LINE, H2_H, B, STREAM = 0.85, 1800, 2000, 1910
W = float(R7.R.WINDOW)
FLOOR_M = {"proc-k1": 15, "size-m20": 20, "size-m30": 30, "size-m50": 50, "size-m100": 100, "size-m200": 200}
# run7.py versions whose 7b outputs are the same planner: 11-10 (0c6c571f, 590 tasks before the 11-11 reboot) and the
# 11-13 guard fix (1dfd1e7f; it only stopped raising "not real time" on a late first decision at T_c = 0)
RUN7_OK = ("0c6c571f5389", "1dfd1e7f10b2")


def jload(p):
    return json.loads(Path(p).read_text())


def resamples():
    return np.random.default_rng(STREAM).integers(0, len(SEEDS), (B, len(SEEDS)))


def q90(v):
    return float(np.percentile(v, 5)), float(np.percentile(v, 95))


def gain(T, once, idx=None):
    """Pooled gain over plan-once, points; T (..., seeds) and once (seeds,), optionally under the seed index idx."""
    if idx is not None:
        T, once = T[..., idx], once[idx]
    return -100.0 * (T.sum(-1) / once.sum(-1) - 1.0)


def d_eq(d1, h):
    """The full-horizon interval of equal rollout volume: 3 W (n + 1) = 6 W rollout_cw(d1, h)."""
    n = 2.0 * R7.rollout_cw(d1, h) - 1.0
    return W / n


# ── F7a: the rate limit ──────────────────────────────────────────────────────────────────────────────────────────

def timings(path=HERE / "dryrun7" / "time.out"):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {FLOOR_M[r["level"]]: r for r in rows if r["at"] == 0.0}


def frontier():
    out = []
    tim = timings()
    speed = {m: t["hfull_par1_s"] / t["hfull_par6_s"] for m, t in tim.items()}
    table = pickle.loads((HERE / "fit_out" / "table.pkl").read_bytes())
    conds = table["conds"]
    rows = F.rows_of(table, np.arange(len(SEEDS)))
    res = []
    for c, r in zip(conds, rows):
        rt = 2.0 * float(np.mean(c["tc"]))                       # serial window-start re-plan, cluster core (s)
        sp = speed.get(c["m"], speed[15])
        d1 = rt / c["scale"]
        res.append({"label": c["label"], "m": c["m"], "rt": rt, "d1": d1, "kept": float(F.kept(r, d1)),
                    "kept_par": float(F.kept(r, d1 / sp)), "speed": sp, "best_d": float(r["ds"][int(np.argmax(r["g"]))])})
    # bootstrap: the share of resamples in which every condition keeps >= 85% (fit.py's 2,000 joint resamples)
    idxs = F.resamples()
    allk, all60 = [], []          # every condition keeps >= 85% at Δ_rt / at D1 60 s (the comparator: seed noise only)
    for idx in idxs:
        ok = ok60 = True
        for c, x in zip(conds, res):
            g = -100.0 * (c["real"][:, idx].sum(1) / c["ref"][idx].sum() - 1.0)
            d1 = 2.0 * float(np.mean(c["tc"][idx])) / c["scale"]
            ok &= bool(np.interp(np.log(d1), np.log(c["ds"]), g) / g.max() >= K_LINE - 1e-12)
            ok60 &= bool(g[0] / g.max() >= K_LINE - 1e-12)
        allk.append(ok)
        all60.append(ok60)
    n_ok = sum(x["kept"] >= K_LINE - 1e-12 for x in res)
    out.append("F7a, the rate limit (README §19; stored outputs, fit_out/table.pkl, dryrun7/time.out)")
    out.append(f"  Δ_rt = 2 x the mean serial wall time per re-plan at D1 60 s (cluster cores), in D1; 6 forks: Δ_rt / "
               f"speed-up ({', '.join(f'm{m} {s:.1f}' for m, s in sorted(speed.items()))}; m20, m30 take m15's)")
    out.append(f"  {'condition':34s} {'Δ_rt (s)':>9s} {'D1':>7s} {'kept, 1 core':>13s} {'kept, 6 forks':>14s} "
               f"{'best at D1':>10s}")
    for x in sorted(res, key=lambda x: x["kept"]):
        out.append(f"  {x['label']:34s} {x['rt']:9.1f} {x['d1']:7.1f} {x['kept']:13.1%} {x['kept_par']:14.1%} "
                   f"{x['best_d']:10.0f}")
    worst = min(res, key=lambda x: x["kept"])
    out.append(f"  conditions with Δ_rt above 60 s (D1): "
               + (", ".join(f"{x['label']} ({x['d1']:.0f} s)" for x in res if x["d1"] > 60) or "none"))
    out.append(f"  min kept share, 1 core: {worst['kept']:.1%} ({worst['label']}); all 35 keep >= 85% in "
               f"{np.mean(allk):.1%} of {len(idxs)} resamples (fit.py's stream), against {np.mean(all60):.1%} for "
               f"D1 60 s everywhere (seed noise of the finest point, no compute in it)")
    out.append(f"  F7a {'PASS' if n_ok == len(res) else 'FAIL'}: {n_ok} of {len(res)} conditions keep >= 85% on one core")
    out.append("  Real-time interval per floor (window-start re-plan, full horizon; S = simulated seconds per wall "
               "second on one core, 6 forks x 6 h / serial time):")
    clus = {c["m"]: 2.0 * float(np.mean(c["tc"])) for c in conds if c["label"] in ("proc k 1", "size m 50",
                                                                                     "size m 100")}
    out.append(f"  {'floor':6s} {'local, 1 core':>14s} {'local, 6 forks':>15s} {'H 1 h, 1 core':>14s} "
               f"{'H 1 h, 6 forks':>15s} {'cluster, 1 core':>16s} {'S local':>9s}")
    for m in sorted(tim):
        t = tim[m]
        out.append(f"  m{m:<5d} {t['hfull_par1_s']:13.1f}s {t['hfull_par6_s']:14.1f}s {t['h3600_par1_s']:13.1f}s "
                   f"{t['h3600_par6_s']:14.1f}s {(f'{clus[m]:.1f}s' if m in clus else '–'):>16s} "
                   f"{6 * W / t['hfull_par1_s']:9.0f}")
    return out, {"pass": n_ok == len(res), "n_ok": n_ok, "rows": res, "all_share": float(np.mean(allk)),
                 "all60_share": float(np.mean(all60))}


# ── 7b: the horizon arm ──────────────────────────────────────────────────────────────────────────────────────────

def load7(root):
    """Stored references and 7b outputs per floor: once (40,), full (grid, 40), mdd_full, and h[(d1, H)] (40,) when
    all 40 seeds exist, with the per-task records."""
    data = {}
    for f in FLOORS:
        base = R7.FLOORS7[f]
        once = np.array([jload(HERE / "out" / f"{base}-fixed" / f"s{s}.json")["best"] for s in SEEDS])
        full = {d: [jload(HERE / "out" / f"{base}-d{d}" / f"s{s}.json") for s in SEEDS] for d in GRID}
        h, recs, partial = {}, {}, {}
        for H in R7.H7 + (R7.H7_EXT,):
            for d in R7.D1_7:
                ps = [R7.out_path(f"{base}-d{d}+h{H}", s, root) for s in SEEDS]
                have = [p for p in ps if p.exists()]
                if len(have) == len(ps):
                    recs[(d, H)] = [jload(p) for p in ps]
                    h[(d, H)] = np.array([x["real"] for x in recs[(d, H)]])
                elif have:
                    partial[(d, H)] = len(have)
        data[f] = {"base": base, "once": once, "full": np.array([[x["real"] for x in full[d]] for d in GRID]),
                   "full_rec": full, "h": h, "recs": recs, "partial": partial}
    return data


def k_point(D, f, d, H, idx=None):
    return gain(D[f]["h"][(d, H)], D[f]["once"], idx) / gain(D[f]["full"][GRID.index(d)], D[f]["once"], idx)


def extension(root):
    """Floors where H 3,600 s fails H1 (K < 0.85 at D1 60 s); exits 2 if any D1 60 s, H 3,600 s task is missing."""
    D = load7(root)
    miss = [f for f in FLOORS if (60, 3600) not in D[f]["h"]]
    if miss:
        print(f"extension: D1 60 s, H 3,600 s incomplete at {miss}", file=sys.stderr)
        sys.exit(2)
    return [f for f in FLOORS if k_point(D, f, 60, 3600) < K_LINE - 1e-12]


def horizon(root):
    out = ["", "7b, the horizon arm (README §19): out7/ under " + str(root)]
    D = load7(root)
    base_need = [(d, H) for H in R7.H7 for d in R7.D1_7]
    incomplete = {f: [k for k in base_need if k not in D[f]["h"]] for f in FLOORS}
    if any(incomplete.values()):
        out.append("  incomplete: " + "; ".join(f"{f} missing {len(v)} of {len(base_need)} (D1, H) sets "
                                                f"(partial: {D[f]['partial']})" for f, v in incomplete.items() if v))
        out.append("  readings not read until every base task exists")
        return out, {"complete": False}
    # C7
    n, bad = 0, []
    for f in FLOORS:
        for (d, H), recs in D[f]["recs"].items():
            for s, x in zip(SEEDS, recs):
                ref = D[f]["full_rec"][d][s]["mdd_ptect"]
                n += 1
                if abs(x["mdd_ptect"] - ref) > 1e-6 * max(1.0, abs(ref)):
                    bad.append((f, d, H, s, x["mdd_ptect"], ref))
    c7 = not bad
    out.append(f"  C7 {'PASS' if c7 else 'FAIL'}: MDD-PTECT of {n} tasks against their stored full-horizon tasks, "
               f"{len(bad)} mismatches" + (f" (first: {bad[0]})" if bad else ""))
    if not c7:
        out.append("  C7 failed: the readings below are not read (rule 1: stop, fix, rerun)")
    idxs = resamples()
    ext = [f for f in FLOORS if (60, R7.H7_EXT) in D[f]["h"]]
    hs = R7.H7 + ((R7.H7_EXT,) if ext else ())
    # K_H tables
    out.append(f"  Kept share of the full-horizon gain, K_H = g(D1, H) / g(D1, ∞) [90%] (gain over plan-once in points "
               f"beside it; floors extended to 7,200 s: {ext or 'none'})")
    K = {}
    for f in FLOORS:
        for d in R7.D1_7:
            gf = gain(D[f]["full"][GRID.index(d)], D[f]["once"])
            cells = [f"∞ {gf:+.2f}"]
            for H in hs:
                if (d, H) not in D[f]["h"]:
                    cells.append(f"{H}: –")
                    continue
                kp = float(k_point(D, f, d, H))
                kb = np.array([k_point(D, f, d, H, idx) for idx in idxs])
                lo, hi = q90(kb)
                K[(f, d, H)] = (kp, lo, hi)
                cells.append(f"{H}: {kp:.1%} [{lo:.1%}, {hi:.1%}] {gain(D[f]['h'][(d, H)], D[f]['once']):+.2f}")
            out.append(f"    {f:5s} D1 {d:4d}  " + "  ".join(cells))
    # H1
    status = {}
    for H in hs:
        st = {}
        for f in FLOORS:
            if (f, 60, H) in K:
                st[f] = "pass" if K[(f, 60, H)][0] >= K_LINE - 1e-12 else "fail"
            elif H == R7.H7_EXT:
                st[f] = "pass (3,600 s passed)" if K[(f, 60, 3600)][0] >= K_LINE - 1e-12 else "missing"
            else:
                st[f] = "missing"
        status[H] = st
        out.append(f"  H1 at H {H:,} s: " + ", ".join(f"{f} {K[(f, 60, H)][0]:.1%} {st[f]}" if (f, 60, H) in K
                                                      else f"{f} {st[f]}" for f in FLOORS)
                   + f" -> {'PASS' if all(v.startswith('pass') for v in st.values()) else 'FAIL'}")
    passing = [H for H in hs if all(v.startswith("pass") for v in status[H].values())]
    h_min = passing[0] if passing else None
    need_ext = [f for f in FLOORS if K[(f, 60, 3600)][0] < K_LINE - 1e-12]
    out.append(f"  H1: H_min = {f'{h_min:,} s' if h_min else 'none (fails at every horizon run)'}; extension rule "
               f"calls for 7,200 s at {need_ext or 'no floor'}" + ("" if set(need_ext) <= set(ext) else
                                                                   f" (NOT YET RUN at {sorted(set(need_ext) - set(ext))})"))
    # H2 and the reported equal-compute comparisons
    out.append("  Equal compute: d = g(D1, H) − g(Δ_eq, ∞), Δ_eq the full-horizon interval of equal rollout volume "
               "(read on the stored curve, linear in log D1) [90%]")
    lg = np.log(np.array(GRID, float))
    H2 = {}
    for d, H in [(60, H) for H in hs] + [(300, H) for H in hs]:
        de = d_eq(d, H)
        cells = []
        for f in FLOORS:
            if (d, H) not in D[f]["h"]:
                cells.append(f"{f} –")
                continue
            if de > GRID[-1]:
                cells.append(f"{f} n/a")
                continue
            fullg = gain(D[f]["full"], D[f]["once"])
            dp = float(gain(D[f]["h"][(d, H)], D[f]["once"]) - np.interp(np.log(de), lg, fullg))
            db = np.array([gain(D[f]["h"][(d, H)], D[f]["once"], idx)
                           - np.interp(np.log(de), lg, gain(D[f]["full"], D[f]["once"], idx)) for idx in idxs])
            lo, hi = q90(db)
            if (d, H) == (60, H2_H):
                H2[f] = (dp, lo, hi)
            cells.append(f"{f} {dp:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        out.append(f"    D1 {d:4d} H {H:5d}: Δ_eq {de:6.0f} s  " + "  ".join(cells))
    above = [f for f, (_, lo, _) in H2.items() if lo > 0]
    below = [f for f, (_, _, hi) in H2.items() if hi < 0]
    h2 = "PASS" if len(above) >= 2 else ("FAIL" if len(below) >= 2 else "indeterminate")
    out.append(f"  H2 (H {H2_H:,} s, Δ_eq {d_eq(60, H2_H):.0f} s): above 0 at {above or 'none'}, below 0 at "
               f"{below or 'none'} -> {h2}")
    # reported: wall time against volume, switches
    out.append("  Wall time (local, serial) against rollout volume, and switches per window (stored full horizon at the "
               "same D1 beside):")
    for f in FLOORS:
        for d in R7.D1_7:
            fs = np.mean([x["switches"] for x in D[f]["full_rec"][d]])
            cells = []
            for H in hs:
                if (d, H) not in D[f]["recs"]:
                    continue
                recs = D[f]["recs"][(d, H)]
                ps = np.mean([x["plan_seconds"] for x in recs])
                cw = R7.rollout_cw(d, H)
                cells.append(f"H {H}: {ps:7.0f} s, {cw:6.1f} c_W, {ps / cw:5.1f} s/c_W, "
                             f"{np.mean([x['switches'] for x in recs]):4.1f} switches")
            out.append(f"    {f:5s} D1 {d:4d} (full: {fs:4.1f} switches)  " + "; ".join(cells))
    from collections import Counter
    recs_all = [x for f in FLOORS for recs in D[f]["recs"].values() for x in recs]
    hosts = Counter(x["stamp"]["host"].split(".")[0] for x in recs_all)
    md5s = Counter(x["stamp"].get("run7_py_md5") for x in recs_all)
    unknown = sorted(set(md5s) - set(RUN7_OK))
    out.append(f"  stamps: run7.py {dict(md5s)}; run.py {dict(Counter(x['stamp']['run_py_md5'] for x in recs_all))}; "
               f"{len(hosts)} host(s), most tasks on {hosts.most_common(1)[0][0]}")
    out.append("  stamps " + (f"OK: every run7.py is one of {RUN7_OK} (the same planner; 11-13 changed only the "
                              "real-time guard)" if not unknown else f"UNKNOWN run7.py version(s) {unknown}: check them"))
    return out, {"complete": True, "c7": c7, "h_min": h_min, "h2": h2, "need_ext": need_ext, "ext": ext}


def main(argv):
    args = list(argv)
    root = HERE
    if "--root" in args:
        i = args.index("--root")
        root = Path(args[i + 1]).resolve()
        del args[i:i + 2]
    if args and args[0] == "extension":
        print(" ".join(extension(root)))
        return
    lines, _ = frontier()
    if args and args[0] == "frontier":
        print("\n".join(lines))
        return
    more, _ = horizon(root)
    text = "\n".join(lines + more) + "\n"
    (Path(root) / "analysis7.out").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main(sys.argv[1:])
