"""
@file axis6_checks.py
@brief rq2-replan-axes, remarks beside README §13's axis 6 readings (not readings; the decision rule stands on
       analysis.out). Four checks:
       (1) the planner at D1 900 s against the 10-29 screen (the switching oracle at 900 s slots, seeds 0-19);
       (2) seed noise in R and R120's reference, as r_noise.py does for axis 1: the best point minus its bootstrap
           bias, a cross-fit (best point chosen on 20 seeds, measured on the other 20), a non-increasing fit, and the
           share of resamples in which each level keeps >= 85%;
       (3) S_m with one size level left out at a time (which levels carry the slope);
       (4) post hoc: Δ* / p̄ against the on-duty AGV share over the size and fleet levels, and decisions per Δ*.
@par Usage
@code{.sh}
.venv/bin/python results/rq2-replan-axes/axis6_checks.py > results/rq2-replan-axes/axis6_checks.out
@endcode
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402

B, SPLITS = 4000, 2000
data, base = A.load(HERE)
SIZE = [("size", k) for k in A.LEVELS["size"]]
FLEET = [("fleet", k) for k in A.LEVELS["fleet"]]
LV = SIZE + [k for k in FLEET if k != ("fleet", 0.52)]          # fleet r 0.52 is size m50
SCREEN = {("size", 15.0): 5.3, ("size", 50.0): 5.6, ("size", 100.0): 2.5, ("fleet", 0.35): 6.0, ("fleet", 0.8): 6.2}
rng = np.random.default_rng(11)


def arrays(key):
    by_d = data[key]
    ds = sorted(by_d)
    seeds = sorted(set.intersection(*(set(by_d[d]) for d in ds)) & set(base[key]))
    real = np.array([[by_d[d][s]["real"] for s in seeds] for d in ds])
    ref = np.array([base[key][s]["best"] for s in seeds])
    return ds, seeds, real, ref


def g_of(real, ref, cols):
    return -100.0 * (real[:, cols].sum(1) / ref[cols].sum() - 1.0)


def pava_dec(y):
    """Least-squares fit that does not rise with Δ (pool adjacent violators, equal weights)."""
    out = []
    for v in y:
        out.append([v, 1])
        while len(out) > 1 and out[-2][0] < out[-1][0]:
            v2, w2 = out.pop()
            v1, w1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    return np.concatenate([[v] * w for v, w in out])


seeds0 = arrays(SIZE[0])[1]
assert all(arrays(k)[1] == seeds0 for k in LV)
n = len(seeds0)
idx = rng.integers(0, n, (B, n))
splits = [rng.permutation(n) for _ in range(SPLITS)]

print("(1) planner at D1 900 s vs the 10-29 screen (switching oracle at 900 s slots), seeds 0-19, gain over plan-once")
for key in LV:
    ds, seeds, real, ref = arrays(key)
    g20 = g_of(real, ref, np.arange(20))[ds.index(900.0)]
    print(f"   {A.label(*key):14s} planner {g20:+.2f}   screen {SCREEN.get(key, float('nan')):+.1f}   "
          f"(40 seeds {g_of(real, ref, np.arange(n))[ds.index(900.0)]:+.2f})")

print()
print(f"(2) seed noise in the reference of R (D1 300 s) and R120 (D1 120 s): {B} paired resamples, {SPLITS} cross-fit "
      f"splits, X {A.X_PRIMARY:.0%}")
print("level           best          kept 300: grid  boot-bias  cross-fit  non-incr  P>=85% | kept 120: grid  "
      "boot-bias  cross-fit  non-incr  P>=85% | boot bias, SE g(60)-g(120)")
keep = {}
for key in LV:
    ds, seeds, real, ref = arrays(key)
    g = g_of(real, ref, np.arange(n))
    boots = np.array([g_of(real, ref, r) for r in idx])
    bias = boots.max(1).mean() - g.max()
    cf = []
    for p in splits:
        a, b = p[: n // 2], p[n // 2:]
        for x, y in ((a, b), (b, a)):
            cf.append(g_of(real, ref, y)[int(np.argmax(g_of(real, ref, x)))])
    cf, iso = np.mean(cf), pava_dec(g)[0]
    row = f"{A.label(*key):14s} {g.max():+.2f} @ {int(ds[int(np.argmax(g))]):3d} "
    for d in (300.0, 120.0):
        i = ds.index(d)
        p = np.mean(boots[:, i] / boots.max(1) >= 1 - A.X_PRIMARY)
        keep[(key, d)] = dict(grid=g[i] / g.max(), bc=g[i] / (g.max() - bias), cf=g[i] / cf, iso=g[i] / iso,
                              ok=boots[:, i] / boots.max(1) >= 1 - A.X_PRIMARY)
        k = keep[(key, d)]
        row += (f"|      {k['grid']:5.1%}   {k['bc']:6.1%}    {k['cf']:6.1%}   {k['iso']:6.1%}   {p:5.0%} ")
    row += f"| {bias:+.2f}, {np.std(boots[:, 0] - boots[:, ds.index(120.0)]):.2f}"
    print(row)
for name, keys in (("size", SIZE), ("fleet", FLEET)):
    keys = [("size", 50.0) if k == ("fleet", 0.52) else k for k in keys]
    for d, what in ((300.0, "R"), (120.0, "R120")):
        both = np.all([keep[(k, d)]["ok"] for k in keys], 0).mean()
        worst = {f: min(keep[(k, d)][f] for k in keys) for f in ("grid", "bc", "cf", "iso")}
        print(f"   {name} {what}: passes in {both:.0%} of resamples; worst kept share: grid {worst['grid']:.1%}, "
              f"bias-corrected {worst['bc']:.1%}, cross-fit {worst['cf']:.1%}, non-increasing {worst['iso']:.1%}")

print()
print("(3) S_m with one size level left out (point slope, 90% joint seed bootstrap)")
res = {}
for key in SIZE:
    ds, seeds, real, ref = arrays(key)
    pb = A.describe(data[key], seeds)["p_mean_plan"]
    d_pt = A.dstar(ds, list(g_of(real, ref, np.arange(n))), A.X_PRIMARY)["dstar"]
    d_bs = [A.dstar(ds, list(g_of(real, ref, r)), A.X_PRIMARY)["dstar"] for r in idx[:A.BOOT]]
    res[key[1]] = (pb, d_pt, d_bs)


def slope(ks, vals):
    return float(np.polyfit(np.log2(ks), np.log2(vals), 1)[0])


ms = [k for _, k in SIZE]
for drop in [None] + ms:
    ks = [m for m in ms if m != drop]
    pt = slope(ks, [res[m][1] / res[m][0] for m in ks])
    bs = [slope(ks, [res[m][2][b] / res[m][0] for m in ks]) for b in range(A.BOOT)
          if all(res[m][2][b] is not None for m in ks)]
    lo, hi = np.percentile(bs, [5, 95])
    print(f"   {'all levels' if drop is None else f'without m{drop:g}':14s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
          f"{A.size_verdict(lo, hi).split(' (')[0]}")

print()
print("(4) post hoc: Δ* / p̄, on-duty AGV share and decisions per Δ* (size and fleet levels)")
print("   level           Δ*/p̄   upper bound   AGV busy on duty   machine busy   decisions per window   per Δ*")
pts = []
for key in SIZE + FLEET:
    ds, seeds, real, ref = arrays(key)
    desc = A.describe(data[key], seeds)
    d = A.dstar(ds, list(g_of(real, ref, np.arange(n))), A.X_PRIMARY)
    u = desc.get("agv_busy_on_duty")
    u = desc["agv_busy_share"] if u is None or np.isnan(u) else u        # B2 has no off-duty AGVs
    dec = np.mean([data[key][60.0][s]["decisions"] for s in seeds])
    win = data[key][60.0][seeds[0]]["window"]
    mb = np.mean([data[key][ds[0]][s]["describe"].get("machine_busy_share", np.nan) for s in seeds])
    pts.append((key, d["dstar"] / desc["p_mean_plan"], u))
    print(f"   {A.label(*key):14s} {d['dstar'] / desc['p_mean_plan']:5.2f}   {str(d['best_d'] == ds[0]):11s}   {u:5.2f}"
          f"              {mb:5.2f}          {dec:7.0f}                {dec * d['dstar'] / win:5.0f}")
lv = [p for p in pts if p[0] != ("fleet", 0.52)]
r = np.corrcoef(np.log2([p[2] for p in lv]), np.log2([p[1] for p in lv]))[0, 1]
print(f"   over the 7 distinct levels: corr(log2 AGV busy on duty, log2 Δ*/p̄) = {r:+.2f}; slope "
      f"{slope([p[2] for p in lv], [p[1] for p in lv]):+.2f} (post hoc, no CI: the levels share seeds)")
