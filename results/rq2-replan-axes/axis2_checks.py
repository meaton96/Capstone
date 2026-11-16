"""
@file axis2_checks.py
@brief rq2-replan-axes, remarks beside README §15's axis 2 readings (not readings; the decision rule stands on
       analysis.out). Three checks:
       (1) seed noise in R_ℓ / R_c / R_mix's reference, as axis6_checks.py does for axis 6: the centre's grid edge
           against the best point, the best point minus its bootstrap bias, a cross-fit (best point chosen on 20 seeds,
           measured on the other 20) and a non-increasing fit, with the share of resamples in which each cell and each
           line keeps >= 85%; the kept share at 180 s at 5,400 s blocks is shown beside it (descriptive);
       (2) the low-tardiness corner (ℓ 0.7, c 3.0): how the plan-once tardiness spreads over seeds, and the reported
           slopes on log2 u and log2 c with one 3 x 3 cell left out at a time (which cells carry them);
       (3) the share of the best gain D1 60 s keeps at every distinct condition of axes 1, 2 and 6.
@par Usage
@code{.sh}
.venv/bin/python results/rq2-replan-axes/axis2_checks.py > results/rq2-replan-axes/axis2_checks.out
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
CELLS = [("ax2", k) for k in A.LEVELS["ax2"]]
EDGE = {5400: 300.0, 1350: 180.0}                      # the centres' grid edges (B2; reg k 0.25, §12)
rng = np.random.default_rng(12)


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


def name(key):
    return A.ax2_name(key[1])


seeds0 = arrays(CELLS[0])[1]
assert all(arrays(k)[1] == seeds0 for k in CELLS), "axis 2 cells do not share their seeds"
n = len(seeds0)
idx = rng.integers(0, n, (B, n))
splits = [rng.permutation(n) for _ in range(SPLITS)]

print(f"(1) seed noise in the reference of R_ℓ / R_c / R_mix (the centre's grid edge: 300 s at 5,400 s blocks, 180 s at "
      f"1,350 s): {B} paired resamples, {SPLITS} cross-fit splits, X {A.X_PRIMARY:.0%}")
print("cell                        best         kept at the edge: grid  boot-bias  cross-fit  non-incr  P>=85%"
      "   (5,400 s blocks: kept at 180 s)")
keep = {}
for key in CELLS:
    b = key[1][2]
    ds, seeds, real, ref = arrays(key)
    g = g_of(real, ref, np.arange(n))
    boots = np.array([g_of(real, ref, r) for r in idx])
    bias = boots.max(1).mean() - g.max()
    cf = []
    for p in splits:
        a, c = p[: n // 2], p[n // 2:]
        for x, y in ((a, c), (c, a)):
            cf.append(g_of(real, ref, y)[int(np.argmax(g_of(real, ref, x)))])
    cf, iso = np.mean(cf), pava_dec(g)[0]
    i = ds.index(EDGE[b])
    ok = boots[:, i] / boots.max(1) >= 1 - A.X_PRIMARY
    keep[key] = dict(grid=g[i] / g.max(), bc=g[i] / (g.max() - bias), cf=g[i] / cf, iso=g[i] / iso, ok=ok)
    k = keep[key]
    extra = f"   {g[ds.index(180.0)] / g.max():6.1%}" if b == 5400 else ""
    print(f"{name(key):26s} {g.max():+6.2f} @ {int(ds[int(np.argmax(g))]):3d}   {k['grid']:6.1%}   {k['bc']:6.1%}"
          f"     {k['cf']:6.1%}   {k['iso']:6.1%}   {ok.mean():5.0%}{extra}")
for b in (5400, 1350):
    for line, cells in A.AX2_LINES.items():
        keys = [("ax2", (ell, c, b, mix)) for ell, c, mix in cells]
        both = np.all([keep[k]["ok"] for k in keys], 0).mean()
        worst = {f: min(keep[k][f] for k in keys) for f in ("grid", "bc", "cf", "iso")}
        print(f"   block {b} {line}: passes in {both:.0%} of resamples; worst kept share: grid {worst['grid']:.1%}, "
              f"bias-corrected {worst['bc']:.1%}, cross-fit {worst['cf']:.1%}, non-increasing {worst['iso']:.1%}")

print()
print("(2) the low-tardiness corner (ℓ 0.7, c 3.0) and the reported slopes with one 3 x 3 cell left out")
for b in (5400, 1350):
    for key in [("ax2", (0.7, 3.0, b, "b2")), ("ax2", (1.0, 2.125, b, "b2"))]:
        _, seeds, _, ref = arrays(key)
        top = np.sort(ref)[::-1]
        print(f"   {name(key):18s} plan-once tardiness {ref.sum():7.0f} (1000 job-s over {len(seeds)} seeds); seeds at 0: "
              f"{int((ref == 0).sum())}; the top 5 seeds carry {top[:5].sum() / ref.sum():.0%}, the top 10 "
              f"{top[:10].sum() / ref.sum():.0%}")
res = {}
for b in (5400, 1350):
    ks = [k for k in A.LEVELS["ax2"] if k[2] == b and k[3] == "b2"]
    for k in ks:
        key = ("ax2", k)
        ds, seeds, real, ref = arrays(key)
        d_pt = A.dstar(ds, list(g_of(real, ref, np.arange(n))), A.X_PRIMARY)
        d_bs = [A.dstar(ds, list(g_of(real, ref, r)), A.X_PRIMARY)["dstar"] for r in idx[:A.BOOT]]
        pb = np.mean([base[key][s]["describe"]["p_mean_plan"] for s in seeds])   # as analyze.py (fixed-pair task)
        u = np.mean([base[key][s]["describe"]["machine_busy_share"] for s in seeds])
        res[k] = (pb, u, d_pt["dstar"], d_bs, d_pt["best_d"] == ds[0])


def fit(ks, vals):
    x = np.column_stack([np.ones(len(ks)), np.log2([res[k][1] for k in ks]), np.log2([k[1] for k in ks])])
    return np.linalg.lstsq(x, np.log2(vals), rcond=None)[0][1:]


for b in (5400, 1350):
    ks = [k for k in A.LEVELS["ax2"] if k[2] == b and k[3] == "b2"]
    print(f"   block {b}: Δ*/p̄ per cell (u = machine busy share; * = upper bound, rule 9): "
          + ", ".join(f"ℓ{k[0]:g} c{k[1]:g} {res[k][2] / res[k][0]:.2f}{'*' if res[k][4] else ''} (u {res[k][1]:.2f})"
                      for k in ks))
    for drop in [None] + ks:
        kk = [k for k in ks if k != drop]
        pt = fit(kk, [res[k][2] / res[k][0] for k in kk])
        bs = np.array([fit(kk, [res[k][3][i] / res[k][0] for k in kk]) for i in range(A.BOOT)
                       if all(res[k][3][i] is not None for k in kk)])
        lo, hi = np.percentile(bs, 5, axis=0), np.percentile(bs, 95, axis=0)
        what = "all 9 cells" if drop is None else f"without ℓ{drop[0]:g} c{drop[1]:g}"
        print(f"      {what:22s} u {pt[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}]   c {pt[1]:+.2f} [{lo[1]:+.2f}, {hi[1]:+.2f}]")

print()
print("(3) D1 60 s against each condition's best gain, every distinct condition of axes 1, 2 and 6 (aliases counted once)")
seen, rows = set(), []
for key in sorted((k for k in data if k[0] != "all"), key=lambda k: (A.ARMS.index(k[0]), str(k[1]))):
    if id(data[key]) in seen or key not in base or 60.0 not in data[key]:
        continue
    seen.add(id(data[key]))
    ds, seeds, real, ref = arrays(key)
    g = g_of(real, ref, np.arange(len(seeds)))
    rows.append((key, g[ds.index(60.0)] / g.max(), g.max(), ds[int(np.argmax(g))]))
for arm in ("proc", "reg", "size", "fleet", "ax2"):
    rs = [r for r in rows if r[0][0] == arm]
    if rs:
        lo = min(rs, key=lambda r: r[1])
        print(f"   {arm:5s} {len(rs):2d} conditions: D1 60 s keeps {min(r[1] for r in rs):.1%}-{max(r[1] for r in rs):.1%}"
              f" (lowest at {A.label(*lo[0])}, best {lo[2]:+.2f} at {lo[3]:g} s); best at 60 s at "
              f"{sum(r[3] == 60.0 for r in rs)}")
lo = min(rows, key=lambda r: r[1])
print(f"   all {len(rows)} conditions: D1 60 s keeps >= {lo[1]:.1%} (lowest at {A.label(*lo[0])}); best at 60 s at "
      f"{sum(r[3] == 60.0 for r in rows)}; best gain {min(r[2] for r in rows):+.2f} to {max(r[2] for r in rows):+.2f} points")
