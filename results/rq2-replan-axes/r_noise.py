"""
@file r_noise.py
@brief rq2-replan-axes, remark in README §12 (not a reading): how much of reading R's failure at 40 seeds could come from
       its reference, the best of the 9 grid points, being biased upward by seed noise. Three references beside the
       analyzer's: the best point minus its seed-bootstrap bias, a cross-fit (best point chosen on 20 seeds, measured on
       the other 20), and the first point of a fit that does not rise with Δ. Also the bootstrap share of resamples in
       which each level (and each arm) passes R, with B2's edge fixed at D1 300 s and recomputed per resample.
@par Usage
@code{.sh}
.venv/bin/python results/rq2-replan-axes/r_noise.py > results/rq2-replan-axes/r_noise.out
@endcode
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402

B, SPLITS, D_REF = 4000, 2000, 300.0
data, base = A.load(HERE)
LV = [("proc", k) for k in A.LEVELS["proc"]] + [("reg", k) for k in A.LEVELS["reg"] if k != 1.0]
rng = np.random.default_rng(7)


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


ds0, seeds0, real0, ref0 = arrays(("proc", 1.0))
assert all(arrays(k)[1] == seeds0 for k in LV)
n = len(seeds0)
idx = rng.integers(0, n, (B, n))
splits = [rng.permutation(n) for _ in range(SPLITS)]
edges = np.array([A.dstar(ds0, list(g_of(real0, ref0, r)), A.X_PRIMARY)["right_grid"] for r in idx])
print(f"seeds {n}, {B} paired seed resamples, {SPLITS} cross-fit splits (both ways); X {A.X_PRIMARY:.0%}")
print("B2's grid edge over the resamples: " + ", ".join(f"D1 {int(e)} s {np.mean(edges == e):.0%}"
                                                       for e in sorted(set(edges)) if np.mean(edges == e) > 0))
print()
print("level       best         g(300)  keep | SE one point, SE g(60)-g(300) | boot bias  keep | cross-fit ref  keep | "
      "non-increasing ref  keep | shortfall | P(keep>=85%): edge 300, B2 edge per resample | keep on the 8-point grid")
res = {}
for key in LV:
    ds, seeds, real, ref = arrays(key)
    g = g_of(real, ref, np.arange(n))
    i300, gmax = ds.index(D_REF), g.max()
    boots = np.array([g_of(real, ref, r) for r in idx])
    bias = boots.max(1).mean() - gmax
    refs = []
    for p in splits:
        a, b = p[: n // 2], p[n // 2:]
        for x, y in ((a, b), (b, a)):
            refs.append(g_of(real, ref, y)[int(np.argmax(g_of(real, ref, x)))])
    cf, iso = np.mean(refs), pava_dec(g)[0]
    ok300 = boots[:, i300] / boots.max(1) >= 1 - A.X_PRIMARY
    oke = np.array([boots[b, ds.index(e)] / boots[b].max() for b, e in enumerate(edges)]) >= 1 - A.X_PRIMARY
    no60 = g[i300] / g[1:].max() if ds[0] == 60 else float("nan")
    short = gmax - g[i300] / (1 - A.X_PRIMARY)
    res[key] = dict(naive=g[i300] / gmax, bc=g[i300] / (gmax - bias), cf=g[i300] / cf, iso=g[i300] / iso,
                    ok300=ok300, oke=oke)
    print(f"{key[0]:4s} k {key[1]:<4g} {gmax:+.2f} @ {int(ds[int(np.argmax(g))]):3d}  {g[i300]:+.2f}  {g[i300] / gmax:4.0%} | "
          f"{boots.std(0).mean():.2f}, {np.std(boots[:, 0] - boots[:, i300]):.2f} | {bias:+.2f}  "
          f"{g[i300] / (gmax - bias):4.0%} | {cf:+.2f}  {g[i300] / cf:4.0%} | {iso:+.2f}  {g[i300] / iso:4.0%} | "
          f"{max(short, 0):.2f} | {ok300.mean():4.0%}, {oke.mean():4.0%} | {no60:4.0%}")
print()
for arm in ("proc", "reg"):
    ks = [x for x in LV if x[0] == arm or x == ("proc", 1.0)]
    print(f"{arm}: R passes in {np.all([res[k]['ok300'] for k in ks], 0).mean():.0%} of resamples with B2's edge at "
          f"D1 300 s, {np.all([res[k]['oke'] for k in ks], 0).mean():.0%} with B2's edge recomputed per resample")
    for lab, f in (("best grid point (the analyzer)", "naive"), ("best point minus bootstrap bias", "bc"),
                   ("cross-fit", "cf"), ("non-increasing fit", "iso")):
        fails = [f"k {k:g}" for a, k in ks if res[(a, k)][f] < 1 - A.X_PRIMARY]
        print(f"   {lab}: worst keep {min(res[k][f] for k in ks):.1%}, below 85% at {', '.join(fails) or '-'}")
