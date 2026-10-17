"""
@file analyze.py
@brief eval-v5-cmix-unity (2026-10-15): do the due-date-trained PPO policies keep their twin gains in Unity? Unity next
       to the twin (../eval-v5-cmix, ../rq3-gen-v5) on the same B2 seeds 0-39 (0-19 the first pass, 20-39 added
       10-16), at due-loose (c 2.5-3.5) and base (c 1.75-2.5).

@details Picks (made on the fitting seeds 2000-2159 in ../eval-v5-cmix): loose s2 (train-v5-loose, c 2.5-3.5), cmix s3
(train-v5-cmix, c 1.25-3.5), base s4 (train-v5-final, c 1.75-2.5; the final policy). Sources:
  Unity due-loose  runs/due-loose/{pa, pb, pc}
  Unity base       runs/base/pc (the loose and cmix picks + MDD-PTECT) and, reused, the 6 v5 pairs and s4 from
                   ../eval-transfer-unity-v5 and the no-breakdown rows of ../rq4-agvfail-v5-rules / -policy
  twin             ../rq3-gen-v5/<cond>/{pdr, ppo} and ../eval-v5-cmix/test/<cond>/{loose, cmix}
Checks first: pc's MDD-PTECT against the reused base rows and against pa at due-loose, seed for seed, and the Unity
instance hashes across groups. Then per condition, on the seeds complete in both simulators: the 6 v5 pairs' ranking
(Kendall tau Unity vs twin, best pair in each), MDD-PTECT Unity vs twin, each pick vs MDD-PTECT and vs the
simulator's best v5 pair (pooled window tardiness, seed-bootstrap 95% CI, seeds better / worse, Wilcoxon, k job-s
saved per instance), the hindsight best v5 pair, and per seed how far each pick's effect agrees between
the simulators. -> analysis.out, summary.csv (seeds 0-39); `--seeds A-B` analyzes that range only ->
analysis_sA-B.out, summary_sA-B.csv (`--seeds 0-19` reproduces the first pass of 10-16 01:26).
@par Usage
@code{.sh}
.venv/bin/python results/eval-v5-cmix-unity/analyze.py [--seeds 0-19]
@endcode
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon

HERE = Path(__file__).resolve().parent
R = HERE.parent
HASH_STR = {"config_hash": str, "instance_hash": str}
V5 = ["SRT-PECT", "SRT-PTECT", "MDD-PECT", "MDD-PTECT", "ATC-PECT", "ATC-PTECT"]
REF = "MDD-PTECT"
SEEDS = list(range(40))   # replaced by --seeds
PICKS = {"train-v5-loose_s2": "loose s2", "train-v5-cmix_s3": "cmix s3", "train-v5-final_s4": "base s4"}
CONDS = ["due-loose", "base"]


def name(p):
    p = str(p).replace("ckpt:", "").replace("/checkpoint", "").replace("results/", "")
    return PICKS.get(p, p)


def read(files, src):
    """Long table (policy, seed, window_tardiness in k job-s, instance_hash, src) of the finished tasks in files."""
    fr = []
    for f in files:
        f = Path(f)
        if f.exists() and (f.parent / "DONE").exists():
            e = pd.read_csv(f, dtype=HASH_STR)
            fr.append(e.assign(src=src, policy=e.policy.map(name)))
    if not fr:
        return pd.DataFrame(columns=["policy", "seed", "window_tardiness", "instance_hash", "src"])
    e = pd.concat(fr, ignore_index=True)
    e = e[e.seed.isin(SEEDS)]
    return e.assign(window_tardiness=e.window_tardiness / 1000.0)[
        ["policy", "seed", "window_tardiness", "instance_hash", "src"]]


def cat(frames):
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else read([], "")


def sources(cond):
    """(Unity long table, twin long table) for cond."""
    if cond == "base":
        u = cat([read((HERE / "runs" / "base" / "pc").glob("s*/episodes.csv"), "pc"),
                 read((R / "eval-transfer-unity-v5" / "runs").glob("*/s*/episodes.csv"), "reused"),
                 read((R / "rq4-agvfail-v5-rules" / "runs" / "none").glob("*/s*/episodes.csv"), "reused"),
                 read((R / "rq4-agvfail-v5-policy" / "runs" / "none").glob("*/s*/episodes.csv"), "reused")])
    else:
        u = cat([read((HERE / "runs" / cond / g).glob("s*/episodes.csv"), g) for g in ("pa", "pb", "pc")])
    t = cat([read([R / "rq3-gen-v5" / cond / g / "episodes.csv"], "rq3-gen-v5") for g in ("pdr", "ppo")]
            + [read([R / "eval-v5-cmix" / "test" / cond / g / "episodes.csv"], "eval-v5-cmix") for g in ("loose", "cmix")])
    return u, t


def same(a, b, what, hashes=True):
    """Seed-for-seed equality of one policy's tardiness (and the Unity instance hash) between two sources."""
    a, b = a.set_index("seed"), b.set_index("seed")
    s = sorted(set(a.index) & set(b.index))
    if not s:
        return f"   {what}: no common seeds yet"
    eq = sum(a.loc[k, "window_tardiness"] == b.loc[k, "window_tardiness"] for k in s)
    ih = sum(a.loc[k, "instance_hash"] == b.loc[k, "instance_hash"] for k in s)
    bad = [k for k in s if a.loc[k, "window_tardiness"] != b.loc[k, "window_tardiness"]]
    return (f"   {what}: tardiness identical on {eq} / {len(s)} seeds"
            + (f", instance hash on {ih} / {len(s)}" if hashes else "")
            + (f"  ** MISMATCH on seeds {bad}" if bad else ""))


def checks(cond, u, t):
    m = u[u.policy == REF]
    if cond == "base":
        new, old = m[m.src == "pc"], m[m.src == "reused"].drop_duplicates("seed")
        print(same(new, old, "Unity base: pc's MDD-PTECT vs the reused rows (eval-transfer-unity-v5 / rq4 none)"))
    else:
        print(same(m[m.src == "pc"], m[m.src == "pa"], "Unity due-loose: pc's MDD-PTECT vs pa's (run to run)"))
        h = u.groupby("seed").instance_hash.nunique()
        print(f"   Unity due-loose: one instance hash per seed across pa / pb / pc on {int((h == 1).sum())} / {len(h)} "
              "seeds")
    tm = t[t.policy == REF]
    print(same(tm[tm.src == "eval-v5-cmix"].drop_duplicates("seed"), tm[tm.src == "rq3-gen-v5"].drop_duplicates("seed"),
               f"twin {cond}: MDD-PTECT in the loose / cmix groups vs rq3-gen-v5", hashes=False))


def wide(e):
    return e.drop_duplicates(["policy", "seed"]).pivot_table(index="seed", columns="policy", values="window_tardiness")


def stats(y, b, rng):
    ok = y.notna() & b.notna()
    yy, bb = y[ok].to_numpy(), b[ok].to_numpy()
    if len(yy) < 5:
        return None
    idx = rng.integers(0, len(yy), (10000, len(yy)))
    bs = 100 * (yy[idx].sum(1) / bb[idx].sum(1) - 1)
    d = yy - bb
    return {"n": len(yy), "pooled_%": 100 * (yy.sum() / bb.sum() - 1), "ci_lo": np.percentile(bs, 2.5),
            "ci_hi": np.percentile(bs, 97.5), "better": int((d < 0).sum()), "worse": int((d > 0).sum()),
            "p": float(wilcoxon(d).pvalue) if np.any(d) else 1.0, "saved_k_per_inst": float(-d.mean())}


def gap(yu, bu, yt, bt, rng):
    """Unity minus twin of the pooled % (y vs b), seeds resampled jointly: does the gain transfer?"""
    yu, bu, yt, bt = (x.to_numpy() for x in (yu, bu, yt, bt))
    idx = rng.integers(0, len(yu), (10000, len(yu)))
    pc = lambda y, b: 100 * (y[idx].sum(1) / b[idx].sum(1) - 1)   # noqa: E731
    d = pc(yu, bu) - pc(yt, bt)
    pt = 100 * (yu.sum() / bu.sum() - yt.sum() / bt.sum())
    return {"n": len(yu), "pooled_%": pt, "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5)}


def fmt(s):
    if s is None:
        return "n<5"
    return (f"{s['pooled_%']:+6.2f} [{s['ci_lo']:+6.2f}, {s['ci_hi']:+6.2f}] {s['better']:2d}/{s['worse']:<2d} "
            f"p={s['p']:<7.2g} saved {s['saved_k_per_inst']:+5.2f} k")


def main(tag=""):
    rng = np.random.default_rng(0)
    print(f"== eval-v5-cmix-unity: Unity vs twin on B2 seeds {SEEDS[0]}-{SEEDS[-1]}; picks loose s2 (trained c 2.5-3.5), "
          "cmix s3 (c 1.25-3.5), base s4 (c 1.75-2.5); pooled window tardiness, < 0 better; saved = k job-s per instance")
    rows = []
    for cond in CONDS:
        u_long, t_long = sources(cond)
        print(f"\n-- {cond}: checks")
        checks(cond, u_long, t_long)
        u, t = wide(u_long), wide(t_long)
        need = V5 + list(PICKS.values())
        miss = [p for p in need if p not in u or p not in t]
        if miss:
            print(f"   incomplete: missing in Unity or twin: {miss}")
            continue
        s = [k for k in SEEDS if k in u.index and k in t.index and u.loc[k, need].notna().all()
             and t.loc[k, need].notna().all()]
        print(f"-- {cond}: {len(s)} seeds complete in both simulators" + ("" if len(s) == len(SEEDS) else
                                                                         f" (missing {sorted(set(SEEDS) - set(s))})"))
        if len(s) < 5:
            continue
        u, t = u.loc[s], t.loc[s]
        tu, tt = u[V5].sum(), t[V5].sum()
        best = {"Unity": tu.idxmin(), "twin": tt.idxmin()}
        k = kendalltau(tu[V5], tt[V5])
        print(f"   best v5 pair: Unity {best['Unity']}, twin {best['twin']}; pair ranking Kendall tau "
              f"{k.statistic:+.2f} (p {k.pvalue:.2g})")
        print(f"   MDD-PTECT mean: Unity {u[REF].mean():.1f} k, twin {t[REF].mean():.1f} k; Unity vs twin "
              f"{fmt(stats(u[REF], t[REF], rng))}")
        print("   pairs vs MDD-PTECT (Unity | twin): " + "  ".join(
            f"{p} {100 * (tu[p] / tu[REF] - 1):+.1f}|{100 * (tt[p] / tt[REF] - 1):+.1f}" for p in V5 if p != REF))
        for sim, w in (("Unity", u), ("twin", t)):
            w = w.assign(**{"hindsight best v5 pair": w[V5].min(axis=1)})
            for m in list(PICKS.values()) + ["hindsight best v5 pair"]:
                for vs, b in ((REF, w[REF]), (f"best pair ({best[sim]})", w[best[sim]])):
                    st = stats(w[m], b, rng)
                    if st:
                        rows.append({"cond": cond, "sim": sim, "method": m, "vs": vs.split(" (")[0], "best": best[sim],
                                     **st})
        hu, ht = u[V5].min(axis=1), t[V5].min(axis=1)
        for m in list(PICKS.values()) + ["hindsight best v5 pair"]:
            for vs in (REF, "best pair"):
                r = {x["sim"]: x for x in rows if x["cond"] == cond and x["method"] == m and x["vs"] == vs}
                g = gap(hu if m.startswith("hindsight") else u[m], u[REF if vs == REF else best["Unity"]],
                        ht if m.startswith("hindsight") else t[m], t[REF if vs == REF else best["twin"]], rng)
                rows.append({"cond": cond, "sim": "Unity - twin", "method": m, "vs": vs, **g})
                print(f"   {m:22s} vs {vs:9s}  Unity {fmt(r.get('Unity'))} | twin {fmt(r.get('twin'))} | "
                      f"Unity - twin {g['pooled_%']:+5.2f} pp [{g['ci_lo']:+5.2f}, {g['ci_hi']:+5.2f}]")
        print("   per seed, pick - MDD-PTECT (k job-s): Unity vs twin correlation; the 3 seeds with the largest "
              "|Unity - twin| and the pooled % without them")
        for m in PICKS.values():
            du, dt = u[m] - u[REF], t[m] - t[REF]
            g = du - dt
            top = g.abs().nlargest(3).index.tolist()
            keep = [k for k in s if k not in top]
            print(f"     {m}: r {np.corrcoef(du, dt)[0, 1]:+.2f}; Unity - twin {g.sum():+.1f} k in all, seeds {top} "
                  f"{g[top].sum():+.1f} k; without them Unity {100 * (u.loc[keep, m].sum() / u.loc[keep, REF].sum() - 1):+.2f}%"
                  f", twin {100 * (t.loc[keep, m].sum() / t.loc[keep, REF].sum() - 1):+.2f}%")
    if rows:
        pd.DataFrame(rows).to_csv(HERE / f"summary{tag}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-39", help="seed range A-B (default 0-39 -> analysis.out, summary.csv)")
    a, b = map(int, ap.parse_args().seeds.split("-"))
    tag = "" if (a, b) == (0, 39) else f"_s{a}-{b}"
    SEEDS = list(range(a, b + 1))
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(tag)
    (HERE / f"analysis{tag}.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
