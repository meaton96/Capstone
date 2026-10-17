"""
@file analyze.py
@brief rq3-flex-v5-unity (2026-10-17): machine flexibility in Unity, B2 seeds 0-19. Per level (flex-p0.3 / flex-p0.5,
       secondary types 1.25x slower) and for no flexibility (B2 in Unity: ../rq4-agvfail-v5-rules' level-none rows
       and ../eval-v5-cmix-unity base): the best v5 pair, the pair ranking against no flexibility (Kendall tau), and the
       final policy (cmix s3), loose s2 and the B2-trained s4 vs MDD-PTECT and vs the level's best pair (pooled window
       tardiness, seed-bootstrap 95% CI, Wilcoxon). Run-to-run check: fc's MDD-PTECT vs fa's. -> analysis.out
"""
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
R = HERE.parent
_spec = importlib.util.spec_from_file_location("u", R / "rq3-gen-v5-unity" / "analyze.py")
u = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(u)
_spec4 = importlib.util.spec_from_file_location("rq4", R / "rq4-agvfail-v5-rules" / "analyze.py")
rq4 = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(rq4)
LEVELS = ["flex-p0.3", "flex-p0.5"]
PICKS = {"train-v5-cmix_s3": "cmix s3", "train-v5-loose_s2": "loose s2", "PPO s4": "B2-trained s4"}


def rename(t):
    return t.rename(columns={c: PICKS.get(c, c) for c in t.columns})


def no_flex():
    """B2 in Unity on seeds 0-19: pairs and s4 from the RQ4 level-none rows, cmix s3 / loose s2 from eval-v5-cmix-unity."""
    d = rq4.load()
    d = d[(d.level == "none") & d.seed.isin(u.SEEDS)].drop_duplicates(["policy", "seed"])
    t = d.pivot_table(index="seed", columns="policy", values="tard")
    c = u.table(sorted((R / "eval-v5-cmix-unity" / "runs" / "base" / "pc").glob("s*/episodes.csv")))
    if not c.empty:
        t = t.join(c[[k for k in c.columns if k in PICKS]], how="left")
    return rename(t)


def report(name, t, ref_rank, rng):
    pairs = [p for p in u.V5 if p in t.columns]
    if len(pairs) < 6:
        print(f"\n-- {name}: pairs not complete ({len(pairs)} / 6)")
        return None
    tot = t[pairs].sum().sort_values()
    best = tot.index[0]
    print(f"\n-- {name}: n {len(t)} seeds; MDD-PTECT mean {t[u.REF].mean():.1f} k job-s; best pair {best}; pairs vs MDD-PTECT: "
          + ", ".join(f"{p} {100 * (tot[p] / tot[u.REF] - 1):+.1f}%" for p in tot.index))
    if ref_rank is not None:
        tau, p = kendalltau(ref_rank[pairs].rank(), tot[pairs].rank())
        print(f"   pair ranking vs no flexibility: Kendall tau {tau:+.2f} (p {p:.2g})")
    t = t.assign(**{"hindsight best pair": t[pairs].min(axis=1)})
    for pol in ("cmix s3", "loose s2", "B2-trained s4", "hindsight best pair"):
        if pol in t.columns:
            print(f"   {pol:20s} vs MDD-PTECT {u.cmp(t[pol], t[u.REF], rng)}   vs best pair {u.cmp(t[pol], t[best], rng)}")
    return tot


def main():
    rng = np.random.default_rng(0)
    print("== rq3-flex-v5-unity: machine flexibility in Unity, B2 seeds 0-19; final policy = cmix pick s3")
    nf = no_flex()
    ref = report("no flexibility (B2)", nf, None, rng)
    for lv in LEVELS:
        fr = {g: rename(u.table(sorted((HERE / "runs" / lv / g).glob("s*/episodes.csv")))) for g in ("fa", "fb", "fc")}
        if any(v.empty for v in fr.values()):
            print(f"\n-- {lv}: groups missing ({[g for g, v in fr.items() if v.empty]})")
            continue
        chk = (fr["fc"][u.REF] - fr["fa"][u.REF]).abs().dropna()
        print(f"\n   {lv}: fc's MDD-PTECT vs fa's identical on {int((chk < 1e-6).sum())} / {len(chk)} seeds")
        t = fr["fa"].join(fr["fb"], how="outer").join(fr["fc"].drop(columns=[u.REF]), how="outer")
        report(lv, t, ref, rng)
        for pol in ("cmix s3", "loose s2", "B2-trained s4", u.REF):
            if pol in t.columns and pol in nf.columns:
                ok = t[pol].notna() & nf[pol].notna()
                print(f"   cost of flexibility {pol:15s} {100 * (t[pol][ok].sum() / nf[pol][ok].sum() - 1):+.1f}% (n {int(ok.sum())})")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
