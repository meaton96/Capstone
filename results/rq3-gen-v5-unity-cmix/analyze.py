"""
@file analyze.py
@brief rq3-gen-v5-unity-cmix (2026-10-17): the RQ3 Unity subset (load-1.2, due-tight, lay-g; B2 seeds 0-19) for the final
       policy (cmix pick s3) and the loose pick s2, next to the 6 pairs and the B2-trained pick s4 from
       ../rq3-gen-v5-unity, in Unity and in the twin on the same seeds (twin: ../rq3-gen-v5 pairs / s4,
       ../rq3-gen-v5-cmix and ../eval-v5-cmix/test for the cmix / loose families). -> analysis.out
"""
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent
_spec = importlib.util.spec_from_file_location("u", R / "rq3-gen-v5-unity" / "analyze.py")
u = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(u)
H = {"config_hash": str, "instance_hash": str}
CONDS = ["load-1.2", "due-tight", "lay-g"]
PICKS = {"train-v5-cmix_s3": "cmix s3", "train-v5-loose_s2": "loose s2", "PPO s4": "B2-trained s4"}


def rename(t):
    return t.rename(columns={c: PICKS.get(c, c) for c in t.columns})


def twin(cond):
    """Twin table on seeds 0-19: rq3-gen-v5 pdr / ppo groups + the cmix / loose families' picks."""
    t = u.table([R / "rq3-gen-v5" / cond / g / "episodes.csv" for g in ("pdr", "ppo")])
    root = R / "eval-v5-cmix" / "test" if cond in ("base", "due-tight", "due-loose") else R / "rq3-gen-v5-cmix"
    f = u.table([root / cond / g / "episodes.csv" for g in ("cmix", "loose")])
    if not f.empty:
        f = f[[c for c in f.columns if c in PICKS]]
        t = t.join(f, how="outer") if not t.empty else f
    return rename(t)


def main():
    rng = np.random.default_rng(0)
    print("== rq3-gen-v5-unity-cmix: Unity vs twin on B2 seeds 0-19; final policy = cmix pick s3")
    for cond in CONDS:
        tu = rename(u.table(sorted((R / "rq3-gen-v5-unity" / "runs" / cond).glob("p[ab]/s*/episodes.csv"))))
        pc = rename(u.table(sorted((HERE / "runs" / cond / "pc").glob("s*/episodes.csv"))))
        if pc.empty:
            print(f"\n-- {cond}: no pc runs yet")
            continue
        chk = pc[u.REF].sub(tu[u.REF]).abs().dropna()
        print(f"\n-- {cond}: pc runs {len(pc)} seeds; MDD-PTECT vs rq3-gen-v5-unity's run: identical on "
              f"{int((chk < 1e-6).sum())} / {len(chk)} seeds")
        tu = tu.join(pc.drop(columns=[u.REF]), how="outer")
        tt = twin(cond)
        for name, t in (("Unity", tu), ("twin", tt)):
            pairs = [p for p in u.V5 if p in t.columns]
            best = min(pairs, key=lambda p: t[p].sum())
            t = t.assign(**{"hindsight best pair": t[pairs].min(axis=1)})
            print(f"   {name}: best pair {best}")
            for pol in ("cmix s3", "loose s2", "B2-trained s4", "hindsight best pair"):
                if pol in t.columns:
                    print(f"     {pol:20s} vs MDD-PTECT {u.cmp(t[pol], t[u.REF], rng)}   vs best pair {u.cmp(t[pol], t[best], rng)}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
