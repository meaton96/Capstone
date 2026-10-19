"""
@file analyze.py
@brief rq3-size-remap (2026-10-19): does rewriting the three size-dependent global scalars into the 15-machine range
       restore the final policy's transfer to 50 / 100 machines? Per floor and remap (none = ../rq3-size-v5, size, wip,
       opt, all): the cmix pick s3 and the cmix mean of 5 vs MDD-PTECT (pooled window tardiness, seed-bootstrap 95% CI,
       Wilcoxon), and each remap vs none, paired. Run-to-run check: MDD-PTECT here = rq3-size-v5's pdr run.
       -> analysis.out, summary.csv
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
_s = importlib.util.spec_from_file_location("size_an", R / "rq3-size-v5" / "analyze.py")
size = importlib.util.module_from_spec(_s)
_s.loader.exec_module(size)
rq3 = size.rq3
REMAPS = ["none", "size", "wip", "opt", "all"]
FLOORS = ["m50", "m100"]
REF = rq3.REF


def cmix_table(remap, cond):
    root = (R / "rq3-size-v5") if remap == "none" else (HERE / remap)
    files = [root / cond / "cmix" / "episodes.csv"] + sorted((root / cond).glob("cmix_s*-*/episodes.csv"))
    t = size.table(files)
    mem = [f"cmix s{k}" for k in range(5)]
    if not t.empty and all(m in t.columns for m in mem):
        t["cmix mean of 5"] = t[mem].mean(axis=1, skipna=False)
    return t


def main():
    rng = np.random.default_rng(0)
    print("== rq3-size-remap: final policy (cmix, trained on 15 machines) with the size-dependent global scalars "
          "rewritten into the 15-machine range; test seeds 0-39; < 0 better")
    rows = []
    for cond in FLOORS:
        ref_pdr = size.table([R / "rq3-size-v5" / cond / "pdr" / "episodes.csv"]
                             + sorted((R / "rq3-size-v5" / cond).glob("pdr_s*-*/episodes.csv")))
        base = cmix_table("none", cond)
        for remap in REMAPS:
            t = cmix_table(remap, cond)
            if t.empty or "cmix mean of 5" not in t.columns or len(t) < 40:
                print(f"\n-- {cond} / {remap}: not complete yet ({len(t)} seeds)")
                continue
            chk = (t[REF] - ref_pdr[REF].reindex(t.index)).abs()
            print(f"\n-- {cond} / {remap}: MDD-PTECT = rq3-size-v5's on {int((chk < 1e-6).sum())} / {len(chk)} seeds")
            for pol in ("cmix s3", "cmix mean of 5"):
                c = rq3.compare(t[pol].to_dict(), t[REF].to_dict(), rng)
                d = rq3.compare(t[pol].to_dict(), base[pol].to_dict(), rng) if remap != "none" else None
                rows.append({"floor": cond, "remap": remap, "policy": pol, **{f"vs_ref_{k}": v for k, v in c.items()},
                             **({f"vs_none_{k}": v for k, v in d.items()} if d else {})})
                line = (f"   {pol:16s} vs MDD-PTECT {c['pooled_%']:+6.2f} [{c['ci_lo']:+6.2f}, {c['ci_hi']:+6.2f}] "
                        f"{c['better']:2d}/{c['worse']:<2d} p={c['wilcoxon_p']:.2g}")
                if d:
                    line += (f"   | vs no remap {d['pooled_%']:+6.2f} [{d['ci_lo']:+6.2f}, {d['ci_hi']:+6.2f}] "
                             f"{d['better']:2d}/{d['worse']:<2d} p={d['wilcoxon_p']:.2g}")
                print(line)
    pd.DataFrame(rows).to_csv(HERE / "summary.csv", index=False)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
