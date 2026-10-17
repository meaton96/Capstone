"""
@file analyze.py
@brief eval-v5-curve (2026-10-14): fixed-instance learning curve of the 5 v5 PPO seeds. Pooled window tardiness on
       the B2 fitting seeds 2000-2159 vs MDD-PTECT (from ../eval-v5-final/fit), at init (untrained), 51,200 steps and
       the final ~100k steps (../eval-v5-final/fit), with seed-bootstrap CIs. -> analysis.out
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIT = HERE.parent / "eval-v5-final" / "fit"
H = {"config_hash": str, "instance_hash": str}
RUNS = {"s0": "train-v5-nokl_s0", "s1": "train-v5-nokl_s1", "s2": "train-v5-nokl_s2", "s3": "train-v5-final_s3",
        "s4": "train-v5-final_s4"}


def pooled(y, b, rng):
    idx = rng.integers(0, len(y), (5000, len(y)))
    bs = 100 * (y[idx].sum(1) / b[idx].sum(1) - 1)
    return 100 * (y.sum() / b.sum() - 1), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((y < b).sum())


def main():
    rng = np.random.default_rng(0)
    f = pd.concat([pd.read_csv(x, dtype=H) for x in sorted(FIT.glob("c*/episodes.csv"))], ignore_index=True)
    f["tard"] = f.window_tardiness / 1000.0
    ref = f[f.policy == "MDD-PTECT"].set_index("seed").tard.sort_index()
    rows, mean = [], {}
    for k, run in RUNS.items():
        for label, src in (("init", f"{run}__checkpoint_init"), ("51.2k", f"{run}__checkpoint_step51200"), ("final", None)):
            if src is None:
                e = f[f.policy == f"ckpt:{run}/checkpoint"]
            else:
                p = HERE / src / "episodes.csv"
                if not (HERE / src / "DONE").exists():
                    continue
                e = pd.read_csv(p, dtype=H)
            t = (e.set_index("seed").window_tardiness / 1000.0).reindex(ref.index)
            if t.isna().any():
                continue
            m, lo, hi, better = pooled(t.to_numpy(), ref.to_numpy(), rng)
            rows.append({"seed": k, "stage": label, "vs_MDD-PTECT_%": m, "ci_lo": lo, "ci_hi": hi, "better_of_160": better})
            mean.setdefault(label, []).append(t.to_numpy())
    r = pd.DataFrame(rows)
    print("== eval-v5-curve: PPO on B2 fitting seeds 2000-2159 (twin), pooled window tardiness vs MDD-PTECT (< 0 better)")
    print(r.pivot(index="seed", columns="stage", values="vs_MDD-PTECT_%").reindex(columns=["init", "51.2k", "final"])
          .round(2).to_string())
    print("\n-- with 95% CIs")
    print(r.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    test = HERE.parent / "eval-v5-final" / "test"
    te = pd.concat([pd.read_csv(x, dtype=H) for x in sorted(test.glob("c*/episodes.csv"))], ignore_index=True)
    tref = (te[te.policy == "MDD-PTECT"].set_index("seed").window_tardiness / 1000.0).sort_index()
    trows, tmean = [], {"51.2k": [], "final": []}
    for k, run in RUNS.items():
        p = HERE / f"{run}__checkpoint_step51200__test"
        if (p / "DONE").exists():
            e51 = (pd.read_csv(p / "episodes.csv", dtype=H).set_index("seed").window_tardiness / 1000.0).reindex(tref.index)
            efin = (te[te.policy == f"ckpt:{run}/checkpoint"].set_index("seed").window_tardiness / 1000.0).reindex(tref.index)
            for label, t in (("51.2k", e51), ("final", efin)):
                m, lo, hi, better = pooled(t.to_numpy(), tref.to_numpy(), rng)
                trows.append({"seed": k, "stage": label, "vs_MDD-PTECT_%": m, "ci_lo": lo, "ci_hi": hi, "better_of_40": better})
                tmean[label].append(t.to_numpy())
    if trows:
        print("\n== test seeds 0-39 (early stopping at 51.2k vs the final checkpoint), pooled vs MDD-PTECT")
        print(pd.DataFrame(trows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        for label in ("51.2k", "final"):
            if len(tmean[label]) == 5:
                m, lo, hi, better = pooled(np.mean(tmean[label], axis=0), tref.to_numpy(), rng)
                print(f"  5-seed mean {label:6s} {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  better on {better}/40")
    print("\n-- per-instance mean over the 5 seeds")
    for label in ("init", "51.2k", "final"):
        if len(mean.get(label, [])) == 5:
            m, lo, hi, better = pooled(np.mean(mean[label], axis=0), ref.to_numpy(), rng)
            print(f"  {label:6s} {m:+.2f} [{lo:+.2f}, {hi:+.2f}]  better on {better}/160")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
