"""
@file select_final.py
@brief eval-v5-final (2026-10-14): choose the final PPO policy on the B2 fitting seeds 2000-2159 and report it on the
       test seeds 0-39. The pick is the seed with the lowest pooled window tardiness on the fitting seeds (the same
       seeds and criterion that chose MDD-PTECT as the best fixed pair in rq2-rebase-v5). Writes selected.txt
       ("s<k> <checkpoint path>") and analysis.out. Also checks that s0-s2 on the test seeds reproduce
       train-v5-nokl/eval exactly (deterministic twin, same code).
@par Usage
@code{.sh}
.venv/bin/python results/eval-v5-final/select_final.py
@endcode
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
R = HERE.parent
HASH_STR = {"config_hash": str, "instance_hash": str}
REF = "MDD-PTECT"
CKPT = {f"PPO s{k}": f"results/train-v5-nokl_s{k}/checkpoint.pt" for k in range(3)}
CKPT.update({f"PPO s{k}": f"results/train-v5-final_s{k}/checkpoint.pt" for k in (3, 4)})


def short(p):
    p = str(p).replace("ckpt:", "").replace("/checkpoint", "").replace("results/", "")
    if "_s" in p and ("train-v5-nokl" in p or "train-v5-final" in p):
        return "PPO s" + p.rsplit("_s", 1)[1]
    return p


def load(split):
    fr = []
    for f in sorted(HERE.glob(f"{split}/c*/episodes.csv")):
        if (f.parent / "DONE").exists():
            fr.append(pd.read_csv(f, dtype=HASH_STR))
    if not fr:
        return pd.DataFrame()
    e = pd.concat(fr, ignore_index=True)
    e["policy"] = e.policy.map(short)
    e["tard"] = e.window_tardiness / 1000.0
    return e.pivot_table(index="seed", columns="policy", values="tard")


def cmp(y, b, rng):
    ok = y.notna() & b.notna()
    yy, bb = y[ok].to_numpy(), b[ok].to_numpy()
    idx = rng.integers(0, len(yy), (10000, len(yy)))
    bs = 100 * (yy[idx].sum(1) / bb[idx].sum(1) - 1)
    d = yy - bb
    p = float(wilcoxon(d).pvalue) if np.any(d) else 1.0
    return (f"{100 * (yy.sum() / bb.sum() - 1):+6.2f} [{np.percentile(bs, 2.5):+6.2f}, {np.percentile(bs, 97.5):+6.2f}] "
            f"{int((d < 0).sum()):3d}/{int((d > 0).sum()):<3d} p={p:.2g} n={len(yy)}")


def main():
    rng = np.random.default_rng(0)
    fit, test = load("fit"), load("test")
    ppo = sorted(c for c in fit.columns if c.startswith("PPO s"))
    print(f"== eval-v5-final: twin, B2; fitting seeds {len(fit)} (2000-2159), test seeds {len(test)} (0-39)")
    if len(fit) < 160 or len(ppo) < 5:
        print(f"   incomplete: {len(fit)} fitting seeds, PPO columns {ppo}; no selection written")
        return
    tot = {p: fit[p].sum() for p in ppo}
    pick = min(tot, key=tot.get)
    print("\n-- fitting seeds 2000-2159: pooled vs MDD-PTECT (seed bootstrap 95% CI, better/worse, Wilcoxon)")
    for p in ppo + ["MDD-PECT"]:
        mark = "   <- selected" if p == pick else ""
        print(f"  {p:22s} {cmp(fit[p], fit[REF], rng)}  total {fit[p].sum():9.1f}{mark}")
    fit5 = fit[ppo].mean(axis=1)
    print(f"  {'PPO mean of 5 seeds':22s} {cmp(fit5, fit[REF], rng)}")
    if len(test) >= 40:
        print("\n-- test seeds 0-39: pooled vs MDD-PTECT")
        for p in ppo + ["MDD-PECT"]:
            mark = "   <- selected (final policy)" if p == pick else ""
            print(f"  {p:22s} {cmp(test[p], test[REF], rng)}{mark}")
        print(f"  {'PPO mean of 5 seeds':22s} {cmp(test[ppo].mean(axis=1), test[REF], rng)}")
        print(f"  spread of the 5 seeds on test: "
              f"{min(100 * (test[p].sum() / test[REF].sum() - 1) for p in ppo):+.2f} to "
              f"{max(100 * (test[p].sum() / test[REF].sum() - 1) for p in ppo):+.2f}%")
        # reproduction check: s0-s2 and MDD-PTECT on the test seeds against train-v5-nokl/eval (same code, twin)
        old = pd.read_csv(R / "train-v5-nokl" / "eval" / "episodes.csv", dtype=HASH_STR)
        old["policy"] = old.policy.map(short)
        old = old.pivot_table(index="seed", columns="policy", values="window_tardiness") / 1000.0
        for p in ["PPO s0", "PPO s1", "PPO s2", REF]:
            dmax = (test[p] - old[p].reindex(test.index)).abs().max()
            print(f"  reproduce train-v5-nokl/eval {p:10s}: max |diff| {dmax:.2e} (1000 job-s)")
        new = R / "train-v5-final" / "eval" / "episodes.csv"
        if new.exists():
            o2 = pd.read_csv(new, dtype=HASH_STR)
            o2["policy"] = o2.policy.map(short)
            o2 = o2.pivot_table(index="seed", columns="policy", values="window_tardiness") / 1000.0
            for p in ["PPO s3", "PPO s4", REF]:
                if p in o2:
                    print(f"  reproduce train-v5-final/eval {p:9s}: max |diff| "
                          f"{(test[p] - o2[p].reindex(test.index)).abs().max():.2e}")
    (HERE / "selected.txt").write_text(f"{pick.split()[1]} {CKPT[pick]}\n")
    print(f"\nselected: {pick} ({CKPT[pick]}) -> selected.txt")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
