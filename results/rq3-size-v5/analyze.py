"""
@file analyze.py
@brief rq3-size-v5 (2026-10-17): floor size in the twin. The 6 v5 pairs and the PPO families (final policy = cmix
       pick s3; loose pick s2; B2-trained pick s4) on 15 (B2: ../rq3-gen-v5 base + ../eval-v5-cmix/test base), 50 and
       100 machines, B2 test seeds 0-39. Per floor: the best pair and the pair ranking against 15 machines (Kendall
       tau), and each policy vs MDD-PTECT and vs the floor's best pair (pooled window tardiness, seed-bootstrap 95% CI,
       Wilcoxon). -> analysis.out, summary.csv
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
_spec = importlib.util.spec_from_file_location("rq3an", R / "rq3-gen-v5" / "analyze.py")
rq3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq3)
H = {"config_hash": str, "instance_hash": str}
SEEDS = list(range(40))
PICKS = {"cmix": "s3", "loose": "s2", "PPO": "s4"}


def name(p):
    p = str(p).replace("ckpt:", "").replace("/checkpoint", "").replace("results/", "")
    for pre, lab in (("train-v5-cmix_s", "cmix s"), ("train-v5-loose_s", "loose s"), ("train-v5-nokl_s", "PPO s"),
                     ("train-v5-final_s", "PPO s")):
        if p.startswith(pre):
            return lab + p.rsplit("_s", 1)[1]
    return p


def table(files):
    fr = [pd.read_csv(f, dtype=H) for f in files if f.exists() and (f.parent / "DONE").exists()]
    if not fr:
        return pd.DataFrame()
    e = pd.concat(fr, ignore_index=True)
    e["policy"] = e.policy.map(name)
    e = e[e.seed.isin(SEEDS)].drop_duplicates(["policy", "seed"])
    return e.pivot_table(index="seed", columns="policy", values="window_tardiness") / 1000.0


def floor_table(cond):
    if cond == "m15":
        return table([R / "rq3-gen-v5" / "base" / g / "episodes.csv" for g in ("pdr", "ppo")]
                     + [R / "eval-v5-cmix" / "test" / "base" / g / "episodes.csv" for g in ("cmix", "loose")])
    files = [HERE / cond / g / "episodes.csv" for g in ("pdr", "ppo", "cmix", "loose")]
    files += sorted((HERE / cond).glob("*_s*-*/episodes.csv"))           # m100 seed chunks
    return table(files)


def main():
    rng = np.random.default_rng(0)
    print("== rq3-size-v5: floor size in the twin, B2 test seeds 0-39; final policy = cmix pick s3 (trained on 15 machines)")
    rows, rank15 = [], None
    for cond in ("m15", "m50", "m100"):
        t = floor_table(cond)
        pairs = [p for p in rq3.V5 if p in t.columns]
        if len(pairs) < 6:
            print(f"\n-- {cond}: pairs not complete yet ({len(pairs)} / 6)")
            continue
        tot = t[pairs].sum().sort_values()
        best = tot.index[0]
        print(f"\n-- {cond}: MDD-PTECT mean {t[rq3.REF].mean():.1f} k job-s; best pair {best}; pairs vs MDD-PTECT: "
              + ", ".join(f"{p} {100 * (tot[p] / tot[rq3.REF] - 1):+.1f}%" for p in tot.index))
        if cond == "m15":
            rank15 = tot
        else:
            tau, p = kendalltau(rank15[pairs].rank(), tot[pairs].rank())
            print(f"   pair ranking vs 15 machines: Kendall tau {tau:+.2f} (p {p:.2g})")
        t = t.assign(**{"hindsight best v5 pair": t[pairs].min(axis=1)})
        for fam in ("cmix", "loose", "PPO"):
            mem = [f"{fam} s{k}" for k in range(5)]
            if all(m in t.columns for m in mem):
                t[f"{fam} mean of 5"] = t[mem].mean(axis=1, skipna=False)
        for pol in ("cmix s3", "cmix mean of 5", "loose s2", "loose mean of 5", "PPO s4", "PPO mean of 5",
                    "hindsight best v5 pair"):
            if pol not in t.columns:
                continue
            for vs, b in (("MDD-PTECT", rq3.REF), ("best pair", best)):
                c = rq3.compare(t[pol].to_dict(), t[b].to_dict(), rng)
                if c:
                    rows.append({"floor": cond, "policy": pol, "vs": vs, "best_pair": best, **c})
                    if vs == "MDD-PTECT":
                        print(f"   {pol:24s} vs MDD-PTECT {c['pooled_%']:+6.2f} [{c['ci_lo']:+6.2f}, {c['ci_hi']:+6.2f}] "
                              f"{c['better']:2d}/{c['worse']:<2d} p={c['wilcoxon_p']:.2g}")
    pd.DataFrame(rows).to_csv(HERE / "summary.csv", index=False)


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
