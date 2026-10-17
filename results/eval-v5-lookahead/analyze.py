"""
@file analyze.py
@brief eval-v5-lookahead (2026-10-15): does PPO use the exact future arrivals? The 5 train-v5-lookahead seeds against
       MDD-PTECT and the state-only PPO (the same 5-seed stack without look-ahead, ../eval-v5-final), with the
       hindsight references on the same seeds: the greedy switching oracle (../rq2-rebase-v5/out/oracle, 900 s) and
       the twin planner on the exact schedule re-planning every 300 s (../rq2-perturbed-v5/B2/exact-slot300, test
       seeds). Pooled window tardiness, seed-bootstrap 95% CIs, Wilcoxon, and the share of the oracle's gain.
       Slot rule use is in train-v5-lookahead/analysis.out (dev_b2_slot_analyze.py). -> analysis.out
"""
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
R = HERE.parent
H = {"config_hash": str, "instance_hash": str}
REF = "MDD-PTECT"


def short(p):
    p = str(p).replace("ckpt:", "").replace("/checkpoint", "").replace("results/", "")
    for pre, lab in (("train-v5-lookahead_s", "look-ahead s"), ("train-v5-nokl_s", "state-only s"),
                     ("train-v5-final_s", "state-only s")):
        if p.startswith(pre):
            return lab + p.rsplit("_s", 1)[1]
    return p


def table(split):
    fr = [pd.read_csv(f, dtype=H) for f in sorted((R / "eval-v5-final" / split).glob("c*/episodes.csv"))]
    suffix = "__test" if split == "test" else ""
    for k in range(5):
        d = HERE / f"train-v5-lookahead_s{k}__checkpoint{suffix}"
        if (d / "DONE").exists():
            fr.append(pd.read_csv(d / "episodes.csv", dtype=H))
    e = pd.concat(fr, ignore_index=True)
    e["policy"] = e.policy.map(short)
    t = e.pivot_table(index="seed", columns="policy", values="window_tardiness") / 1000.0
    orc = {}
    for s in t.index:
        f = R / "rq2-rebase-v5" / "out" / "oracle" / f"s{s}.npz"
        if f.exists():
            orc[s] = float(np.load(f)["oracle"])
    t["oracle (900 s, hindsight)"] = pd.Series(orc)
    if split == "test":
        pl = {}
        for s in t.index:
            f = R / "rq2-perturbed-v5" / "B2" / "exact-slot300" / f"s{s}.json"
            if f.exists():
                pl[s] = json.loads(f.read_text())["real"]
        t["planner, exact schedule, 300 s"] = pd.Series(pl)
    return t


def cmp(y, b, rng):
    ok = y.notna() & b.notna()
    yy, bb = y[ok].to_numpy(), b[ok].to_numpy()
    if len(yy) < 10:
        return None
    idx = rng.integers(0, len(yy), (10000, len(yy)))
    bs = 100 * (yy[idx].sum(1) / bb[idx].sum(1) - 1)
    d = yy - bb
    return {"n": len(yy), "pooled_%": 100 * (yy.sum() / bb.sum() - 1), "ci_lo": np.percentile(bs, 2.5),
            "ci_hi": np.percentile(bs, 97.5), "better": int((d < 0).sum()), "worse": int((d > 0).sum()),
            "p": float(wilcoxon(d).pvalue) if np.any(d) else 1.0}


def main():
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 220)
    for split, label in (("fit", "fitting seeds 2000-2159"), ("test", "test seeds 0-39")):
        t = table(split)
        la = [c for c in t.columns if c.startswith("look-ahead s")]
        so = [c for c in t.columns if c.startswith("state-only s")]
        if la:
            t["look-ahead mean of %d" % len(la)] = t[la].mean(axis=1, skipna=False)
        if len(so) == 5:
            t["state-only mean of 5"] = t[so].mean(axis=1, skipna=False)
        oracle_gain = None
        if t["oracle (900 s, hindsight)"].notna().all():
            o, b = t["oracle (900 s, hindsight)"], t[REF]
            oracle_gain = o.sum() - b.sum()
        print(f"\n== {label}: vs MDD-PTECT (pooled %, CI, better / worse, Wilcoxon; share of the oracle's gain)")
        rows = []
        for c in [c for c in t.columns if c != REF]:
            s = cmp(t[c], t[REF], rng)
            if s:
                share = 100 * (t[c].sum() - t[REF].sum()) / oracle_gain if oracle_gain else np.nan
                rows.append({"policy": c, **s, "share_of_oracle_%": share})
        print(pd.DataFrame(rows).sort_values("pooled_%").to_string(index=False,
                                                                   float_format=lambda v: f"{v:.3g}" if abs(v) < 0.01 else f"{v:.2f}"))
        if la and len(so) == 5:
            s = cmp(t["look-ahead mean of %d" % len(la)], t["state-only mean of 5"], rng)
            print(f"   look-ahead mean vs state-only mean (paired): {s['pooled_%']:+.2f} [{s['ci_lo']:+.2f}, {s['ci_hi']:+.2f}] "
                  f"{s['better']}/{s['worse']} p={s['p']:.2g}")


if __name__ == "__main__":
    buf = io.StringIO()
    with redirect_stdout(buf):
        main()
    (HERE / "analysis.out").write_text(buf.getvalue())
    sys.stdout.write(buf.getvalue())
