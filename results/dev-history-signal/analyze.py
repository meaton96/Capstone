"""
@file analyze.py
@brief dev-history-signal (2026-10-06): would memory (recurrence / attention over past slots) help the slot policy?
       Cheap check before any architecture change: does the state of the previous slots predict "ATC vs MDD-TECT"
       beyond the current observation?

Uses the per-slot states and targets of ../dev-congestion-signal (oracle trajectories, B2 seeds 0-39, 960 slots; no new
simulation). Feature sets, all at the first decision of slot k:
  - now:        obs v3 global scalars + event flags (what the policy's MLP sees directly)
  - +history:   now + the same columns at slots k-1 and k-2 and the change since k-1 and k-4 (what a recurrent or
                temporal-attention policy could summarize; missing at the episode start)
  - +prev pair: now + the pair played in slot k-1 (job and machine rule, one-hot), what a policy with its previous
                action in the input would know
Targets as in dev-congestion-signal: dev1 / dev4 (swap ATC-TECT vs MDD-TECT into 1 slot / 1 h of the oracle schedule)
and hold (held to the end). Out-of-fold by seed, 10 partitions; slots 0-1 are dropped so every set has its lags.
@par Usage
@code{.sh}
OMP_NUM_THREADS=4 .venv/bin/python results/dev-history-signal/analyze.py > results/dev-history-signal/analysis.out
@endcode
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "dev-congestion-signal"
_spec = importlib.util.spec_from_file_location("cong", SRC / "analyze.py")
cong = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cong)


def main():
    d = pd.concat([pd.read_csv(p) for p in sorted((SRC / "rows").glob("s*.csv"))], ignore_index=True)
    d = d.merge(pd.concat([pd.read_csv(p) for p in sorted((SRC / "dev").glob("s*.csv"))]), on=["seed", "slot"])
    d = d.sort_values(["seed", "slot"]).reset_index(drop=True)
    now = [c for c in d.columns if (c[0] in "sf" and c[1:].isdigit()) and d[c].nunique() > 1]
    g = d.groupby("seed")
    hist = []
    for c in now:
        for lag in (1, 2):
            d[f"{c}_l{lag}"] = g[c].shift(lag)
            hist.append(f"{c}_l{lag}")
        d[f"{c}_d1"] = d[c] - g[c].shift(1)
        d[f"{c}_d4"] = d[c] - g[c].shift(4)
        hist += [f"{c}_d1", f"{c}_d4"]
    prev = g.oracle_pair.shift(1).fillna("none")
    for r in ("SRT", "SPT", "MDD", "EDD", "ATC"):
        d[f"prev_job_{r}"] = prev.str.startswith(r + "-").astype(float)
    for r in ("ECT", "TECT", "SRWT"):
        d[f"prev_machine_{r}"] = prev.str.endswith("-" + r).astype(float)
    prevc = [c for c in d.columns if c.startswith("prev_") and d[c].nunique() > 1]
    dd = d[d.slot >= 2].copy()
    print(f"{dd.seed.nunique()} seeds, {len(dd)} slots (slots 2-23); now {len(now)} columns, history {len(hist)}, "
          f"prev pair {len(prevc)}")

    print("\n== strongest history columns (|Spearman| with dev4, partial: after the current-state model)")
    for c in sorted(hist, key=lambda c: -abs(spearmanr(dd[c], dd.dev4, nan_policy="omit")[0]))[:8]:
        print(f"  {c:12s} {spearmanr(dd[c], dd.dev4, nan_policy='omit')[0]:+.2f}")

    # Reuse dev-congestion-signal's comparison: "obs" = now, "cong" = the extra set, "both" = now + extra.
    for extra, label in ((hist, "+history"), (prevc, "+prev pair")):
        print(f"\n######## {label} (printed as 'cong' = extra alone, 'both' = now + extra)")
        for target in ("dev1", "dev4", "d_ATC-TECT"):
            t = dd[dd[target].abs() > 1e-9] if target.startswith("dev") else dd
            cong.compare(t, now, extra, target, f"{label}, target {target}")


if __name__ == "__main__":
    main()
