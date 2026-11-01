"""
@file pick.py
@brief rq3-size-m1000: picks the seed of each new training arm for the 1000-machine column, before any of them runs
       there: the seed with the lowest pooled window tardiness vs MDD-PTECT at 200 machines in ../rq3-size-grid
       (summary.csv). The same rule picks s3 (the final policy) for the 15-machine arm. -> picks.json
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ARMS = {"t50": "T50", "t100": "T100", "tmix": "Tmix", "t15": "T15"}

s = pd.read_csv(HERE.parent / "rq3-size-grid" / "summary.csv")
s = s[(s.floor == "m200") & (s.vs == "MDD-PTECT")]
picks, out = {}, []
for g, arm in ARMS.items():
    r = s[s.policy.str.startswith(f"{arm} s")]
    if len(r) != 5:
        raise SystemExit(f"{arm}: {len(r)} / 5 seeds at m200 in rq3-size-grid/summary.csv")
    best = r.loc[r["pooled_%"].idxmin()]
    picks[g] = int(best.policy.rsplit("s", 1)[1])
    out.append(f"{arm}: " + ", ".join(f"{p.split()[-1]} {v:+.2f}%" for p, v in zip(r.policy, r["pooled_%"]))
               + f" -> s{picks[g]}")
if picks["t15"] != 3:
    print("note: the rule does not give the final policy (s3) for T15; t15 still runs s3")
picks.pop("t15")
(HERE / "picks.json").write_text(json.dumps(picks, indent=1) + "\n")
print("\n".join(out))
print("picks.json:", picks)
