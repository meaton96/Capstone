"""
@file task.py
@brief rq3-size-v5 (2026-10-17, user: set up RQ3 floor size): the v5 rules and the PPO families (final policy = cmix
       pick s3) on larger floors in the twin, B2 test seeds 0-39. A thin wrapper around ../rq3-gen-v5/run_task.py (same
       frozen code, B2 window / slot / reward settings and fleet wrapper) with two new conditions:
  m50   50 machines (10 per type), layout D, floor results/agv-ratio-scale/floors/m50 (50 bays); dev-hetero-tune's m50
        family: 26 AGVs on duty (short-fleet blocks 13), B2 load mix scaled by 0.9, arrival cap off; due dates c ~ U[1.75, 2.5]
  m100  100 machines (20 per type), floor .../floors/m100 (100 bays); the same per-machine scaling: 52 AGVs (short 26)
  (m15 = rq3-gen-v5 / eval-v5-cmix base: 15 machines, 7 AGVs.)
The policies were trained on the 15-machine floor only; this asks whether they, and the rule ranking, carry over.
Groups: pdr (6 v5 pairs + MDD-TECT), ppo (B2-trained s0-s4), cmix (s0-s4), loose (s0-s4); each policy group + MDD-PTECT
as a run-to-run check. Output: <cond>/<group>/; a finished task (DONE) is skipped.
@par Usage
@code{.sh}
python results/rq3-size-v5/task.py --cond m50 --group cmix [--seeds 0-39]
@endcode
"""
import argparse
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "rq3-gen-v5"))
import run_task as rt                                              # noqa: E402  (rq3-gen-v5's harness)

FLOORS = REPO / "results" / "agv-ratio-scale" / "floors"
LOAD = 0.9                                                         # dev-hetero-tune / dev-pect-rule m50 (HETERO_LOAD)


def size_params(per_type, fleet, short):
    p = copy.deepcopy(rt.B2)
    p.update(machines_per_type=per_type, agv_count=fleet, agv_max_utilization=10.0)
    for prof in p["load_mix"]:
        prof["utilization"] = [round(u * LOAD, 4) for u in prof["utilization"]]
        if prof["name"] == "short":
            prof["agv_count"] = short
    return p


rt.CONDITIONS["m50"] = (size_params(10, 26, 13), FLOORS / "m50" / "des_floor.json", None, (26, 50))
rt.CONDITIONS["m100"] = (size_params(20, 52, 26), FLOORS / "m100" / "des_floor.json", None, (52, 100))
for name, run in (("loose", "train-v5-loose"), ("cmix", "train-v5-cmix")):
    ck = [f"results/{run}_s{k}/checkpoint.pt" for k in range(5)]
    rt.GROUPS[name] = (900, ["--pdr", "MDD-PTECT"] + [a for c in ck for a in ("--checkpoint", c)])
rt.GROUPS.pop("bc", None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["m50", "m100"])
    ap.add_argument("--group", required=True, choices=["pdr", "ppo", "cmix", "loose"])
    ap.add_argument("--seeds", default="0-39")
    ap.add_argument("--out-root", default=str(HERE))
    a = ap.parse_args(argv)
    rt.main(["--cond", a.cond, "--group", a.group, "--seeds", a.seeds, "--out-root", a.out_root])


if __name__ == "__main__":
    main()
