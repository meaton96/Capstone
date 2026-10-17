"""
@file run_task.py
@brief rq3-gen-v5 (2026-10-14): one (condition, group) evaluation of the RQ3 conditions sweep on the v5 rules, twin,
       B2 test seeds 0-39. The v5 rerun of ../rq3-gen-twin (same 11 conditions, same floors, same fleet wrapper),
       now with the v5 pairs and the final PPO stack (5 seeds) instead of the H15 pairs and one seed per method.

@details Calls evaluate.py from the v5 frozen code in-process (results/rq2-rebase-v5/code/env = the 6d13412a snapshot
the PPO and BC checkpoints were trained and evaluated with) with the B2 settings of
results/scripts/dev_b2_slot_train_eval.sh (tardiness reward spec, random warm-up, 21,600 s window, deterministic,
900 s slot actions, 5,400 s discount horizon as trained), changed per condition (CONDITIONS below). Output:
<condition>/<group>/ (episodes.csv, eval.out, cond.json, DONE). A task whose DONE exists is skipped; an unfinished
one is wiped and rerun, so resubmitting resumes.

Groups:
  pdr  the 6 v5 pairs {SRT, MDD, ATC} x {PECT, PTECT} (--pdr all) + MDD-TECT (the H15 best pair on B2)
  ppo  PPO train-v5-nokl s0 / s1 / s2 + train-v5-final s3 / s4 (the 5-seed final stack) + MDD-PTECT
  bc   BC of the v5 oracle, rq2-oracle-bc-v5 bc_full_s0 / s1 / s2 + MDD-PTECT
(MDD-PTECT in ppo / bc is a run-to-run check against the pdr group: the twin is deterministic.)

Fleet conditions (agv5/7/9-bay9): as in rq3-gen-twin, evaluate.py --agvs cannot express them on the twin (the twin
requires the scenario's agvCount to equal the floor's bays, and the B2 short-fleet blocks set their own on-duty count
through "agvSchedule"). The generator is wrapped instead: agvCount = 9 (the D9 export's bays and vehicles), and the
fleet schedule puts N on duty wherever the instance had its full fleet (7) and keeps the short-fleet blocks at 2.
AGVs above N stay parked in their bays. Jobs are unchanged, so seeds are paired across conditions.

@par Usage (from the repo root; the cluster array calls it with one line of tasks.txt)
@code{.sh}
.venv/bin/python results/rq3-gen-v5/run_task.py --cond base --group pdr [--seeds 0-39] [--device cpu]
@endcode
"""
import argparse
import copy
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CODE = REPO / "results" / "rq2-rebase-v5" / "code" / "env"   # = train-v5-nokl / train-v5-final snapshot (6d13412a)

# B2 (results/scripts/dev_b2_slot_train_eval.sh PARAMS)
B2 = {"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400,
      "horizon_seconds": 30600,
      "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2},
                   {"name": "surge", "utilization": [1.0, 1.6]},
                   {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}
LULL = (0.2, 0.4)          # RandomizedParams.lull_utilization default (lull segments inside every profile)


def scaled_load(f):
    """B2 with every utilization range scaled by f (the profiles' ranges and the lull segments' range)."""
    p = copy.deepcopy(B2)
    for prof in p["load_mix"]:
        prof["utilization"] = [round(u * f, 4) for u in prof["utilization"]]
        prof["lull_utilization"] = [round(u * f, 4) for u in LULL]
    return p


def with_due(lo, hi):
    p = copy.deepcopy(B2)
    p["due_date_allowance_range"] = [lo, hi]
    return p


FLOORS = REPO / "results" / "rq3-gen-twin" / "floors"   # the rq3-gen-twin Unity exports (D7 = dev-popart-smoke/floor7)
# name -> (params, floor export, layout override, (on-duty fleet, bays) or None)
CONDITIONS = {
    "base":      (B2, FLOORS / "D7" / "des_floor.json", None, None),
    "load-0.8":  (scaled_load(0.8), FLOORS / "D7" / "des_floor.json", None, None),
    "load-1.2":  (scaled_load(1.2), FLOORS / "D7" / "des_floor.json", None, None),
    "due-tight": (with_due(1.25, 1.75), FLOORS / "D7" / "des_floor.json", None, None),
    "due-loose": (with_due(2.5, 3.5), FLOORS / "D7" / "des_floor.json", None, None),
    "agv5-bay9": (B2, FLOORS / "D9" / "des_floor.json", None, (5, 9)),
    "agv7-bay9": (B2, FLOORS / "D9" / "des_floor.json", None, (7, 9)),
    "agv9-bay9": (B2, FLOORS / "D9" / "des_floor.json", None, (9, 9)),
    "lay-c":     (B2, FLOORS / "C7" / "des_floor.json", "C", None),
    "lay-g":     (B2, FLOORS / "G7" / "des_floor.json", "G", None),
    "lay-j":     (B2, FLOORS / "J7" / "des_floor.json", "J", None),
}
PPO = ["results/train-v5-nokl_s0/checkpoint.pt", "results/train-v5-nokl_s1/checkpoint.pt",
       "results/train-v5-nokl_s2/checkpoint.pt", "results/train-v5-final_s3/checkpoint.pt",
       "results/train-v5-final_s4/checkpoint.pt"]
BC = [f"results/rq2-oracle-bc-v5/bc/bc_full_s{k}.pt" for k in range(3)]
GROUPS = {
    "pdr": (900, ["--pdr", "all,MDD-TECT"]),
    "ppo": (900, ["--pdr", "MDD-PTECT"] + [a for c in PPO for a in ("--checkpoint", c)]),
    "bc":  (900, ["--pdr", "MDD-PTECT"] + [a for c in BC for a in ("--checkpoint", c)]),
}


def n_policies(group):
    _, a = GROUPS[group]
    pdr = a[1].replace("all", ",".join(["x"] * 6))      # "all" = the 6 v5 pairs
    return len([p for p in pdr.split(",") if p]) + a.count("--checkpoint")


def fleet_wrapper(on_duty, bays):
    """Wraps scenarios.REGISTRY["randomized"]: agvCount = bays, full-fleet blocks get `on_duty` AGVs on duty."""
    import scenarios
    orig = scenarios.REGISTRY["randomized"]

    def factory(*a, **kw):
        gen = orig(*a, **kw)

        def g(seed):
            s = copy.deepcopy(gen(seed))
            full = int(s["agvCount"])                    # 7: the generator's fleet (the largest block fleet)
            sched = s.get("agvSchedule") or [{"start": 0, "agvCount": full}]
            s["agvSchedule"] = [{"start": e["start"],
                                 "agvCount": on_duty if int(e["agvCount"]) == full else min(int(e["agvCount"]), on_duty)}
                                for e in sched]
            s["agvCount"] = bays
            return s
        return g

    scenarios.REGISTRY["randomized"] = factory


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cond", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    ap.add_argument("--seeds", default="0-39")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-root", default=str(HERE), help="parent of <cond>/<group>/ (default: this folder)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    params, floor, layout, fleet = CONDITIONS[args.cond]
    slot, extra = GROUPS[args.group]
    out = Path(args.out_root) / args.cond / args.group
    if (out / "DONE").exists():
        print(f"{args.cond}/{args.group}: done, skipped")
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ev = ["--twin", str(floor), "--reward-spec", str(CODE / "config" / "rewards" / "tardiness.json"),
          "--scenario-generator", "randomized", "--random-warmup", "--episode-duration-seconds", "21600",
          "--params", json.dumps(params), "--seeds", args.seeds, "--slot-seconds", str(slot),
          "--discount-horizon-s", "5400", "--device", args.device, *extra, "--out", str(out)]
    if layout:
        ev += ["--layout", layout]
    (out / "cond.json").write_text(json.dumps({"cond": args.cond, "group": args.group, "params": params,
                                               "floor": str(floor.relative_to(REPO)), "layout": layout,
                                               "fleet_on_duty_bays": fleet, "slot_seconds": slot,
                                               "stamp": (CODE.parent / "STAMP").read_text(),
                                               "evaluate_args": ev}, indent=1))
    sys.path.insert(0, str(CODE))
    import os
    os.chdir(REPO)                       # checkpoint paths are repo-relative
    if fleet is not None:
        fleet_wrapper(*fleet)
    import evaluate
    t0 = time.time()
    with open(out / "eval.out", "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        evaluate.main(ev)
    import csv
    with open(out / "episodes.csv") as f:
        rows = list(csv.DictReader(f))
    seeds = evaluate.parse_seeds(args.seeds)
    want = n_policies(args.group) * len(seeds)
    if len(rows) != want:
        raise SystemExit(f"{args.cond}/{args.group}: {len(rows)} episodes, expected {want} (see {out}/eval.out)")
    (out / "DONE").write_text(f"{time.time() - t0:.0f}\n")
    print(f"{args.cond}/{args.group}: {len(rows)} episodes in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
