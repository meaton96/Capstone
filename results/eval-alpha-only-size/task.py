"""
@file task.py
@brief eval-alpha-only-size (2026-10-28, user: "set up the alpha head only check"): one task of the alpha-head-only
       evaluation of the law + offset arms (train-alpha-off-het / -b2, v7 code: alpha = min(4, max(0, 3 - 6 sigma) x
       offset)) on B2 and the floor-size grid's 50 / 100 / 200-machine floors, twin, B2 test seeds 0-39. In an
       alpha-only policy the job and machine heads are fixed to MDD-PTECT and only the network's offset head chooses, so
       it plays MDD-PTECT@x<offset> each slot. The floors are homogeneous (sigma 0), so the law is alpha 3 and the offsets
       give alpha 3 / 1.5 / 0.75 / 4 / 4 (x1 / x0.5 / x0.25 / x1.5 / x2).
       The heads are factorized (one Categorical per branch given the state, models/actor_critic.py), so fixing heads 0
       and 1 leaves the offset head's distribution unchanged; it sees the states its own MDD-PTECT play produces.
Floors (../rq3-size-grid conditions, as ../eval-alpha-off-size; m15 = ../rq3-gen-v5 base = B2):
  m15   B2 (floor7, 7 AGVs)                               one task per group (40 seeds)
  m50   26 (13) AGVs, ../agv-ratio-scale/floors/m50      one task per group (40 seeds)
  m100  52 (26) AGVs, ../agv-ratio-scale/floors/m100     10-seed chunks
  m200  104 (52) AGVs, ../dev-size-floors/floors/m200    5-seed chunks
Groups:
  ao-het   train-alpha-off-het s0-s4 alpha-only, plus MDD-PTECT (the run-to-run check against the grid's pdr group);
           on m15 also MDD-PTECT@1.5 / @0.75 (m50-m200 have them in ../eval-alpha-off-size's alpha group)
  ao-b2    train-alpha-off-b2 s0-s4 alpha-only (no MDD-PTECT: ao-het's tasks check every floor and seed)
  off-het, off-b2 (m15 only)  the full arms, plus MDD-PTECT: a harness check against ../eval-alpha-offset's B2 runs
The B2 window / slot / reward arguments, the fleet wrapper and the address-space cap are ../eval-alpha-off-size's.
evaluate.py is the arms' own v7 snapshot (../train-alpha-off-het/code/env, identical for both arms); for the ao groups
this process replaces evaluate.CheckpointPolicy with AlphaOnlyPolicy before build_policies runs (the frozen code is not
edited). Output: out/<floor>/<group>_s<a>-<b>/ (episodes.csv, decisions.csv, eval.out, cond.json, blocks.json, DONE).
A task whose DONE exists is skipped; an unfinished one is wiped and rerun, so relaunching resumes.
@par Usage (from the repo root; tasks.txt lists the tasks, launch.sh runs them)
@code{.sh}
.venv/bin/python results/eval-alpha-only-size/task.py m200 ao-het 0-4
# dry-run check of the wrapper: offset head forced to index K must equal MDD-PTECT@x<offset K>
.venv/bin/python results/eval-alpha-only-size/task.py m15 ao-het 0-1 --force-offset 2 --rules MDD-PTECT@0.75 \
    --checkpoints results/train-alpha-off-het_s0/checkpoint.pt --out-root <scratch>
@endcode
"""
import argparse
import contextlib
import csv
import importlib.util
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
REPO = HERE.parents[1]
FLOORS = ("m15", "m50", "m100", "m200")
CHUNKS = {"m15": ["0-39"], "m50": ["0-39"], "m100": [f"{a}-{a + 9}" for a in range(0, 40, 10)],
          "m200": [f"{a}-{a + 4}" for a in range(0, 40, 5)]}
ARMS = {"ao-het": "train-alpha-off-het", "ao-b2": "train-alpha-off-b2",
        "off-het": "train-alpha-off-het", "off-b2": "train-alpha-off-b2"}
GROUPS = tuple(ARMS)
## @brief The fixed rules each group also runs: ao-het and the full arms carry the MDD-PTECT check; on m15 ao-het also
##        runs the fixed alphas the offsets reach.
RULES = {"ao-het": "MDD-PTECT", "ao-b2": "none", "off-het": "MDD-PTECT", "off-b2": "MDD-PTECT"}
RULES_M15 = {"ao-het": "MDD-PTECT,MDD-PTECT@1.5,MDD-PTECT@0.75"}
## @brief Address-space cap of a task (as ../eval-alpha-off-size; an m200 episode peaks near 0.74 GB resident).
AS_BYTES = 12 << 30


def code_of(group):
    return R / ARMS[group] / "code" / "env"


def checkpoints(group):
    return [f"results/{ARMS[group]}_s{k}/checkpoint.pt" for k in range(5)]


def conditions():
    """The grid's conditions {floor: (params, floor export, layout, (on duty, bays))} (no scenarios import)."""
    spec = importlib.util.spec_from_file_location("grid_task", R / "rq3-size-grid" / "task.py")
    gt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gt)
    return gt.rt, {f: gt.rt.CONDITIONS[f] for f in FLOORS}


def alpha_only(evaluate, force=None):
    """Make evaluate.build_policies load checkpoints as AlphaOnlyPolicy: heads 0 / 1 fixed to MDD / PTECT, head 2 the
    network's argmax (or @p force, an offset index, for the dry-run check). Policy names get '~alphaonly'."""
    job, machine = evaluate.JOB_HEAD_RULES.index("MDD"), evaluate.MACHINE_HEAD_RULES.index("PTECT")

    class AlphaOnlyPolicy(evaluate.CheckpointPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.name += "~alphaonly" + ("" if force is None else f"~force{force}")

        def __call__(self, obs):
            action = super().__call__(obs)
            return job, machine, int(action[2] if force is None else force)

    evaluate.CheckpointPolicy = AlphaOnlyPolicy


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("floor", choices=FLOORS)
    ap.add_argument("group", choices=GROUPS)
    ap.add_argument("seeds")
    ap.add_argument("--checkpoints", default=None, help="comma list replacing the arm's checkpoints (dry runs)")
    ap.add_argument("--rules", default=None, help="replaces the group's --pdr list (dry runs)")
    ap.add_argument("--force-offset", type=int, default=None, help="ao groups: play this offset index (dry-run check)")
    ap.add_argument("--out-root", default=str(HERE / "out"), help="parent of <floor>/<group>_s<seeds>/ (dry runs)")
    a = ap.parse_args(argv)
    dry = a.checkpoints or a.rules or a.force_offset is not None
    if a.seeds not in CHUNKS[a.floor] and not dry:
        ap.error(f"{a.floor} runs in chunks {CHUNKS[a.floor]}")
    if a.group.startswith("off-") and a.floor != "m15":
        ap.error("the full arms run on m15 only (the harness check); ../eval-alpha-off-size has them on m50-m200")
    out = Path(a.out_root).resolve() / a.floor / f"{a.group}_s{a.seeds}"
    if (out / "DONE").exists():
        print(f"{a.floor}/{out.name}: done, skipped")
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    os.chdir(REPO)                                       # checkpoint paths are repo-relative
    rt, conds = conditions()
    params, floor, layout, fleet = conds[a.floor]
    code = code_of(a.group)
    ckpts = a.checkpoints.split(",") if a.checkpoints else checkpoints(a.group)
    missing = [c for c in ckpts if not (REPO / c).exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    rules = a.rules or (RULES_M15.get(a.group) if a.floor == "m15" else None) or RULES[a.group]
    ev = ["--twin", str(floor), "--reward-spec", str(code / "config" / "rewards" / "tardiness.json"),
          "--scenario-generator", "randomized", "--random-warmup", "--episode-duration-seconds", "21600",
          "--params", json.dumps(params), "--seeds", a.seeds, "--slot-seconds", "900", "--discount-horizon-s", "5400",
          "--device", "cpu", "--pdr", rules, *[x for c in ckpts for x in ("--checkpoint", c)], "--out", str(out),
          "--decision-log"]
    if layout:
        ev += ["--layout", layout]
    (out / "cond.json").write_text(json.dumps({"floor": a.floor, "group": a.group, "seeds": a.seeds, "params": params,
                                               "floor_export": str(Path(floor).resolve().relative_to(REPO)),
                                               "fleet_on_duty_bays": fleet, "code": str(code.relative_to(REPO)),
                                               "stamp": (code.parent / "STAMP").read_text(), "checkpoints": ckpts,
                                               "alpha_only": a.group.startswith("ao-"),
                                               "force_offset": a.force_offset, "evaluate_args": ev}, indent=1))
    resource.setrlimit(resource.RLIMIT_AS, (AS_BYTES, AS_BYTES))
    sys.path.insert(0, str(code))                        # v7 env first: scenarios, config, evaluate are the arm's
    if fleet is not None:                                # m15 (base) keeps the generator's own fleet
        rt.fleet_wrapper(*fleet)
    import evaluate
    import scenarios
    if not Path(evaluate.__file__).resolve().is_relative_to(code) or \
            not Path(scenarios.__file__).resolve().is_relative_to(code):
        raise SystemExit(f"imported the wrong code: {evaluate.__file__}, {scenarios.__file__}")
    if a.group.startswith("ao-"):
        alpha_only(evaluate, a.force_offset)
    t0 = time.time()
    with open(out / "eval.out", "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        evaluate.main(ev)
    seeds = evaluate.parse_seeds(a.seeds)
    gen = scenarios.REGISTRY["randomized"](episode_duration_seconds=21600.0, random_warmup=True,
                                           params_overrides=params)
    (out / "blocks.json").write_text(json.dumps({s: gen(s)["_meta"]["blocks"] for s in seeds}))
    with open(out / "episodes.csv") as f:
        rows = list(csv.DictReader(f))
    want = (len(evaluate.resolve_pdr_names(rules)) + len(ckpts)) * len(seeds)
    if len(rows) != want:
        raise SystemExit(f"{a.floor}/{out.name}: {len(rows)} episodes, expected {want} (see {out}/eval.out)")
    (out / "DONE").write_text(f"{time.time() - t0:.0f}\n")
    print(f"{a.floor}/{out.name}: {len(rows)} episodes in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
