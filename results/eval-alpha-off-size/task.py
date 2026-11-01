"""
@file task.py
@brief eval-alpha-off-size (2026-10-26, user: "evaluation of train-alpha-off-b2 and train-alpha-off-het on the size grid
       (50/100/200 machines), with pre-declared criteria"): one task of the evaluation of the law + offset arms (v7 code:
       alpha = min(4, max(0, 3 - 6 sigma) x offset)) on the floor-size grid's 50 / 100 / 200-machine floors, twin, B2
       test seeds 0-39. The floors are homogeneous (sigma 0), so the law is alpha 3 = MDD-PTECT everywhere and only the
       head's offsets can move alpha: x0.5 / x0.25 give alpha 1.5 / 0.75, x1.5 / x2 give 4 (the cap).
Floors (../rq3-size-grid conditions, same per-machine scaling as ../rq3-size-v5): M bays, round(0.52 M) AGVs on duty
(short blocks half), B2 load x 0.9, arrival cap off, c ~ U[1.75, 2.5]:
  m50   26 (13) AGVs, ../agv-ratio-scale/floors/m50      one task per group (40 seeds)
  m100  52 (26) AGVs, ../agv-ratio-scale/floors/m100     10-seed chunks
  m200  104 (52) AGVs, ../dev-size-floors/floors/m200    5-seed chunks (one m200 episode takes 5-13 min)
Groups (each also runs MDD-PTECT, the run-to-run check against the grid's pdr group, bit for bit):
  off-b2   train-alpha-off-b2 s0-s4, decision log
  off-het  train-alpha-off-het s0-s4, decision log
  alpha    MDD-PTECT at alpha 1.5 and 0.75 (the fixed alphas the offsets reach at sigma 0; v7 "@<alpha>" = absolute)
The B2 window / slot / reward arguments are ../rq3-gen-v5/run_task.py's (tardiness spec, random warm-up, 21,600 s
window, 900 s slots, 5,400 s discount horizon), and so is the fleet wrapper (agvCount = bays, full-fleet blocks get
the on-duty fleet). evaluate.py is the arm's own v7 snapshot (../train-alpha-off-*/code/env, identical for both arms),
run in this process with an address-space cap. Each task also writes blocks.json (each seed's regime blocks, for the
per-regime offset analysis). Output: out/<floor>/<group>_s<a>-<b>/ (episodes.csv, decisions.csv, eval.out, cond.json,
blocks.json, DONE). A task whose DONE exists is skipped; an unfinished one is wiped and rerun, so relaunching resumes.
@par Usage (from the repo root; tasks.txt lists the 39 tasks, launch.sh runs them)
@code{.sh}
.venv/bin/python results/eval-alpha-off-size/task.py m200 off-b2 0-4
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
FLOORS = ("m50", "m100", "m200")
CHUNKS = {"m50": ["0-39"], "m100": [f"{a}-{a + 9}" for a in range(0, 40, 10)],
          "m200": [f"{a}-{a + 4}" for a in range(0, 40, 5)]}
ARMS = {"off-b2": "train-alpha-off-b2", "off-het": "train-alpha-off-het"}
GROUPS = ("off-b2", "off-het", "alpha")
ALPHA_RULES = "MDD-PTECT,MDD-PTECT@1.5,MDD-PTECT@0.75"
## @brief Address-space cap of a task (as ../eval-alpha-offset; an m200 episode peaks near 0.74 GB resident).
AS_BYTES = 12 << 30


def code_of(group):
    return R / ARMS.get(group, ARMS["off-b2"]) / "code" / "env"


def checkpoints(group):
    return [f"results/{ARMS[group]}_s{k}/checkpoint.pt" for k in range(5)] if group in ARMS else []


def conditions():
    """The grid's conditions {floor: (params, floor export, layout, (on duty, bays))} (no scenarios import)."""
    spec = importlib.util.spec_from_file_location("grid_task", R / "rq3-size-grid" / "task.py")
    gt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gt)
    return gt.rt, {f: gt.rt.CONDITIONS[f] for f in FLOORS}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("floor", choices=FLOORS)
    ap.add_argument("group", choices=GROUPS)
    ap.add_argument("seeds")
    ap.add_argument("--checkpoints", default=None, help="comma list replacing the arm's checkpoints (dry runs)")
    ap.add_argument("--rules", default=None, help="replaces the group's --pdr list (dry runs)")
    ap.add_argument("--out-root", default=str(HERE / "out"), help="parent of <floor>/<group>_s<seeds>/ (dry runs)")
    a = ap.parse_args(argv)
    if a.seeds not in CHUNKS[a.floor] and not (a.checkpoints or a.rules):
        ap.error(f"{a.floor} runs in chunks {CHUNKS[a.floor]}")
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
    rules = a.rules or ("MDD-PTECT" if a.group in ARMS else ALPHA_RULES)
    ev = ["--twin", str(floor), "--reward-spec", str(code / "config" / "rewards" / "tardiness.json"),
          "--scenario-generator", "randomized", "--random-warmup", "--episode-duration-seconds", "21600",
          "--params", json.dumps(params), "--seeds", a.seeds, "--slot-seconds", "900", "--discount-horizon-s", "5400",
          "--device", "cpu", "--pdr", rules, *[x for c in ckpts for x in ("--checkpoint", c)], "--out", str(out)]
    if ckpts:
        ev.append("--decision-log")
    if layout:
        ev += ["--layout", layout]
    (out / "cond.json").write_text(json.dumps({"floor": a.floor, "group": a.group, "seeds": a.seeds, "params": params,
                                               "floor_export": str(Path(floor).resolve().relative_to(REPO)),
                                               "fleet_on_duty_bays": fleet, "code": str(code.relative_to(REPO)),
                                               "stamp": (code.parent / "STAMP").read_text(), "checkpoints": ckpts,
                                               "evaluate_args": ev}, indent=1))
    resource.setrlimit(resource.RLIMIT_AS, (AS_BYTES, AS_BYTES))
    sys.path.insert(0, str(code))                        # v7 env first: scenarios, config, evaluate are the arm's
    rt.fleet_wrapper(*fleet)
    import evaluate
    import scenarios
    if not Path(evaluate.__file__).resolve().is_relative_to(code) or \
            not Path(scenarios.__file__).resolve().is_relative_to(code):
        raise SystemExit(f"imported the wrong code: {evaluate.__file__}, {scenarios.__file__}")
    t0 = time.time()
    with open(out / "eval.out", "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        evaluate.main(ev)
    seeds = evaluate.parse_seeds(a.seeds)
    gen = scenarios.REGISTRY["randomized"](episode_duration_seconds=21600.0, random_warmup=True,
                                           params_overrides=params)
    (out / "blocks.json").write_text(json.dumps({s: gen(s)["_meta"]["blocks"] for s in seeds}))
    with open(out / "episodes.csv") as f:
        rows = list(csv.DictReader(f))
    want = (len(rules.split(",")) + len(ckpts)) * len(seeds)
    if len(rows) != want:
        raise SystemExit(f"{a.floor}/{out.name}: {len(rows)} episodes, expected {want} (see {out}/eval.out)")
    (out / "DONE").write_text(f"{time.time() - t0:.0f}\n")
    print(f"{a.floor}/{out.name}: {len(rows)} episodes in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
