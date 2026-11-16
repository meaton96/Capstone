"""
@file task.py
@brief eval-alpha-transport-size (2026-10-28, user: "design the size / transport term in the law, with the pair heads
       restrained above ~50 machines, with pre-declared criteria"): stage 2, one task of the evaluation of the
       law + offset arm train-alpha-off-het (v7) with the transport term fitted in ../rq3-alpha-transport (law.json)
       and a size gate on the pair heads, twin, B2 test seeds 0-39.
The law (law.json): alpha = min(4, max(0, 3 - 6 sigma) x h(x) x offset), h(x) = clip((x1 - x) / (x1 - x0), 0, 1), x the
selected floor statistic (a cell constant from its fitting instances). The floors are homogeneous, so the law plays
3 h(x) x offset. This process replaces env_wrappers.twin_env.alpha_law (and evaluate.alpha_law, which only labels the
episode rows) with v7's law times the cell's h, rounded to 3 decimals as law.json's alpha; the frozen code is not
edited and the policy's inputs do not include the law.
Cells (homogeneous; r052 = the size grid's conditions, ../rq3-size-grid; r035 / r080 = the same floor with round(r M)
AGVs on duty, short blocks half, as ../rq3-size-headroom):
  m50_r052   26 (13) AGVs      one task per group (40 seeds)
  m75_r052   39 (20)           10-seed chunks
  m100_r052  52 (26)           10-seed chunks
  m100_r035  35 (18)           10-seed chunks
  m100_r080  80 (40)           10-seed chunks
  m200_r052  104 (52)          5-seed chunks
Groups (train-alpha-off-het s0-s4, the carried-forward arm):
  lg  the law with the transport term and the gate: above 50 machines the job and machine heads are fixed to
      MDD-PTECT and only the offset head chooses (../eval-alpha-only-size's AlphaOnlyPolicy); at 50 machines or fewer
      the full policy. Also runs MDD-PTECT (the run-to-run check against the simulator's alpha-3 runs).
  lf  the law with the transport term, full policy (no gate); cells above 50 machines.
  v7  the v7 law (no transport term), full policy, on the cells ../eval-alpha-off-size does not cover (m75_r052,
      m100_r035, m100_r080).
The fixed-alpha rules (the alpha grid, MDD-PTECT at the law's alpha) are the simulator's (../rq3-alpha-size,
../rq3-alpha-transport), which equals this harness (checked 10-28). The window / slot / reward arguments, the fleet
wrapper and the address-space cap are ../eval-alpha-off-size's; evaluate.py is the arm's own v7 snapshot. Output:
out/<cell>/<group>_s<a>-<b>/ (episodes.csv, decisions.csv, eval.out, cond.json, blocks.json, DONE). A task whose
DONE exists is skipped; an unfinished one is wiped and rerun, so relaunching resumes.
@par Usage (from the repo root; tasks.txt lists the tasks, launch.sh runs them)
@code{.sh}
.venv/bin/python results/eval-alpha-transport-size/task.py m200_r052 lg 0-4
# dry run: offset head forced to x1 under lg must equal MDD-PTECT at the law's alpha
.venv/bin/python results/eval-alpha-transport-size/task.py m100_r035 lg 0-1 --force-offset 0 \
    --checkpoints results/train-alpha-off-het_s0/checkpoint.pt --rules MDD-PTECT --law <law.json> --out-root <scratch>
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
ARM = "train-alpha-off-het"
CHUNKS10 = [f"{a}-{a + 9}" for a in range(0, 40, 10)]
CHUNKS = {"m50_r052": ["0-39"], "m75_r052": CHUNKS10, "m100_r052": CHUNKS10, "m100_r035": CHUNKS10,
          "m100_r080": CHUNKS10, "m200_r052": [f"{a}-{a + 4}" for a in range(0, 40, 5)]}
GROUPS = {"lg": list(CHUNKS), "lf": [c for c in CHUNKS if c != "m50_r052"],
          "v7": ["m75_r052", "m100_r035", "m100_r080"]}
RULES = {"lg": "MDD-PTECT", "lf": "none", "v7": "none"}
GATE_MACHINES = 50
RATIOS = {"r035": 0.35, "r052": 0.52, "r080": 0.80}
LAW = R / "rq3-alpha-transport" / "law.json"
## @brief Address-space cap of a task (as ../eval-alpha-off-size; an m200 episode peaks near 0.74 GB resident).
AS_BYTES = 12 << 30


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def code():
    return R / ARM / "code" / "env"


def checkpoints():
    return [f"results/{ARM}_s{k}/checkpoint.pt" for k in range(5)]


def condition(cell):
    """(run_task module, (params, floor export, layout, (on duty, bays))) of a cell, no scenarios import."""
    gt = _module("grid_task", R / "rq3-size-grid" / "task.py")
    m, fleet = cell.split("_")
    params, floor, layout, grid_fleet = gt.rt.CONDITIONS[m]
    size = int(m[1:])
    on = round(RATIOS[fleet] * size)
    assert layout is None and grid_fleet[1] == size, (cell, layout, grid_fleet)
    if fleet == "r052":
        assert grid_fleet[0] == on and params == gt.st.size_params(size // 5, on, round(on / 2)), cell
    return gt.rt, (gt.st.size_params(size // 5, on, round(on / 2)), floor, layout, (on, size))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", choices=list(CHUNKS))
    ap.add_argument("group", choices=list(GROUPS))
    ap.add_argument("seeds")
    ap.add_argument("--checkpoints", default=None, help="comma list replacing the arm's checkpoints (dry runs)")
    ap.add_argument("--rules", default=None, help="replaces the group's --pdr list (dry runs)")
    ap.add_argument("--force-offset", type=int, default=None, help="gated lg: play this offset index (dry-run check)")
    ap.add_argument("--law", default=str(LAW), help="law.json (dry runs may pass a stand-in)")
    ap.add_argument("--out-root", default=str(HERE / "out"), help="parent of <cell>/<group>_s<seeds>/ (dry runs)")
    a = ap.parse_args(argv)
    dry = a.checkpoints or a.rules or a.force_offset is not None or a.law != str(LAW)
    if a.seeds not in CHUNKS[a.cell] and not dry:
        ap.error(f"{a.cell} runs in chunks {CHUNKS[a.cell]}")
    if a.cell not in GROUPS[a.group]:
        ap.error(f"group {a.group} runs on {GROUPS[a.group]}")
    law = json.loads(Path(a.law).read_text())
    if law["F1"] != "PASS" and not dry:
        raise SystemExit(f"{a.law}: gate F1 {law['F1']}, stage 2 does not run (README)")
    out = Path(a.out_root).resolve() / a.cell / f"{a.group}_s{a.seeds}"
    if (out / "DONE").exists():
        print(f"{a.cell}/{out.name}: done, skipped")
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    os.chdir(REPO)                                       # checkpoint paths are repo-relative
    rt, (params, floor, layout, fleet) = condition(a.cell)
    env = code()
    ckpts = a.checkpoints.split(",") if a.checkpoints else checkpoints()
    missing = [c for c in ckpts if not (REPO / c).exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    rules = a.rules or RULES[a.group]
    size = fleet[1]
    gated = a.group == "lg" and size > GATE_MACHINES
    h = None if a.group == "v7" else float(law["cells"][a.cell]["h"])
    ev = ["--twin", str(floor), "--reward-spec", str(env / "config" / "rewards" / "tardiness.json"),
          "--scenario-generator", "randomized", "--random-warmup", "--episode-duration-seconds", "21600",
          "--params", json.dumps(params), "--seeds", a.seeds, "--slot-seconds", "900", "--discount-horizon-s", "5400",
          "--device", "cpu", "--pdr", rules, *[x for c in ckpts for x in ("--checkpoint", c)], "--out", str(out),
          "--decision-log"]
    if layout:
        ev += ["--layout", layout]
    (out / "cond.json").write_text(json.dumps({
        "cell": a.cell, "group": a.group, "seeds": a.seeds, "params": params,
        "floor_export": str(Path(floor).resolve().relative_to(REPO)), "fleet_on_duty_bays": fleet,
        "code": str(env.relative_to(REPO)), "stamp": (env.parent / "STAMP").read_text(), "checkpoints": ckpts,
        "law": None if h is None else {"file": str(Path(a.law).resolve()), "law": law["law"], "variable": law["variable"],
                                       "x0": law["x0"], "x1": law["x1"], **law["cells"][a.cell]},
        "gated": gated, "force_offset": a.force_offset, "evaluate_args": ev}, indent=1))
    resource.setrlimit(resource.RLIMIT_AS, (AS_BYTES, AS_BYTES))
    sys.path.insert(0, str(env))                         # v7 env first: scenarios, config, evaluate are the arm's
    rt.fleet_wrapper(*fleet)
    import evaluate
    import scenarios
    import env_wrappers.twin_env as te
    for mod in (evaluate, scenarios, te):
        if not Path(mod.__file__).resolve().is_relative_to(env):
            raise SystemExit(f"imported the wrong code: {mod.__file__}")
    if h is not None:
        v7_law = te.alpha_law

        def transport_law(sigma):
            return round(v7_law(sigma) * h, 3)

        te.alpha_law = evaluate.alpha_law = transport_law
    if gated:
        _module("alpha_only_task", R / "eval-alpha-only-size" / "task.py").alpha_only(evaluate, a.force_offset)
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
        raise SystemExit(f"{a.cell}/{out.name}: {len(rows)} episodes, expected {want} (see {out}/eval.out)")
    (out / "DONE").write_text(f"{time.time() - t0:.0f}\n")
    print(f"{a.cell}/{out.name}: {len(rows)} episodes in {time.time() - t0:.0f} s")


def tasks():
    """Longest first: m200, then the 100-machine cells (r035 heaviest), m75, m50."""
    order = ["m200_r052", "m100_r035", "m100_r052", "m100_r080", "m75_r052", "m50_r052"]
    return [(c, g, ch) for c in order for g in GROUPS if c in GROUPS[g] for ch in CHUNKS[c]]


if __name__ == "__main__":
    if sys.argv[1:2] == ["list"]:
        for t in tasks():
            print(*t)
    else:
        main()
