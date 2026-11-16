"""
@file task.py
@brief eval-alpha-online (2026-10-29, user: "design the online alpha experiment (h from the on-duty fleet per regime,
       plus the spread law) with pre-declared criteria, and set it up in the lab"): stage 2, one task of the evaluation
       of the carried-forward arm train-alpha-off-het (v7) with the online law of ../rq3-alpha-online
       (law_online.json "online") and the size gate of ../eval-alpha-transport-size, twin, test seeds 0-39.
The online law: alpha = min(4, max(0, 3 - 6 sigma) x h(tau / a_t) x offset), a_t = AGVs on duty per machine at the
decision, h the selected hinge(s). This process wraps env_wrappers.twin_env.TwinSchedulingEnv.step so that before
every decision self.alpha_law is the law at the on-duty count of that moment (onlinelaw.py; offset_alpha then applies
the offset head). evaluate.alpha_law, which only labels the episode rows, gives the full-fleet alpha. The frozen v7
code is not edited, and the policy's inputs do not contain the law.
Group og (the only group): the arm's 5 checkpoints, MDD-PTECT (two-element action: alpha 3; the run-to-run check
against the reference group's MDD-PTECT) and MDD-PTECT@x1 (the online law alone, offset x1; it must equal the
simulator's online-law run, ../rq3-alpha-online/run.py law). Above 50 machines the job and machine heads are fixed to
MDD-PTECT and only the offset head chooses (../eval-alpha-only-size's AlphaOnlyPolicy).
Cells and harnesses:
  m15_b2, m15_s1     ../eval-alpha-offset's: ../rq3-gen-v5 B2 arguments through the arm's hetero.py --spread s,s
                     --entry evaluate, here in this process (so the wrapped step applies); reference: its off-het group
  m50_r052 ... m200_r052   ../eval-alpha-transport-size's (the size grid's conditions, fleet wrapper, 12 GB cap);
                     reference: its lg group
A cell runs only where the online law's pair (alpha_full, alpha_short) differs from the static law's (law_online.json);
elsewhere the two policies are the same by construction. Output: out/<cell>/og_s<a>-<b>/ (episodes.csv, decisions.csv,
eval.out, cond.json, blocks.json, DONE). A task whose DONE exists is skipped; an unfinished one is wiped and rerun.
@par Usage (from the repo root; launch.sh runs task.py list)
@code{.sh}
.venv/bin/python results/eval-alpha-online/task.py m100_r052 0-9
# dry run with a stand-in law and one checkpoint:
.venv/bin/python results/eval-alpha-online/task.py m15_b2 0-1 --law <law_online.json> \
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
ARM = "train-alpha-off-het"
CODE = R / ARM / "code"
LAW = R / "rq3-alpha-online" / "law_online.json"
CHUNKS10 = [f"{a}-{a + 9}" for a in range(0, 40, 10)]
CHUNKS = {"m15_b2": ["0-39"], "m15_s1": ["0-39"], "m50_r052": CHUNKS10, "m75_r052": CHUNKS10,
          "m100_r052": CHUNKS10, "m100_r035": CHUNKS10, "m100_r080": CHUNKS10,
          "m200_r052": [f"{a}-{a + 4}" for a in range(0, 40, 5)]}
ORDER = ["m200_r052", "m100_r035", "m100_r052", "m100_r080", "m75_r052", "m50_r052", "m15_s1", "m15_b2"]
SPREAD = {"m15_b2": 0.0, "m15_s1": 1.0}
RULES = "MDD-PTECT,MDD-PTECT@x1"
GATE_MACHINES = 50
RATIOS = {"r035": 0.35, "r052": 0.52, "r080": 0.80}
## @brief Address-space cap of a task (as ../eval-alpha-transport-size; an m200 episode peaks near 0.74 GB resident).
AS_BYTES = 12 << 30


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OL = _module("onlinelaw", HERE / "onlinelaw.py")


def checkpoints():
    return [f"results/{ARM}_s{k}/checkpoint.pt" for k in range(5)]


def changed(spec, cell):
    """True where the online law plays another pair than the static law on the cell (cell-mean sigma)."""
    if cell not in spec["cells"]:
        return False
    pairs = spec["cells"][cell]["pair"]
    on, st = pairs[spec["selected"]], pairs["S0"]
    return any(abs(a - b) > 1e-3 for a, b in zip(on, st))


def grid_condition(cell):
    """(run_task module, (params, floor export, layout, (on duty, bays))) of a size-grid cell
    (../eval-alpha-transport-size task.condition)."""
    gt = _module("grid_task", R / "rq3-size-grid" / "task.py")
    m, fleet = cell.split("_")
    params, floor, layout, grid_fleet = gt.rt.CONDITIONS[m]
    size = int(m[1:])
    on = round(RATIOS[fleet] * size)
    assert layout is None and grid_fleet[1] == size, (cell, layout, grid_fleet)
    if fleet == "r052":
        assert grid_fleet[0] == on and params == gt.st.size_params(size // 5, on, round(on / 2)), cell
    return gt.rt, (gt.st.size_params(size // 5, on, round(on / 2)), floor, layout, (on, size))


def wrap_step(te, law, cs, log):
    """TwinSchedulingEnv.step sets self.alpha_law from the online law at the decision's on-duty count. @p log collects
    (twin id, sim time, on duty, alpha) whenever the law's alpha changes (a compact trace for the checks)."""
    orig = te.TwinSchedulingEnv.step
    state = {"tw": None, "f": None, "last": None}

    def step(self, action):
        tw = self._twin
        if state["tw"] is not tw:
            state.update(tw=tw, f=OL.law_alpha_fn(law, cs, float(self.speed_sigma or 0.0)), last=None)
        k = OL.on_duty(tw)
        self.alpha_law = state["f"](k, OL.full_fleet(tw))
        if self.alpha_law != state["last"]:
            log.append((id(tw), round(float(tw.now), 3), int(k), self.alpha_law))
            state["last"] = self.alpha_law
        return orig(self, action)

    te.TwinSchedulingEnv.step = step


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", choices=list(CHUNKS))
    ap.add_argument("seeds")
    ap.add_argument("--checkpoints", default=None, help="comma list replacing the arm's checkpoints (dry runs)")
    ap.add_argument("--law", default=str(LAW), help="law_online.json (dry runs may pass a stand-in)")
    ap.add_argument("--out-root", default=str(HERE / "out"), help="parent of <cell>/og_s<seeds>/ (dry runs)")
    a = ap.parse_args(argv)
    dry = a.checkpoints or a.law != str(LAW)
    if a.seeds not in CHUNKS[a.cell] and not dry:
        ap.error(f"{a.cell} runs in chunks {CHUNKS[a.cell]}")
    spec = json.loads(Path(a.law).read_text())
    if spec["G1"] != "PASS" and not dry:
        raise SystemExit(f"{a.law}: gate G1 {spec['G1']}, stage 2 does not run (README)")
    out = Path(a.out_root).resolve() / a.cell / f"og_s{a.seeds}"
    if (out / "DONE").exists():
        print(f"{a.cell}/{out.name}: done, skipped")
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    os.chdir(REPO)                                       # checkpoint paths are repo-relative
    ckpts = a.checkpoints.split(",") if a.checkpoints else checkpoints()
    missing = [c for c in ckpts if not (REPO / c).exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    law, cs = spec["laws"]["online"], spec["cells"][a.cell]
    env = CODE / "env"
    small = a.cell in SPREAD
    if small:
        sys.path.insert(0, str(R / "rq3-gen-v5"))
        import run_task as rt
        params, floor = rt.CONDITIONS["base"][0], rt.CONDITIONS["base"][1]
        layout, fleet = None, None
    else:
        rt, (params, floor, layout, fleet) = grid_condition(a.cell)
    size = int(a.cell.split("_")[0][1:])
    gated = size > GATE_MACHINES
    ev = ["--twin", str(floor), "--reward-spec", str(env / "config" / "rewards" / "tardiness.json"),
          "--scenario-generator", "randomized", "--random-warmup", "--episode-duration-seconds", "21600",
          "--params", json.dumps(params), "--seeds", a.seeds, "--slot-seconds", "900", "--discount-horizon-s", "5400",
          "--device", "cpu", "--pdr", RULES, *[x for c in ckpts for x in ("--checkpoint", c)], "--out", str(out),
          "--decision-log"]
    if layout:
        ev += ["--layout", layout]
    (out / "cond.json").write_text(json.dumps({
        "cell": a.cell, "seeds": a.seeds, "params": params, "spread": SPREAD.get(a.cell),
        "floor_export": str(Path(floor).resolve().relative_to(REPO)), "fleet_on_duty_bays": fleet,
        "code": str(env.relative_to(REPO)), "stamp": (CODE / "STAMP").read_text(), "checkpoints": ckpts,
        "law_file": str(Path(a.law).resolve()), "law": law, "selected": spec["selected"], "cell_stats": cs,
        "gated": gated, "evaluate_args": ev}, indent=1))
    resource.setrlimit(resource.RLIMIT_AS, (AS_BYTES, AS_BYTES))
    sys.path.insert(0, str(env))                         # v7 env first: scenarios, config, evaluate are the arm's
    if fleet is not None:
        rt.fleet_wrapper(*fleet)
    import evaluate
    import scenarios
    import env_wrappers.twin_env as te
    for mod in (evaluate, scenarios, te):
        if not Path(mod.__file__).resolve().is_relative_to(env):
            raise SystemExit(f"imported the wrong code: {mod.__file__}")
    seeds = evaluate.parse_seeds(a.seeds)
    gen = scenarios.REGISTRY["randomized"](episode_duration_seconds=21600.0, random_warmup=True,
                                           params_overrides=params)       # before hetero.py wraps the registry
    (out / "blocks.json").write_text(json.dumps({s: gen(s)["_meta"]["blocks"] for s in seeds}))
    trace = []                          # filled in this process only (evaluate's single env; forked workers drop it)
    wrap_step(te, law, cs, trace)
    evaluate.alpha_law = lambda sigma: OL.full_alpha(law, cs, sigma)
    if gated:
        _module("alpha_only_task", R / "eval-alpha-only-size" / "task.py").alpha_only(evaluate)
    t0 = time.time()
    with open(out / "eval.out", "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        if small:
            hetero = _module("hetero_v7", CODE / "hetero.py")
            s = SPREAD[a.cell]
            sys.argv = [str(CODE / "hetero.py"), "--spread", f"{s:g},{s:g}", "--entry", "evaluate", "--", *ev]
            hetero.main()
        else:
            evaluate.main(ev)
    (out / "alpha_trace.json").write_text(json.dumps(trace))
    with open(out / "episodes.csv") as f:
        rows = list(csv.DictReader(f))
    want = (len(RULES.split(",")) + len(ckpts)) * len(seeds)
    if len(rows) != want:
        raise SystemExit(f"{a.cell}/{out.name}: {len(rows)} episodes, expected {want} (see {out}/eval.out)")
    (out / "DONE").write_text(f"{time.time() - t0:.0f}\n")
    print(f"{a.cell}/{out.name}: {len(rows)} episodes in {time.time() - t0:.0f} s")


def tasks(law_file=LAW):
    """Longest first; only the cells where the online law changes the pair."""
    spec = json.loads(Path(law_file).read_text())
    return [(c, ch) for c in ORDER if changed(spec, c) for ch in CHUNKS[c]]


if __name__ == "__main__":
    if sys.argv[1:2] == ["list"]:
        for t in tasks(*sys.argv[2:3]):
            print(*t)
    else:
        main()
