"""
@file task.py
@brief eval-alpha-unity (2026-10-30, user: "Set up plan §3.2: the Unity check of the static law + gate through the
       Python rule mode on B2 and one large floor, with pre-declared criteria, in the lab"): one task of the Unity
       evaluation of paper B's α policy, the carried-forward arm train-alpha-off-het (v7: law x offset head) with the
       static transport law and the size gate, on the twin runs' own instances (test seeds 0-39).
How v7 runs in Unity (no C# change; code/STAMP lists the files taken from the main env):
  - The player's agent has two discrete branches (job head, machine head); v7 has three (+ the offset head). This
    process sets env_wrappers.unity_env.ACTION_BRANCHES to the first two (the branch check and the action tuple) and
    pads Unity's action mask with the offset head's part as the twin builds it (all five offsets where the machine head
    matters, else offset 0 only), so the network sees v7's 10-entry mask.
  - Every three-head action (and every "@x<offset>" / "@<alpha>" fixed rule) is played in Python rule mode
    (player -pyrules, env/rules_api): the job and machine heads name the built-in Python ports, and AlphaPair sets the
    view's PECT / PTECT alpha to law x offset (capped at 4, v7's offset_alpha) before it picks. A two-head action
    (a bare "MDD-PTECT") goes to the player as before and plays the episode's alpha 3 in C#.
  - The law is the static transport law of ../rq3-alpha-transport (law.json, as ../eval-alpha-transport-size):
    alpha_law = round(3 h, 3) on these homogeneous floors (sigma 0 is asserted per instance), a cell constant.
  - Above 50 machines the job and machine heads are fixed to MDD-PTECT and only the offset head chooses
    (../eval-alpha-only-size's AlphaOnlyPolicy, the gate).
Cells (instances, window, slots and reward exactly as the twin runs they are compared with):
  b2          B2 (15 machines, 7 AGVs, ../rq3-gen-v5 "base"), law alpha 3 (h 1): the full policy. Twin reference
              ../eval-alpha-offset/out/0/off-het (hetero spread 0 = the plain B2 instance).
  m100_r052   the size grid's 100-machine floor (layout D, 100 bays, 52 AGVs on duty, short blocks 26; ../rq3-size-grid),
              law alpha round(3 h, 3), the gate. Twin reference ../eval-alpha-transport-size/out/m100_r052/lg_s*.
Policies (one per task; key -> what runs):
  ck0 ... ck4   train-alpha-off-het s0-s4 (gated on m100_r052)
  pdr           MDD-PTECT, C# rule at the episode's alpha 3 (the reference every % is against)
  py3   (b2)    MDD-PTECT@3 in Python rule mode: must equal pdr seed for seed (check K1, the alpha plumbing)
  law   (m100)  MDD-PTECT@x1 in Python rule mode: the static law alone (alpha_law)
Output: out/<cell>/<policy>_s<a>-<b>/ (episodes.csv, decisions.csv, eval.out, Player.log, cond.json, DONE). A task
whose DONE exists is skipped; an unfinished one is wiped and rerun, so relaunching resumes. Each task has its own
player port (worker id from its index in tasks()).
@par Usage (from the repo root; launch.sh runs the task list)
@code{.sh}
.venv/bin/python results/eval-alpha-unity/task.py list
.venv/bin/python results/eval-alpha-unity/task.py b2 ck0 0-9
# dry runs (scratch output, any seeds, forced offset under the gate):
.venv/bin/python results/eval-alpha-unity/task.py m100_r052 ck0 0-0 --force-offset 0 --out-root <scratch>
@endcode
"""
import argparse
import contextlib
import csv
import hashlib
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
CODE = HERE / "code" / "env"
ARM = "train-alpha-off-het"
PLAYER = REPO / "linux_server" / "capstone.x86_64"
## @brief The resume player (09f53cbb); B2 and the size grid have machine failures off, so it equals 5e6e0f87 here.
PLAYER_MD5 = "22891261e82b72d3bb3dc0fbc3764e38"
LAW = R / "rq3-alpha-transport" / "law.json"
GATE_MACHINES = 50
## @brief mlagents' wait for the player (s): long random warm-ups run inside Unity with no exchange (as
##        ../rq4-agvfail-v5-rules/eval_wrap.py; 300 s is the frozen default).
TIMEOUT_WAIT = 3600
## @brief Address-space cap of the Python process (the player is a child and is not capped by it in practice: RLIMIT_AS
##        is inherited, so it is set after the player starts, see main()).
AS_BYTES = 12 << 30
BASE_WORKER_ID = 300
CHUNKS10 = [f"{a}-{a + 9}" for a in range(0, 40, 10)]
## cell -> (machines, seed chunks, policy keys)
CELLS = {
    "b2": (15, CHUNKS10, ["ck0", "ck1", "ck2", "ck3", "ck4", "pdr", "py3"]),
    "m100_r052": (100, [f"{a}-{a}" for a in range(40)], ["ck0", "ck1", "ck2", "ck3", "ck4", "pdr", "law"]),
}
RULE_OF = {"pdr": "MDD-PTECT", "py3": "MDD-PTECT@3", "law": "MDD-PTECT@x1"}


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def condition(cell):
    """(params, (on duty, bays) or None, h) of a cell, as the twin references built them."""
    law = json.loads(LAW.read_text())
    if cell == "b2":
        rt = _module("gen_v5_run_task", R / "rq3-gen-v5" / "run_task.py")
        params, _floor, layout, fleet = rt.CONDITIONS["base"]
        assert layout is None and fleet is None
        return rt, params, None, 1.0
    rt, (params, _floor, layout, fleet) = _module("ats_task", R / "eval-alpha-transport-size" / "task.py").condition(cell)
    assert layout is None
    return rt, params, fleet, float(law["cells"][cell]["h"])


def tasks():
    """Every (cell, policy, seeds), longest first: m100 policies, then B2. The index fixes the task's player port."""
    out = []
    for cell in ("m100_r052", "b2"):
        _, chunks, keys = CELLS[cell]
        out += [(cell, k, ch) for ch in chunks for k in keys]
    return out


def worker_id(cell, key, seeds):
    return BASE_WORKER_ID + tasks().index((cell, key, seeds))


def port_free(port):
    """True if nothing listens on @p port. gRPC binds with SO_REUSEPORT, so two evaluate.py processes on one port both
    start and their players can cross-connect (seen in the 10-30 dry run); this bind test has no reuse flag."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def adapt_unity(ue, cfg, rules_api, law_alpha, record):
    """Map v7's three-head actions onto the two-branch player (module docstring). @p record counts decisions by
    (pair, alpha) for cond.json's check."""
    branches = tuple(cfg.ACTION_BRANCHES[:2])
    ue.ACTION_BRANCHES = branches
    job_n, mach_n, off_n = cfg.ACTION_BRANCHES
    orig_mask = ue.action_mask_from

    def action_mask_from(steps):
        m = orig_mask(steps)
        if len(m) == cfg.ACTION_MASK_LEN:      # no mask from Unity: every action enabled (orig pads to the v7 length)
            return m
        assert len(m) == job_n + mach_n, (len(m), branches)
        machine_matters = m[job_n:job_n + mach_n].sum() > 1
        off = ue.np.ones(off_n, dtype=ue.np.float32) if machine_matters else \
            ue.np.eye(1, off_n, 0, dtype=ue.np.float32)[0]
        return ue.np.concatenate([m, off]).astype(ue.np.float32)

    ue.action_mask_from = action_mask_from

    class AlphaPair(rules_api.RulePair):
        """A built-in job x machine pair that plays PECT / PTECT at a set alpha (the view's episode alpha replaced)."""

        def __init__(self, job, machine, alpha):
            super().__init__(job, machine)
            self.alpha = float(alpha)
            self.name = f"{self.job.name}-{self.machine.name}@{self.alpha:g}"

        def pick(self, ctx):
            if ctx.pt is not None:
                ctx.pt = dict(ctx.pt, alpha=self.alpha)
            key = (self.name, ctx.kind)
            record[key] = record.get(key, 0) + 1
            return super().pick(ctx)

    pairs = {}

    def pair(job, machine, alpha):
        k = (job, machine, round(float(alpha), 6))
        if k not in pairs:
            pairs[k] = AlphaPair(job, machine, alpha)
        return pairs[k]

    orig_step = ue.UnitySchedulingEnv.step

    def step(self, action):
        if isinstance(action, str):
            base, suffix = cfg.split_alpha(cfg.normalize_rule(action))
            if suffix is None:
                return orig_step(self, action)
            alpha = cfg.offset_alpha(law_alpha, suffix[1]) if suffix[0] == "x" else suffix[1]
            job, machine = base.split("-")
            return orig_step(self, pair(job, machine, alpha))
        if getattr(action, "is_py_rule", False):
            return orig_step(self, action)
        heads = [int(a) for a in ue.np.asarray(action).reshape(-1)]
        if len(heads) == 2:                    # v5's two-element action: C#, the episode's alpha
            return orig_step(self, tuple(heads))
        if len(heads) != 3:
            raise ValueError(f"action {heads}: expected 2 or 3 head indices")
        alpha = cfg.offset_alpha(law_alpha, cfg.ALPHA_OFFSET_VALUES[heads[2]])
        return orig_step(self, pair(cfg.JOB_HEAD_RULES[heads[0]], cfg.MACHINE_HEAD_RULES[heads[1]], alpha))

    ue.UnitySchedulingEnv.step = step
    orig_init = ue.UnitySchedulingEnv.__init__

    def init(self, *a, **kw):
        kw["py_rules"] = True                  # -pyrules: every decision's view is sent (C# picks are unchanged by it)
        orig_init(self, *a, **kw)

    ue.UnitySchedulingEnv.__init__ = init


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", choices=list(CELLS))
    ap.add_argument("policy")
    ap.add_argument("seeds")
    ap.add_argument("--force-offset", type=int, default=None, help="gated checkpoints: play this offset index (dry run)")
    ap.add_argument("--out-root", default=str(HERE / "out"), help="parent of <cell>/<policy>_s<seeds>/ (dry runs)")
    a = ap.parse_args(argv)
    size, chunks, keys = CELLS[a.cell]
    dry = a.force_offset is not None or a.out_root != str(HERE / "out")
    if a.policy not in keys:
        ap.error(f"{a.cell} runs policies {keys}")
    if a.seeds not in chunks and not dry:
        ap.error(f"{a.cell} runs in chunks {chunks}")
    out = Path(a.out_root).resolve() / a.cell / f"{a.policy}_s{a.seeds}"
    if (out / "DONE").exists():
        print(f"{a.cell}/{out.name}: done, skipped")
        return
    md5 = hashlib.md5((PLAYER.parent / "capstone_Data" / "Managed" / "Simulation.dll").read_bytes()).hexdigest()
    if md5 != PLAYER_MD5:
        raise SystemExit(f"linux_server/ Simulation.dll is {md5}, not {PLAYER_MD5} (the player this check was set up on)")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    os.chdir(REPO)                                       # checkpoint paths are repo-relative
    rt, params, fleet, h = condition(a.cell)
    gated = size > GATE_MACHINES
    ckpt = None
    if a.policy.startswith("ck"):
        ckpt = f"results/{ARM}_s{int(a.policy[2:])}/checkpoint.pt"
        if not (REPO / ckpt).exists():
            raise SystemExit(f"missing checkpoint {ckpt}")
    wid = BASE_WORKER_ID + 1000 + os.getpid() % 3000 if dry else worker_id(a.cell, a.policy, a.seeds)
    if not port_free(5005 + wid):                        # mlagents' default base port 5005
        raise SystemExit(f"port {5005 + wid} (worker id {wid}) is in use: another evaluation holds it")
    ev =["--unity-path", str(PLAYER), "--no-graphics", "--base-worker-id", str(wid),
          "--reward-spec", str(CODE / "config" / "rewards" / "tardiness.json"),
          "--scenario-generator", "randomized", "--random-warmup", "--episode-duration-seconds", "21600",
          "--params", json.dumps(params), "--seeds", a.seeds, "--slot-seconds", "900", "--discount-horizon-s", "5400",
          "--device", "cpu", "--pdr", RULE_OF.get(a.policy, "none"), "--out", str(out), "--decision-log"]
    if ckpt:
        ev += ["--checkpoint", ckpt]
    sys.path.insert(0, str(CODE))                        # this snapshot first: config, evaluate, rules_api are its own
    if fleet is not None:
        rt.fleet_wrapper(*fleet)
    import config as cfg
    import evaluate
    import rules_api
    import scenarios
    import env_wrappers.unity_env as ue
    for mod in (cfg, evaluate, rules_api, scenarios, ue):
        if not Path(mod.__file__).resolve().is_relative_to(CODE):
            raise SystemExit(f"imported the wrong code: {mod.__file__}")
    rules_api.load_rules("builtin")
    law_alpha = round(cfg.alpha_law(0.0) * h, 3)
    seeds = evaluate.parse_seeds(a.seeds)
    gen = scenarios.REGISTRY["randomized"](episode_duration_seconds=21600.0, random_warmup=True, params_overrides=params)
    planned = {s: gen(s) for s in seeds}
    for s, sc in planned.items():
        sigma = cfg.scenario_speed_sigma(sc)
        if sigma != 0.0:
            raise SystemExit(f"seed {s}: speed spread sigma {sigma}, the cell's law assumes a homogeneous floor")
    (out / "blocks.json").write_text(json.dumps({s: sc["_meta"]["blocks"] for s, sc in planned.items()}))
    record = {}
    adapt_unity(ue, cfg, rules_api, law_alpha, record)
    evaluate.alpha_law = lambda sigma: law_alpha         # labels the episode rows (alpha_law column)
    if gated and ckpt:
        _module("alpha_only_task", R / "eval-alpha-only-size" / "task.py").alpha_only(evaluate, a.force_offset)
    elif a.force_offset is not None:
        raise SystemExit("--force-offset applies to gated checkpoints only")
    import mlagents_envs.environment as me
    orig_me_init = me.UnityEnvironment.__init__

    def me_init(self, *args, **kw):
        kw["timeout_wait"] = max(int(kw.get("timeout_wait") or 0), TIMEOUT_WAIT)
        orig_me_init(self, *args, **kw)

    me.UnityEnvironment.__init__ = me_init
    cond = {"cell": a.cell, "policy": a.policy, "seeds": a.seeds, "params": params, "fleet_on_duty_bays": fleet,
            "machines": size, "gated": gated, "force_offset": a.force_offset, "law_h": h, "law_alpha": law_alpha,
            "law_file": str(LAW.relative_to(REPO)), "checkpoint": ckpt, "rule": RULE_OF.get(a.policy),
            "code": str(CODE.relative_to(REPO)), "stamp": (CODE.parent / "STAMP").read_text(),
            "player_md5": md5, "worker_id": wid, "evaluate_args": ev}
    (out / "cond.json").write_text(json.dumps(cond, indent=1))
    t0 = time.time()
    with open(out / "eval.out", "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        evaluate.main(ev)
    cond["python_picks"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(record.items())}
    (out / "cond.json").write_text(json.dumps(cond, indent=1))
    with open(out / "episodes.csv") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(seeds):
        raise SystemExit(f"{a.cell}/{out.name}: {len(rows)} episodes, expected {len(seeds)} (see {out}/eval.out)")
    bad = [r["seed"] for r in rows if r["deadlock"] == "True" or r["timed_out"] == "True" or r["tick_error"] == "True"]
    if bad:
        print(f"{a.cell}/{out.name}: deadlock / time-out / tick error on seeds {bad} (kept; the analysis reports them)")
    (out / "DONE").write_text(f"{time.time() - t0:.0f}\n")
    print(f"{a.cell}/{out.name}: {len(rows)} episodes in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    if sys.argv[1:2] == ["list"]:
        for t in tasks():
            print(*t)
    else:
        main()
