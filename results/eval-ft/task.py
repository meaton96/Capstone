"""
@file task.py
@brief eval-ft (2026-10-25, plan §2b): one task of the fine-tune evaluation. One budget of one fine-tune arm (its 5
       training seeds) + MDD-PTECT on one cell, twin, B2 test seeds, deterministic, frozen v5 code (6d13412a), B2
       window / slot / reward settings of ../rq3-gen-v5/run_task.py (tardiness spec, random warm-up, 21,600 s window,
       900 s slot actions, 5,400 s discount horizon, c ~ U[1.75, 2.5]).
Arms: m100 = ../train-ft-m100, het = ../train-ft-het (the final policy train-v5-cmix s0-s4 resumed on the target floor).
Budgets: b in {2, 5, 10, 24} fine-tune updates = checkpoint_step<98,304 + 2,048 b>.pt (b 24: checkpoint.pt, 147,456).
Cells:
  m100  the 100-machine floor as ../rq3-size-v5 and ../rq3-size-grid evaluate it (agv-ratio-scale m100, 52 AGVs on
        duty, short blocks 26, B2 load x 0.9, arrival cap off): ../rq3-gen-v5/run_task.py in this process, 10-seed
        chunks as the grid
  s1    B2 with machine-speed spread s = 1 (capacity-matched, ../train-ft-het/code/hetero.py --spread 1,1), as
        ../eval-alpha-spread's s 1 cell
  s0    B2 itself through the same hetero.py path (s 0 leaves the instances untouched: 240 / 240 episodes equal
        ../eval-v5-cmix/test/base/cmix, checked 10-25)
Output: out/<cell>/<arm>-b<b>[_s<a>-<b>]/ (episodes.csv, eval.out, task.json or cond.json, DONE). A task whose DONE
exists is skipped; an unfinished one is wiped and rerun, so relaunching resumes.
@par Usage (from the repo root; tasks_het.txt / tasks_m100.txt list the tasks, launch.sh runs them)
@code{.sh}
.venv/bin/python results/eval-ft/task.py s1 het 10 0-39
.venv/bin/python results/eval-ft/task.py m100 m100 24 0-9
@endcode
"""
import argparse
import csv
import importlib.util
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
REPO = HERE.parents[1]
START = 98304                                   # train-v5-cmix checkpoint.pt global_step (48 updates x 2,048)
BUDGETS = (2, 5, 10, 24)
ARMS = {"m100": "train-ft-m100", "het": "train-ft-het"}
CELLS = {"m100": ("m100", "s0"), "het": ("s1", "s0")}
CHUNKS = ("0-9", "10-19", "20-29", "30-39")
HETERO = R / "train-ft-het" / "code"            # frozen v5 env + hetero.py (= ../eval-alpha-spread/code_v5)
SPREAD = {"s1": 1.0, "s0": 0.0}
## @brief Address-space cap per evaluate.py process (as ../eval-alpha-spread): a runaway task fails alone.
AS_BYTES = 12 << 30


def step(b):
    return START + b * 128 * 16


def checkpoints(arm, b):
    """The 5 training seeds' checkpoints after b fine-tune updates (repo-relative)."""
    name = "checkpoint.pt" if b == BUDGETS[-1] else f"checkpoint_step{step(b)}.pt"
    return [f"results/{ARMS[arm]}_s{k}/{name}" for k in range(5)]


def check_checkpoints(ckpts, b):
    """Every checkpoint exists and sits at the budget's step (b 24: checkpoint.pt must be the finished run's)."""
    import torch
    missing = [c for c in ckpts if not (REPO / c).exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    for c in ckpts:
        g = int(torch.load(REPO / c, map_location="cpu", weights_only=True)["global_step"])
        if g != step(b):
            raise SystemExit(f"{c}: global_step {g:,}, expected {step(b):,} for budget {b}")


def parse_seeds(text):
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def cap(nbytes):
    """preexec_fn: limit the child's address space to nbytes."""
    return lambda: resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))


def run_m100(ckpts, seeds, group, root):
    """../rq3-gen-v5/run_task.py in process, with rq3-size-v5's m100 condition and this budget as a group."""
    spec = importlib.util.spec_from_file_location("size_task", R / "rq3-size-v5" / "task.py")
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    rt = st.rt
    rt.GROUPS[group] = (900, ["--pdr", "MDD-PTECT"] + [a for c in ckpts for a in ("--checkpoint", c)])
    resource.setrlimit(resource.RLIMIT_AS, (AS_BYTES, AS_BYTES))
    rt.main(["--cond", "m100", "--group", group, "--seeds", seeds, "--out-root", str(root)])


def run_hetero(cell, arm, b, ckpts, seeds, out):
    """evaluate.py through hetero.py --spread s,s (../eval-alpha-spread/task.py's run_evaluate, own checkpoints)."""
    sys.path.insert(0, str(R / "rq3-gen-v5"))
    import run_task as rt
    params, floor = rt.CONDITIONS["base"][0], rt.CONDITIONS["base"][1]
    s = SPREAD[cell]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ev = [sys.executable, str(HETERO / "hetero.py"), "--spread", f"{s:g},{s:g}", "--entry", "evaluate", "--",
          "--twin", str(floor), "--reward-spec", str(HETERO / "env" / "config" / "rewards" / "tardiness.json"),
          "--scenario-generator", "randomized", "--random-warmup", "--episode-duration-seconds", "21600",
          "--params", json.dumps(params), "--seeds", seeds, "--slot-seconds", "900", "--discount-horizon-s", "5400",
          "--device", "cpu", "--pdr", "MDD-PTECT", *[a for c in ckpts for a in ("--checkpoint", c)], "--out", str(out)]
    (out / "task.json").write_text(json.dumps({"cell": cell, "spread": s, "arm": arm, "budget": b, "seeds": seeds,
                                               "code": str(HETERO), "checkpoints": ckpts, "args": ev[1:]}, indent=1))
    with open(out / "eval.out", "w") as log:
        subprocess.run(ev, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"), preexec_fn=cap(AS_BYTES))
    rows = list(csv.DictReader(open(out / "episodes.csv")))
    want = (1 + len(ckpts)) * len(parse_seeds(seeds))
    if len(rows) != want:
        raise SystemExit(f"{cell}/{out.name}: {len(rows)} episodes, expected {want} (see {out}/eval.out)")
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", choices=("m100", "s1", "s0"))
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("budget", type=int, choices=BUDGETS)
    ap.add_argument("seeds", nargs="?", default="0-39")
    ap.add_argument("--checkpoints", default=None,
                    help="comma list replacing the budget's checkpoints (dry runs: no step or chunk check)")
    ap.add_argument("--out-root", default=str(HERE / "out"), help="parent of <cell>/<group>/ (dry runs elsewhere)")
    a = ap.parse_args(argv)
    if a.cell not in CELLS[a.arm]:
        ap.error(f"arm {a.arm} is evaluated on {CELLS[a.arm]}")
    if a.cell == "m100" and a.seeds not in CHUNKS and not a.checkpoints:
        ap.error(f"m100 runs in 10-seed chunks {CHUNKS}")
    os.chdir(REPO)                                   # checkpoint paths are repo-relative
    group = f"{a.arm}-b{a.budget}" + (f"_s{a.seeds}" if a.cell == "m100" else "")
    root = Path(a.out_root).resolve()
    out = root / a.cell / group
    if (out / "DONE").exists():
        print(f"{a.cell}/{group}: done, skipped")
        return
    ckpts = a.checkpoints.split(",") if a.checkpoints else checkpoints(a.arm, a.budget)
    if not a.checkpoints:
        check_checkpoints(ckpts, a.budget)
    t = time.time()
    if a.cell == "m100":
        run_m100(ckpts, a.seeds, group, root)         # run_task writes its own DONE
    else:
        n = run_hetero(a.cell, a.arm, a.budget, ckpts, a.seeds, out)
        (out / "DONE").write_text(f"{time.time() - t:.0f}\n")
        print(f"{a.cell}/{group}: {n} episodes in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
