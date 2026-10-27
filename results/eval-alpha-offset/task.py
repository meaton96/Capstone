"""
@file task.py
@brief eval-alpha-offset (2026-10-25, user: "the law sets alpha, the head picks an offset"): one task of the evaluation
       of the offset arms (train-alpha-off-het / -b2, v7 code: alpha = min(4, max(0, 3 - 6 sigma) x offset)) on 15-machine
       B2 floors with machine-speed spread s in CELLS (s 0 = B2 itself), test seeds 0-39: ../eval-alpha-spread's cells,
       seeds and arguments, so its final / het / b2 / grid outputs are the references (not rerun).
Groups:
  off-het  train-alpha-off-het s0-s4 + MDD-PTECT (two-element: alpha 3) + MDD-PTECT@x1 (the law alone, continuous alpha),
           v7 code (../train-alpha-off-het/code), decision log (the offset chosen per slot)
  off-b2   train-alpha-off-b2 s0-s4 + the same two rules, v7 code (../train-alpha-off-b2/code), decision log
Both use ../rq3-gen-v5/run_task.py's B2 arguments (twin, tardiness reward, random warm-up, 21,600 s window, 900 s slots,
5,400 s discount horizon) through hetero.py --spread s,s --entry evaluate (unchanged from ../dev-alpha-action, check O2
in ../dev-alpha-offset). A task writes out/<cell>/<group>/ (evaluate.py), then DONE_<group>_<seeds> in out/<cell>/; a
task whose DONE exists is skipped, an unfinished one is wiped and rerun, so relaunching resumes. One process per task
with a 12 GB address-space cap (as eval-alpha-spread).
@par Usage (from the repo root)
@code{.sh}
.venv/bin/python results/eval-alpha-offset/task.py 0.5 off-het 0-39
@endcode
"""
import argparse
import csv
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
CELLS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = 5
CKPTS = {"off-het": [f"results/train-alpha-off-het_s{k}/checkpoint.pt" for k in range(SEEDS)],
         "off-b2": [f"results/train-alpha-off-b2_s{k}/checkpoint.pt" for k in range(SEEDS)]}
CODE = {"off-het": R / "train-alpha-off-het" / "code", "off-b2": R / "train-alpha-off-b2" / "code"}
RULES = "MDD-PTECT,MDD-PTECT@x1"
EVAL_AS_BYTES = 12 << 30         # one evaluate.py run (torch): measured 0.5 GB resident (eval-alpha-spread)


def parse_seeds(text):
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def cell_name(s):
    return f"{s:g}"


def cap(nbytes):
    """preexec_fn: limit the child's address space to nbytes."""
    return lambda: resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))


def run_evaluate(s, group, seeds, root, ckpts=None):
    out = root / "out" / cell_name(s) / group
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    sys.path.insert(0, str(R / "rq3-gen-v5"))
    import run_task as rt
    params, floor = rt.CONDITIONS["base"][0], rt.CONDITIONS["base"][1]
    code = CODE[group]
    ckpts = ckpts or CKPTS[group]
    missing = [c for c in ckpts if not (REPO / c).exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    ev = [sys.executable, str(code / "hetero.py"), "--spread", f"{s:g},{s:g}", "--entry", "evaluate", "--",
          "--twin", str(floor), "--reward-spec", str(code / "env" / "config" / "rewards" / "tardiness.json"),
          "--scenario-generator", "randomized", "--random-warmup", "--episode-duration-seconds", "21600",
          "--params", json.dumps(params), "--seeds", seeds, "--slot-seconds", "900", "--discount-horizon-s", "5400",
          "--device", "cpu", "--pdr", RULES, *[a for c in ckpts for a in ("--checkpoint", c)], "--decision-log",
          "--out", str(out)]
    (out / "task.json").write_text(json.dumps({"cell": s, "group": group, "seeds": seeds, "code": str(code),
                                               "checkpoints": ckpts, "args": ev[1:]}, indent=1))
    with open(out / "eval.out", "w") as log:
        subprocess.run(ev, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"), preexec_fn=cap(EVAL_AS_BYTES))
    rows = list(csv.DictReader(open(out / "episodes.csv")))
    want = (len(RULES.split(",")) + len(ckpts)) * len(parse_seeds(seeds))
    if len(rows) != want:
        raise SystemExit(f"{s}/{group}: {len(rows)} episodes, expected {want} (see {out}/eval.out)")
    return f"{len(rows)} episodes"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", type=float)
    ap.add_argument("group", choices=tuple(CKPTS))
    ap.add_argument("seeds", nargs="?", default="0-39")
    ap.add_argument("--checkpoints", default=None, help="comma list replacing the group's checkpoints (dry runs)")
    ap.add_argument("--root", default=str(HERE), help="output root (default: this folder; dry runs elsewhere)")
    a = ap.parse_args(argv)
    if a.cell not in CELLS:
        raise SystemExit(f"cell {a.cell} not in {CELLS}")
    root = Path(a.root)
    done = root / "out" / cell_name(a.cell) / f"DONE_{a.group}_{a.seeds}"
    if done.exists():
        print(f"{done.name} ({cell_name(a.cell)}): done, skipped")
        return
    t = time.time()
    msg = run_evaluate(a.cell, a.group, a.seeds, root, a.checkpoints.split(",") if a.checkpoints else None)
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(f"{time.time() - t:.0f}\n")
    print(f"s {cell_name(a.cell)} {a.group} {a.seeds}: {msg} in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
