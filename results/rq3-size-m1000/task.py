"""
@file task.py
@brief rq3-size-m1000 (2026-10-20, user: reduced 1000-machine column of the floor-size grid): one episode per task on the
       1000-machine floor (../dev-size-floors/floors/m1000: 1000 bays, 520 AGVs on duty, short blocks 260, B2 load x 0.9,
       arrival cap off, c ~ U[1.75, 2.5]), test seeds 0-4, twin, the grid's harness (../rq3-size-grid/task.py ->
       rq3-gen-v5 run_task, frozen v5 code). One episode takes ~6 h locally (seed 0, 7.7 GB) and ~1.6x that on the
       cluster's nodes, and episode time varies +-50% by seed, so every (policy, seed) is its own task.
Groups:
  mdd   MDD-PTECT (the reference; the best pair at 50-200 machines)
  srt   SRT-PTECT (the runner-up at 150-200 machines)
  t15   the final policy (train-v5-cmix s3; also the 15-machine family's best seed at 200 machines)
  t50 / t100 / tmix   train-size-m50 / -m100 / -mix, the seed in picks.json (pick.py: lowest tardiness at 200
        machines in ../rq3-size-grid, the same rule that gives s3 for t15)
Output: m1000/<group>_s<seed>/ (a finished task, DONE, is skipped). Seed 0 of mdd / t15 repeats
../dev-size-1000-probe (local) on the cluster: a cross-machine reproduction check.
@par Usage
@code{.sh}
python results/rq3-size-m1000/task.py --group mdd --seed 3
@endcode
"""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("grid_task", HERE.parent / "rq3-size-grid" / "task.py")
gt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gt)
rt, st = gt.rt, gt.st
rt.CONDITIONS["m1000"] = (st.size_params(200, 520, 260), gt.FLOORS / "m1000" / "des_floor.json", None, (520, 1000))
ARM_RUNS = {"t50": "train-size-m50", "t100": "train-size-m100", "tmix": "train-size-mix"}
GROUPS = ["mdd", "srt", "t15", *ARM_RUNS]
_n_policies = rt.n_policies


def n_policies(group):
    """run_task's count with "--pdr none" counted as no pair (it counts it as one)."""
    _, a = rt.GROUPS[group]
    return _n_policies(group) - (a[a.index("--pdr") + 1] == "none")


rt.n_policies = n_policies


def group_args(group):
    if group == "mdd":
        return ["--pdr", "MDD-PTECT"]
    if group == "srt":
        return ["--pdr", "SRT-PTECT"]
    if group == "t15":
        return ["--pdr", "none", "--checkpoint", "results/train-v5-cmix_s3/checkpoint.pt"]
    picks = json.loads((HERE / "picks.json").read_text())
    return ["--pdr", "none", "--checkpoint", f"results/{ARM_RUNS[group]}_s{picks[group]}/checkpoint.pt"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=GROUPS)
    ap.add_argument("--seed", required=True, type=int, choices=range(5))
    ap.add_argument("--out-root", default=str(HERE))
    a = ap.parse_args(argv)
    name = f"{a.group}_s{a.seed}"
    rt.GROUPS[name] = (900, group_args(a.group))
    rt.main(["--cond", "m1000", "--group", name, "--seeds", str(a.seed), "--out-root", a.out_root])


if __name__ == "__main__":
    main()
