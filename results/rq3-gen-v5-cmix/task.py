"""
@file task.py
@brief rq3-gen-v5-cmix (2026-10-17, user: the final policy is the PPO trained on mixed due dates): the RQ3 conditions
       sweep for the due-date-trained PPO families. A thin wrapper around ../rq3-gen-v5/run_task.py (same frozen code,
       floors, B2 settings, fleet wrapper and condition definitions), like ../eval-v5-cmix/cmix_task.py.
Groups (each + MDD-PTECT as a run-to-run check against rq3-gen-v5's pdr group):
  cmix   PPO trained at c 1.25-3.5, train-v5-cmix s0-s4 (the final policy is s3, picked on the fitting seeds)
  loose  PPO trained at c 2.5-3.5, train-v5-loose s0-s4 (reported next to it)
Conditions: the 8 that eval-v5-cmix did not run (base / due-tight / due-loose on test seeds 0-39 are there).
Output: <cond>/<group>/ (episodes.csv, eval.out, cond.json, DONE); a finished task is skipped.
@par Usage
@code{.sh}
python results/rq3-gen-v5-cmix/task.py --cond lay-j --group cmix
@endcode
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rq3-gen-v5"))
import run_task as rt                                              # noqa: E402  (rq3-gen-v5's harness)

CONDS = ["load-0.8", "load-1.2", "agv5-bay9", "agv7-bay9", "agv9-bay9", "lay-c", "lay-g", "lay-j"]
for name, run in (("loose", "train-v5-loose"), ("cmix", "train-v5-cmix")):
    ck = [f"results/{run}_s{k}/checkpoint.pt" for k in range(5)]
    rt.GROUPS[name] = (900, ["--pdr", "MDD-PTECT"] + [a for c in ck for a in ("--checkpoint", c)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=CONDS)
    ap.add_argument("--group", required=True, choices=["cmix", "loose"])
    a = ap.parse_args()
    rt.main(["--cond", a.cond, "--group", a.group, "--seeds", "0-39", "--out-root", str(HERE)])


if __name__ == "__main__":
    main()
