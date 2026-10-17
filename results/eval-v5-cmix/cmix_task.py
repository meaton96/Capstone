"""
@file cmix_task.py
@brief eval-v5-cmix (2026-10-14): one (split, condition, group) of the due-date evaluation of the PPO families trained
       at different due-date ranges. A thin wrapper around ../rq3-gen-v5/run_task.py (same frozen code, floors, B2
       settings and condition definitions; due-tight c [1.25, 1.75], base [1.75, 2.5], due-loose [2.5, 3.5]) that
       adds the new checkpoint groups and seed splits.

Groups (each + MDD-PTECT as a run-to-run check, except pdr):
  pdr    the 6 v5 pairs + MDD-TECT                         (rq3-gen-v5's)
  ppo    PPO trained at B2's c 1.75-2.5, train-v5-nokl s0-s2 + train-v5-final s3-s4   (rq3-gen-v5's)
  loose  PPO trained at c 2.5-3.5, train-v5-loose s0-s4
  cmix   PPO trained at c 1.25-3.5, train-v5-cmix s0-s4
Splits: test = seeds 0-39; fit2000 / fit2040 / fit2080 / fit2120 = the fitting seeds 2000-2159 in chunks of 40 (the
picks are made there). test x {pdr, ppo} is not rerun: analyze.py reads rq3-gen-v5's identical outputs.
Output: <split>/<cond>/<group>/ (episodes.csv, eval.out, cond.json, DONE); a finished task is skipped.
@par Usage
@code{.sh}
python results/eval-v5-cmix/cmix_task.py --split fit2000 --cond due-loose --group loose
@endcode
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rq3-gen-v5"))
import run_task as rt                                              # noqa: E402  (rq3-gen-v5's harness)

SPLITS = {"test": "0-39", "fit2000": "2000-2039", "fit2040": "2040-2079", "fit2080": "2080-2119",
          "fit2120": "2120-2159"}
CONDS = ["due-tight", "base", "due-loose"]
for name, run in (("loose", "train-v5-loose"), ("cmix", "train-v5-cmix")):
    ck = [f"results/{run}_s{k}/checkpoint.pt" for k in range(5)]
    rt.GROUPS[name] = (900, ["--pdr", "MDD-PTECT"] + [a for c in ck for a in ("--checkpoint", c)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--cond", required=True, choices=CONDS)
    ap.add_argument("--group", required=True, choices=sorted(rt.GROUPS))
    a = ap.parse_args()
    rt.main(["--cond", a.cond, "--group", a.group, "--seeds", SPLITS[a.split], "--out-root", str(HERE / a.split)])


if __name__ == "__main__":
    main()
