"""
@file task.py
@brief eval-v5-slot (2026-10-21, user): the PPO slot-length sweep's evaluation. Each arm (train-v5-slot<X> s0-s2; 900 s =
       the final-policy run train-v5-cmix s0-s4) plus MDD-PTECT on B2 (condition "base", c ~ U[1.75, 2.5]), test seeds
       0-39, through ../rq3-gen-v5/run_task.py (frozen code 6d13412a, the harness that scored the final policy), with
       --slot-seconds = the arm's training slot. Output: base/slot<X>/ (DONE = finished).
@par Usage
@code{.sh}
python results/eval-v5-slot/task.py --slot 300
@endcode
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rq3-gen-v5"))
import run_task as rt                                              # noqa: E402

SLOTS = (60, 120, 180, 300, 600, 900, 1200, 1500, 1800)


def checkpoints(x):
    if x == 900:
        return [f"results/train-v5-cmix_s{k}/checkpoint.pt" for k in range(5)]
    return [f"results/train-v5-slot{x}_s{k}/checkpoint.pt" for k in range(3)]


for x in SLOTS:
    rt.GROUPS[f"slot{x}"] = (float(x), ["--pdr", "MDD-PTECT"] + [a for c in checkpoints(x) for a in ("--checkpoint", c)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, required=True, choices=SLOTS)
    ap.add_argument("--seeds", default="0-39")
    ap.add_argument("--out-root", default=str(HERE))
    a = ap.parse_args(argv)
    missing = [c for c in checkpoints(a.slot) if not (HERE.parents[1] / c).exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    rt.main(["--cond", "base", "--group", f"slot{a.slot}", "--seeds", a.seeds, "--out-root", a.out_root])


if __name__ == "__main__":
    main()
