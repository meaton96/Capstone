"""
@file run_task.py
@brief eval-v5-lookahead (2026-10-15): one checkpoint of train-v5-lookahead (the v5 stack + exact-future look-ahead
       features), played in the twin on the B2 fitting seeds 2000-2159 (--split fit) or test seeds 0-39 (--split test).
@details Same settings as ../eval-v5-final/run.sh and ../eval-v5-curve (frozen code rq2-rebase-v5/code/env, floor
         dev-popart-smoke/floor7, B2 params, random warm-up, 21,600 s window, 900 s slots, 5,400 s horizon,
         deterministic). evaluate.py computes the look-ahead features because the checkpoint's policy reads them.
         The state-only PPO, MDD-PTECT, the oracle and the planner on the same seeds come from eval-v5-final,
         rq2-rebase-v5 and rq2-perturbed-v5. A finished task (DONE) is skipped.
@par Usage
@code{.sh}
python results/eval-v5-lookahead/run_task.py --run train-v5-lookahead_s0 --ckpt checkpoint --split test
@endcode
"""
import argparse
import contextlib
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CODE = REPO / "results" / "rq2-rebase-v5" / "code" / "env"
B2 = {"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400,
      "horizon_seconds": 30600,
      "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2},
                   {"name": "surge", "utilization": [1.0, 1.6]},
                   {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", default="2000-2159")
    ap.add_argument("--split", default="fit", choices=["fit", "test"], help="test: B2 test seeds 0-39")
    a = ap.parse_args()
    if a.split == "test":
        a.seeds = "0-39"
    out = HERE / (f"{a.run}__{a.ckpt}" + ("__test" if a.split == "test" else ""))
    if (out / "DONE").exists():
        print(f"{out.name}: done, skipped")
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ck = f"results/{a.run}/{a.ckpt}.pt"
    ev = ["--twin", "results/dev-popart-smoke/floor7/des_floor.json",
          "--reward-spec", str(CODE / "config" / "rewards" / "tardiness.json"), "--scenario-generator", "randomized",
          "--random-warmup", "--episode-duration-seconds", "21600", "--params", json.dumps(B2), "--seeds", a.seeds,
          "--pdr", "none", "--slot-seconds", "900", "--discount-horizon-s", "5400", "--device", "cpu",
          "--checkpoint", ck, "--out", str(out)]
    sys.path.insert(0, str(CODE))
    os.chdir(REPO)
    import evaluate
    t0 = time.time()
    with open(out / "eval.out", "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        evaluate.main(ev)
    rows = list(csv.DictReader(open(out / "episodes.csv")))
    want = len(evaluate.parse_seeds(a.seeds))
    if len(rows) != want:
        raise SystemExit(f"{out.name}: {len(rows)} episodes, expected {want}")
    (out / "DONE").write_text(f"{time.time() - t0:.0f}\n")
    print(f"{out.name}: {len(rows)} episodes in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
