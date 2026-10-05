"""
@file run.py
@brief rq2-twin-blocks-ctrl: controls for rq2-twin-blocks (10-04). The blocks setting raised in-episode switching
       headroom 1.4% -> 5.8% median, but its 21,600 s window also lets the backlog grow (best-pair tardiness per hour
       about 3x from the first to the last hour) and gives the oracle 24 instead of 6 switch points. Two controls, same
       seeds and oracle (machinery imported from ../rq2-twin-blocks/run.py):
  - long:        one regime per episode (blocks off), 21,600 s window: length and backlog, no regime changes;
  - blocks-base: regime blocks (c and op mean per block) with base load only: regime changes without sustained overload.

@par Usage
@code{.sh}
nohup nice -n 19 .venv/bin/python results/rq2-twin-blocks-ctrl/run.py --workers 16 > results/rq2-twin-blocks-ctrl/run.out 2>&1 &
@endcode
"""
import hashlib
import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("blocks", REPO / "results/rq2-twin-blocks/run.py")
blocks = importlib.util.module_from_spec(_spec)
sys.modules["blocks"] = blocks          # so the pool can pickle blocks.run_task by name (workers fork this state)
_spec.loader.exec_module(blocks)

blocks.HERE = HERE
blocks.SETTINGS["long"] = {"window": 21600.0, "params": {"horizon_seconds": 30600.0}}
blocks.SETTINGS["blocks-base"] = {"window": 21600.0, "params": {"regime_block_seconds": 5400.0, "horizon_seconds": 30600.0,
                                                                "load_mix": ({"name": "base"},)}}
_scenario = blocks.scenario


def scenario(setting, seed):
    import dataclasses
    from scenarios.randomized import DEFAULT_PARAMS, randomized_generator
    if setting != "blocks-base":
        return _scenario(setting, seed)
    st = blocks.SETTINGS[setting]
    params = dataclasses.replace(DEFAULT_PARAMS, failure_probability=0.0, due_date_allowance_range=(1.75, 2.5),
                                 **st["params"])
    return randomized_generator(st["window"], random_warmup=True, params=params)(seed)


blocks.scenario = scenario


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seeds", default="0-39")
    args = ap.parse_args()
    (HERE / "tasks").mkdir(exist_ok=True)
    stamp = blocks.due.code_stamp()
    for f in (Path(__file__), REPO / "results/rq2-twin-blocks/run.py"):
        stamp["files"][str(f.relative_to(REPO))] = hashlib.md5(f.read_bytes()).hexdigest()[:12]
    stamp["id"] = hashlib.md5(json.dumps(stamp["files"], sort_keys=True).encode()).hexdigest()[:12]
    blocks.STAMP = stamp
    blocks.due.floor()
    print(f"code {stamp['id']} at {stamp['git_head']}", flush=True)
    tasks = [(st, s) for s in blocks.due.parse_seeds(args.seeds) for st in ("long", "blocks-base")]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (name, msg, wall) in enumerate(pool.imap_unordered(blocks.run_task, tasks), 1):
            print(f"[{i}/{len(tasks)} {time.time() - t0:6.0f}s] {name}: {msg} ({wall}s)", flush=True)


if __name__ == "__main__":
    main()
