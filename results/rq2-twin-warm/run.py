"""
@file run.py
@brief rq2-twin-warm: is the short-episode deficit a warm-up transient? (10-04, control for rq2-twin-fleet)

@details rq2-twin-fleet found 2.4% median in-episode headroom for 1.5 h windows vs 8.2% for 6 h ones at the same
stationary load, and dev-load-calib shows tardiness per hour ramping for ~3 h after the warm-up. Same oracle and load
as rq2-twin-fleet's single-cal (imported from ../rq2-twin-fleet/run.py), with the warm-up drawn from segment starts
>= 2 h (single-warm2h) or >= 3 h (single-warm3h); horizon 30,600 s leaves room. Seeds 0-39.

@par Usage
@code{.sh}
nohup nice -n 10 .venv/bin/python results/rq2-twin-warm/run.py --workers 16 > results/rq2-twin-warm/run.out 2>&1 &
@endcode
"""
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("fleet", REPO / "results/rq2-twin-fleet/run.py")
fleet = importlib.util.module_from_spec(_spec)
sys.modules["fleet"] = fleet
_spec.loader.exec_module(fleet)

fleet.HERE = HERE
MIN_WARMUP = {"single-warm2h": 7200.0, "single-warm3h": 10800.0}
for name in MIN_WARMUP:
    fleet.SETTINGS[name] = {"window": 5400.0, "params": {**fleet.NORMAL, "horizon_seconds": 30600.0}}


def scenario(setting, seed):
    from scenarios.randomized import DEFAULT_PARAMS, randomized_generator
    st = fleet.SETTINGS[setting]
    params = dataclasses.replace(DEFAULT_PARAMS, failure_probability=0.0, due_date_allowance_range=(1.75, 2.5),
                                 **st["params"])
    return randomized_generator(st["window"], random_warmup=True, params=params,
                                min_warmup_seconds=MIN_WARMUP.get(setting, 0.0))(seed)


fleet.scenario = scenario


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    (HERE / "tasks").mkdir(exist_ok=True)
    stamp = fleet.due.code_stamp()
    for f in (Path(__file__), REPO / "results/rq2-twin-fleet/run.py"):
        stamp["files"][str(f.relative_to(REPO))] = hashlib.md5(f.read_bytes()).hexdigest()[:12]
    stamp["id"] = hashlib.md5(json.dumps(stamp["files"], sort_keys=True).encode()).hexdigest()[:12]
    fleet.STAMP = stamp
    fleet.due.floor()
    print(f"code {stamp['id']} at {stamp['git_head']}", flush=True)
    tasks = [(st, s) for s in range(40) for st in MIN_WARMUP]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (name, msg, wall) in enumerate(pool.imap_unordered(fleet.run_task, tasks), 1):
            print(f"[{i}/{len(tasks)} {time.time() - t0:6.0f}s] {name}: {msg} ({wall}s)", flush=True)


if __name__ == "__main__":
    main()
