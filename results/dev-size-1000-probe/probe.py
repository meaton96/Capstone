"""
@file probe.py
@brief dev-size-1000-probe (2026-10-19, user: floor-size study): can the twin run a 1000-machine floor, and how long does
       one B2-structured episode take? One seed (0), one MDD-PTECT episode and one episode of the final policy (cmix s3),
       on ../dev-size-floors/floors/m1000 (1000 bays) with the floor-size study's scaling (0.52 AGVs per machine: 520
       on duty, short blocks 260; B2 load x 0.9; arrival cap off; c ~ U[1.75, 2.5]). Wall time per episode goes to
       timing.json. Uses ../rq3-size-v5/task.py's harness (rq3-gen-v5 run_task) with an extra condition.
@par Usage
@code{.sh}
python results/dev-size-1000-probe/probe.py --group ref    # MDD-PTECT;  --group s3: the final policy
@endcode
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("size_task", HERE.parent / "rq3-size-v5" / "task.py")
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)
rt = st.rt
FLOOR = HERE.parent / "dev-size-floors" / "floors" / "m1000" / "des_floor.json"
rt.CONDITIONS["m1000"] = (st.size_params(200, 520, 260), FLOOR, None, (520, 1000))
rt.GROUPS["ref"] = (900, ["--pdr", "MDD-PTECT"])
rt.GROUPS["s3"] = (900, ["--pdr", "none", "--checkpoint", "results/train-v5-cmix_s3/checkpoint.pt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["ref", "s3"])
    a = ap.parse_args()
    t0 = time.time()
    rt.main(["--cond", "m1000", "--group", a.group, "--seeds", "0", "--out-root", str(HERE)])
    f = HERE / "timing.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    d[a.group] = {"wall_s": round(time.time() - t0, 1)}
    f.write_text(json.dumps(d, indent=1))
    print(a.group, d[a.group])


if __name__ == "__main__":
    main()
