"""
@file task.py
@brief rq3-size-remap (2026-10-19, user): a no-training diagnostic of why the final policy (cmix pick s3, trained on 15
       machines) does not transfer to 100 machines (../rq3-size-v5). The policy's set encoders and rule actions are
       size-agnostic, but three global scalars carry the floor size and were constant or narrow in training:
         s[12] floor size, n / 100 (always 0.15 in training);
         s[1]  active jobs, squash(WIP, 30) (near-saturated on large floors);
         s[15] decision options, squash(options, 5); for a routing decision = the eligible machines of the type,
               3 at 15 machines vs 10 / 20 at 50 / 100.
       This rewrites them at test time into the 15-machine range before the policy sees them, and nothing else:
         size  s[12] := 15 / 100
         wip   s[1]  := squash(WIP x 15 / n, 30)
         opt   s[15] := squash(options x 15 / n, 5) on routing decisions (event flag 1); dispatch options are a
               machine's queue, which does not grow with the floor
         all   the three together
       The counts are recovered exactly (inverse squash, rounded to the integer they are), so at n = 15 every remap is
       the identity, and run on B2 it reproduces ../eval-v5-cmix exactly (the plumbing check).
@details Reuses ../rq3-size-v5/task.py (conditions m50 / m100, groups, 10-seed chunks) on the frozen v5 code; the remap
         wraps TwinSchedulingEnv._obs. MDD-PTECT runs in every task as the run-to-run check (rules ignore the observation).
         Output: <remap>/<cond>/<group>[_s<a>-<b>]/; a finished task (DONE) is skipped.
@par Usage
@code{.sh}
python results/rq3-size-remap/task.py --remap all --cond m100 --chunk 0-9
python results/rq3-size-remap/task.py --remap all --cond base --seeds 0-1 --out-root /tmp/x   # identity check
@endcode
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("size_task", HERE.parent / "rq3-size-v5" / "task.py")
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)                     # registers m50 / m100, the cmix / loose groups and their chunks
rt = st.rt
REMAPS = {"size": {"size"}, "wip": {"wip"}, "opt": {"opt"}, "all": {"size", "wip", "opt"}}
WIP_SCALE, COUNT_SCALE = 30.0, 5.0               # des_twin/observation.py (frozen v5 code)


def _squash(x, scale):
    return np.float32(0.0) if x <= 0.0 else np.float32(x / (x + scale))


def _count(s, scale):
    """The integer count behind squash(count, scale) (exact for the counts an observation holds)."""
    s = float(s)
    return 0 if s <= 0.0 else int(round(scale * s / (1.0 - s)))


def remap(gs, flags, n, which):
    out = np.array(gs, dtype=np.float32, copy=True)
    k = 15.0 / float(n)
    if "size" in which:
        out[12] = np.float32(15) / np.float32(100)
    if "wip" in which:
        out[1] = _squash(_count(gs[1], WIP_SCALE) * k, WIP_SCALE)
    if "opt" in which and float(flags[1]) == 1.0:
        out[15] = _squash(_count(gs[15], COUNT_SCALE) * k, COUNT_SCALE)
    return out


def install(which):
    import env_wrappers.twin_env as te
    obs_fn = te.TwinSchedulingEnv._obs

    def _obs(self):
        o = obs_fn(self)
        o["global_scalars"] = remap(o["global_scalars"], o["event_flags"], len(self._twin.machines), which)
        return o

    te.TwinSchedulingEnv._obs = _obs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--remap", required=True, choices=sorted(REMAPS))
    ap.add_argument("--cond", required=True, choices=["m50", "m100", "base"])
    ap.add_argument("--group", default="cmix", choices=["cmix", "ppo"])
    ap.add_argument("--seeds", default="0-39")
    ap.add_argument("--chunk", choices=st.CHUNKS, default=None)
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(rt.CODE))
    install(REMAPS[a.remap])
    group, seeds = (f"{a.group}_s{a.chunk}", a.chunk) if a.chunk else (a.group, a.seeds)
    root = Path(a.out_root) if a.out_root else HERE / a.remap
    rt.main(["--cond", a.cond, "--group", group, "--seeds", seeds, "--out-root", str(root)])


if __name__ == "__main__":
    main()
