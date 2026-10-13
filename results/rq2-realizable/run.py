"""
@file run.py
@brief rq2-realizable (2026-10-07): how much of the oracle's headroom can a policy without hindsight get? Simple,
       fully observable switching rules, fitted on training instances and scored on held-out ones.

Why (user, 10-07): the oracle's 10.8% over MDD-TECT (B2) is chosen in hindsight; RL, the action prior and oracle
cloning all end near MDD-TECT, and no state feature predicts the oracle's choice well (AUC <= 0.70). This bounds the
gain from the other side, with policies a planner could write down:
  - fixed:     the 15 H15 pairs held all episode
  - regime map: one pair per regime type (normal / surge / short fleet), switching at the block boundaries; every map
               over the 6 strongest pairs (216). The regime is observable (fleet size on duty, arrival load).
  - threshold: pair A while a state feature is below a threshold, pair B otherwise, re-checked at each slot start;
               features WIP and jobs waiting for an AGV per on-duty AGV (dev-congestion-signal's strongest), 6
               thresholds each, ordered pairs of the 4 strongest pairs (144).
One call = one (setting, seed): every policy's window tardiness -> <setting>/s<seed>.json (skips existing).
analyze.py picks each family's best policy on training seeds 2000-2159 and scores it on test seeds 0-39.
Settings (../rq2-oracle-steady/sim.py): B2 (the benchmark of every RL result so far) and S6 (steady state, 6 h blocks).
@par Usage
@code{.sh}
python results/rq2-realizable/run.py --setting B2 --seed 0
@endcode
"""
import argparse
import importlib.util
import itertools
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sim", HERE.parent / "rq2-oracle-steady" / "sim.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

MAP_PAIRS = ["MDD-TECT", "MDD-ECT", "ATC-TECT", "ATC-ECT", "SRT-ECT", "SRT-TECT"]
THRESH_PAIRS = ["MDD-TECT", "ATC-TECT", "MDD-ECT", "ATC-ECT"]
THRESHOLDS = {"wip": [15, 20, 30, 40, 60, 90], "wait_per_agv": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]}
REGIMES = ("normal", "surge", "short")


def policies():
    out = {f"fixed:{p}": [p] for p in sim.PAIRS}
    for combo in itertools.product(MAP_PAIRS, repeat=3):
        m = dict(zip(REGIMES, combo))
        out["map:" + "/".join(combo)] = (lambda m: lambda tw, dec, k, ep: m[ep.regime_at(tw.now)])(m)
    for feat, ths in THRESHOLDS.items():
        for th in ths:
            for lo, hi in itertools.permutations(THRESH_PAIRS, 2):
                out[f"thr:{feat}<{th}:{lo}|{hi}"] = (lambda f, th, lo, hi: lambda tw, dec, k, ep:
                                                     lo if sim.state_features(tw)[f] < th else hi)(feat, th, lo, hi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=tuple(sim.SETTINGS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    out = HERE / a.setting / f"s{a.seed}.json"
    if out.exists():
        print("exists", out)
        return
    t = time.time()
    ep = sim.Episode(a.setting, a.seed)
    res = {name: ep.play(pol)[0] for name, pol in policies().items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": a.seed, "setting": a.setting, "warmup": ep.warmup, "tard": res,
                               "wall_s": time.time() - t}))
    fx = {k: v for k, v in res.items() if k.startswith("fixed:")}
    print(f"{a.setting} s{a.seed}: {len(res)} policies, best fixed {min(fx, key=fx.get)} {min(fx.values()):.1f}, "
          f"best overall {min(res, key=res.get)} {min(res.values()):.1f}, {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
