"""
@file collect.py
@brief rq2-oracle-bc-future (2026-10-15): the data for cloning the v5 switching oracle with the future job table.

Two task kinds, B2 twin, frozen v5 code (common.py), one CPU each:
  label <seeds>   new training labels: rq2-rebase-v5's greedy oracle (run_oracle, same steps: 6 pairs x 24 slots of
                  900 s, hold-to-end tails, the slot-start observation at caps 15 / 256 with an all-ones action mask),
                  plus, at the same capture, the exact future job table (128 rows, 2 h) and the slot-start time
                  -> out/labels/s<seed>.npz (rq2-rebase-v5 oracle keys + future, t_slot)
  future-level <level> <seeds>  the evaluation seeds' tables at a forecast level (redrawn: no information) along the
                  stored oracle trajectory -> out/future_<level>/s<seed>.npz (bc_train.py's per-slot redrawn check)
  future <seeds>  the evaluation seeds (test 0-39, fitting 2000-2159), whose labels exist in ../rq2-rebase-v5/out/oracle:
                  replay the stored oracle schedule, check the replay and the observations against the stored ones,
                  capture the future table and slot-start time -> out/future/s<seed>.npz
<seeds> is "3000" or "3000-3009". Finished seeds are skipped (resubmitting resumes).
@par Usage
@code{.sh}
python results/rq2-oracle-bc-future/collect.py label 3000-3009
python results/rq2-oracle-bc-future/collect.py future 0-39
python results/rq2-oracle-bc-future/collect.py list > results/rq2-oracle-bc-future/tasks_collect.txt
python results/rq2-oracle-bc-future/collect.py list4k   # data scaling (10-16): label tasks for seeds 4000-6999
@endcode
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                 # noqa: E402  (frozen code first; overlays the future table)

import numpy as np                                 # noqa: E402

OBS_KEYS = ("factory_grid", "machine_table", "job_table", "global_scalars", "event_flags", "action_mask")


def _capturer(builder, obs, fut, t_slot):
    def capture(tw, dec, k):
        o = builder.build(tw, dec)
        o["action_mask"] = np.ones_like(o["action_mask"])
        obs[k] = o
        fut[k] = C.future_table.future_job_table(tw, C.STORE_ROWS, C.HORIZON)
        t_slot[k] = tw.now
    return capture


def _fill(n, *dicts):
    """A slot with no decision keeps the previous slot's capture (rq2-rebase-v5 run_oracle)."""
    missing = [k for k in range(n) if k not in dicts[0]]
    for d in dicts:
        for k in missing:
            d[k] = d[max(i for i in d if i < k)]
    return missing


def label(seed, out_dir=None):
    """rq2-rebase-v5 run.py run_oracle(seed), plus the future table and the slot-start time at each capture."""
    from des_twin.observation import ObservationBuilder
    out = Path(out_dir or C.HERE / "out" / "labels") / f"s{seed}.npz"
    if out.exists():
        return f"exists {out}"
    t = time.time()
    ep = C.sim.Episode("B2", seed)
    n = C.sim.n_slots("B2")
    prefix, tails = [], np.zeros((n, len(C.V5)))
    for k in range(n):
        for i, p in enumerate(C.V5):
            tails[k, i] = ep.play(prefix + [p])[0]
        prefix.append(C.V5[int(np.argmin(tails[k]))])
    builder = ObservationBuilder(C.sim.sig.floor(), *C.CAPS)
    obs, fut, t_slot = {}, {}, {}
    replay, _ = ep.play(prefix, capture=_capturer(builder, obs, fut, t_slot))
    missing = _fill(n, obs, fut, t_slot)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, **{k: np.stack([obs[i][k] for i in range(n)]).astype(
        np.float16 if k in ("factory_grid", "job_table") else np.float32) for k in OBS_KEYS},
        tails=tails, schedule=np.array(prefix), pairs=np.array(C.V5), seed=seed, oracle=float(tails[-1].min()),
        replay=float(replay), warmup=ep.warmup, stamp=json.dumps(C.stamp()),
        regimes=np.array([ep.regime_at(ep.warmup + (k + 0.5) * ep.cfg["slot"]) for k in range(n)]),
        future=np.stack([fut[i] for i in range(n)]).astype(np.float32),
        t_slot=np.array([t_slot[i] for i in range(n)]))
    os.replace(tmp, out)
    fixed = dict(zip(C.V5, tails[0]))
    best = min(fixed, key=fixed.get)
    return (f"label s{seed}: best fixed {best} {fixed[best]:.2f}, oracle {tails[-1].min():.2f} "
            f"({100 * (tails[-1].min() / fixed[best] - 1):+.1f}%), replay diff {abs(replay - tails[-1].min()):.1e}, "
            f"missing obs {len(missing)}, {time.time() - t:.0f} s")


def future(seed):
    """Replay the stored rq2-rebase-v5 oracle schedule of an evaluation seed; capture the future table."""
    from des_twin.observation import ObservationBuilder
    out = C.HERE / "out" / "future" / f"s{seed}.npz"
    if out.exists():
        return f"exists {out}"
    t = time.time()
    z = np.load(C.ORACLE_DIR / f"s{seed}.npz")
    ep = C.sim.Episode("B2", seed)
    n = C.sim.n_slots("B2")
    builder = ObservationBuilder(C.sim.sig.floor(), *C.CAPS)
    obs, fut, t_slot = {}, {}, {}
    replay, _ = ep.play(list(z["schedule"]), capture=_capturer(builder, obs, fut, t_slot))
    _fill(n, obs, fut, t_slot)
    if replay != float(z["replay"]):
        raise SystemExit(f"s{seed}: replay {replay} != stored {float(z['replay'])}")
    for k in OBS_KEYS:
        got = np.stack([obs[i][k] for i in range(n)]).astype(z[k].dtype)
        if not np.array_equal(got, z[k]):
            raise SystemExit(f"s{seed}: observation {k} differs from the stored labels")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, future=np.stack([fut[i] for i in range(n)]).astype(np.float32),
                        t_slot=np.array([t_slot[i] for i in range(n)]), seed=seed, replay=replay,
                        stamp=json.dumps(C.stamp()))
    os.replace(tmp, out)
    return f"future s{seed}: replay = stored {replay:.3f}, observations equal, {time.time() - t:.1f} s"


def future_level(level, seed):
    """The evaluation seed's future tables at forecast @p level (common.Published: rq2-perturbed-v5's published
    schedule through eval-v5-lookahead-fcast's adapter), along the stored oracle trajectory -> out/future_<level>/.
    For the per-slot check of whether a clone's pick depends on the table (bc_train.py)."""
    out = C.HERE / "out" / f"future_{level}" / f"s{seed}.npz"
    if out.exists():
        return f"exists {out}"
    t = time.time()
    z = np.load(C.ORACLE_DIR / f"s{seed}.npz")
    ep = C.sim.Episode("B2", seed)
    n = C.sim.n_slots("B2")
    view = C.Published(seed, level)
    fut = {}

    def capture(tw, dec, k):
        fut[k] = view(tw, C.STORE_ROWS)

    replay, _ = ep.play(list(z["schedule"]), capture=capture)
    _fill(n, fut)
    if replay != float(z["replay"]):
        raise SystemExit(f"s{seed}: replay {replay} != stored {float(z['replay'])}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, future=np.stack([fut[i] for i in range(n)]).astype(np.float32), seed=seed, level=level,
                        stamp=json.dumps(C.stamp()))
    os.replace(tmp, out)
    return f"future {level} s{seed}: {time.time() - t:.1f} s"


def main(argv):
    if argv[0] == "list4k":                        # data scaling (10-16): 3,000 more training seeds 4000-6999
        for lo in range(4000, 7000, 10):
            print(f"label {lo}-{lo + 9}")
        return
    if argv[0] == "list":
        for lo in range(C.TRAIN_SEEDS[0], C.TRAIN_SEEDS[-1] + 1, 10):
            print(f"label {lo}-{lo + 9}")
        for lo in range(0, 40, 20):
            print(f"future {lo}-{lo + 19}")
        for lo in range(2000, 2160, 20):
            print(f"future {lo}-{lo + 19}")
        return
    if argv[0] == "future-level":                  # future-level <level> <seeds>
        for s in C.sim.parse_seeds(argv[2]):
            print(future_level(argv[1], s), flush=True)
        C.check_frozen()
        return
    kind, seeds = argv[0], C.sim.parse_seeds(argv[1])
    fn = {"label": label, "future": future}[kind]
    for s in seeds:
        print(fn(s), flush=True)
    C.check_frozen()


if __name__ == "__main__":
    C.check_frozen()
    main(sys.argv[1:])
