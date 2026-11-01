"""
@file run.py
@brief rq3-size-headroom (2026-10-25, user: remaining work plan 10-23 §2c, "why does switching headroom shrink on
       larger floors?"). The greedy switching oracle of ../rq2-rebase-v5 (900 s slots, hold-to-end tails, its frozen
       code and sim.py) over the v5 pairs and over v5 + MDD-TECT, at 50 / 100 / 200 machines x AGVs on duty per machine
       {0.35, 0.52, 0.8}, plus per-slot load and transport descriptors under MDD-PTECT and MDD-TECT held all episode.
       It separates the two untested explanations of the shrinking ceiling: smoother aggregate dynamics on big floors
       (averaging) and transport scarcity (trips get longer with size at a fixed AGV ratio).
Floors: the floor-size grid's (../rq3-size-grid/task.py, through ../rq3-size-oracle/sizesim.py's grid task): m50 /
m100 the agv-ratio-scale exports, m200 the dev-size-floors export, M bays, B2 load x 0.9, arrival cap off. Only the
fleet changes: round(r M) AGVs on duty in full-fleet blocks, half that (rounded) in short-fleet blocks. r 0.52 is the
grid's own condition. The job stream does not depend on r (checked: `check`, and every desc file carries a hash).
  oracle <size> <ratio> <set> <seed>   -> out/m<size>/<ratio>/<set>/s<seed>.json (as ../rq3-size-portfolio)
  desc <size> <ratio> <seed>           -> out/m<size>/<ratio>/desc/s<seed>.json: per-slot state at each slot's first
                                          decision under MDD-PTECT and MDD-TECT held, and their window tardiness
  check <size>                         MDD-PTECT at r 0.52 = the grid's (seeds 0-2); job-stream hash equal at every r
  list                                 task lines; outputs that ../rq3-size-oracle or ../rq3-size-portfolio already
                                       hold (r 0.52) are not rerun, `analyze.py` reads them from there
Outputs are skipped when they exist (rerunning resumes). One task per process: sizesim.apply wraps the scenario
registry in place, and looping over runs in one process leaked memory in ../rq3-alpha-size.
@par Usage
@code{.sh}
python results/rq3-size-headroom/run.py check m100
python results/rq3-size-headroom/run.py desc 100 r035 0
python results/rq3-size-headroom/run.py oracle 200 r080 v5tect 3
python results/rq3-size-headroom/run.py list
@endcode
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
SIZES = [200, 100, 50]                                   # longest first
RATIOS = {"r035": 0.35, "r052": 0.52, "r080": 0.80}
V5 = [f"{j}-{m}" for j in ("SRT", "MDD", "ATC") for m in ("PECT", "PTECT")]
SETS = {"v5": V5, "v5tect": V5 + ["MDD-TECT"]}
SEEDS = list(range(20))                                  # B2 test seeds
DESC_RULES = ("MDD-PTECT", "MDD-TECT")


def reused(size, ratio, name, seed):
    """An oracle output an earlier experiment already holds for this cell (same code, floor and fleet), or None."""
    if ratio != "r052":
        return None
    if name == "v5":
        f = R / "rq3-size-oracle" / "out" / f"m{size}" / f"s{seed}.json"
    elif name == "v5tect" and size == 100:
        f = R / "rq3-size-portfolio" / "out" / "m100" / "v5tect" / f"s{seed}.json"
    else:
        return None
    return f if f.exists() else None


def fleet(size, ratio):
    on = round(RATIOS[ratio] * size)
    return on, round(on / 2)


def load(size, ratio):
    """Frozen code first (rq2-rebase-v5/run.py), then the grid's condition m<size> with the fleet set by `ratio`.
    The same steps as sizesim.apply, with size_params' fleet and the fleet wrapper's on-duty count replaced."""
    spec = importlib.util.spec_from_file_location("rb_run", R / "rq2-rebase-v5" / "run.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    spec = importlib.util.spec_from_file_location("sizesim", R / "rq3-size-oracle" / "sizesim.py")
    ss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss)
    gt = ss.grid_task()
    params, floor, layout, grid_fleet = gt.rt.CONDITIONS[f"m{size}"]
    on, short = fleet(size, ratio)
    assert layout is None and grid_fleet[1] == size, (size, layout, grid_fleet)
    if ratio == "r052":
        assert grid_fleet[0] == on and params == gt.st.size_params(size // 5, on, short), size
    params = gt.st.size_params(size // 5, on, short)
    gt.rt.fleet_wrapper(on, size)
    from scenarios import REGISTRY
    from des_twin import Floor
    gen = REGISTRY["randomized"](21600.0, random_warmup=True, warmup_dispatching_rule=None, agv_move_speed=None,
                                 agv_handshake_duration=None, params_overrides=params)
    rb.sim.scenario = lambda setting, seed: gen(seed)
    fl = Floor.load(str(floor))
    rb.sim.sig.floor = lambda: fl
    return rb, ss


def jobs_hash(sc):
    """md5 of the instance without its AGV fields (jobs, phases, regime blocks less their fleet, machine speeds)."""
    meta = sc["_meta"]
    blocks = [{k: v for k, v in b.items() if k != "agv_count"} for b in meta["blocks"]]
    key = {"jobs": sc["jobs"], "phases": sc["_phases"], "blocks": blocks, "speed": meta["machine_speed"],
           "types": sc["machineTypeLayout"], "op_mean": meta["op_mean_seconds"], "c": meta["due_date_allowance"]}
    return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()


def write(out, obj):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, out)


def run_oracle(size, ratio, name, seed):
    out = HERE / "out" / f"m{size}" / ratio / name / f"s{seed}.json"
    if out.exists():
        return f"exists {out}"
    if reused(size, ratio, name, seed):
        return f"reused {reused(size, ratio, name, seed)}"
    pairs = SETS[name]
    rb, _ = load(size, ratio)
    import numpy as np
    t = time.time()
    ep = rb.sim.Episode("B2", seed)
    n = rb.sim.n_slots("B2")
    prefix, tails = [], np.zeros((n, len(pairs)))
    for k in range(n):
        for i, p in enumerate(pairs):
            tails[k, i] = ep.play(prefix + [p])[0]
        prefix.append(pairs[int(np.argmin(tails[k]))])
    write(out, {"size": size, "ratio": ratio, "fleet": fleet(size, ratio), "set": name, "pairs": pairs, "seed": seed,
                "oracle": float(tails[-1].min()), "schedule": prefix, "fixed": dict(zip(pairs, tails[0].tolist())),
                "tails": tails.tolist(), "warmup": ep.warmup, "jobs_hash": jobs_hash(ep.sc), "stamp": rb.stamp(),
                "seconds": time.time() - t})
    return f"oracle m{size}/{ratio}/{name} s{seed}: {float(tails[-1].min()):.3f} k job-s, {time.time() - t:.0f} s"


def on_duty_seconds(sched, n_agv, a, b):
    """AGV-seconds on duty in [a, b] under the twin's agv_schedule ((start, count) pairs; len(agvs) before the first)."""
    pts = [(-1e18, n_agv)] + [(float(t), int(k)) for t, k in sched] + [(1e18, None)]
    tot = 0.0
    for (t0, k), (t1, _) in zip(pts, pts[1:]):
        lo, hi = max(a, t0), min(b, t1)
        if hi > lo:
            tot += (hi - lo) * k
    return tot


def run_desc(size, ratio, seed):
    out = HERE / "out" / f"m{size}" / ratio / "desc" / f"s{seed}.json"
    if out.exists():
        return f"exists {out}"
    rb, _ = load(size, ratio)
    sig = rb.sim.sig
    t = time.time()
    ep = rb.sim.Episode("B2", seed)
    res = {"size": size, "ratio": ratio, "fleet": fleet(size, ratio), "seed": seed, "warmup": ep.warmup,
           "jobs_hash": jobs_hash(ep.sc), "sched": [list(map(float, e)) for e in ep.sched], "rules": {},
           "stamp": rb.stamp()}
    for rule in DESC_RULES:
        rows = []

        def cap(tw, dec, k):
            f, idle_tot = sig.congestion(tw, 0.0, None)
            loads = tw.all_machine_loads()
            lv = [float(v) for v in loads.values()]
            mbusy = sum(m.busy + (tw.now - m.proc_start if m.proc_start is not None else 0.0) for m in tw.machines)
            rows.append({"k": k, "t": tw.now, "regime": ep.regime_at(tw.now), "on_duty": f["c_agv_on_duty"],
                         "wip": f["c_wip"], "wait_n": f["c_wait_n"], "wait_per_agv": f["c_wait_per_agv"],
                         "queued_per_machine": f["c_queued_per_machine"],
                         "load_per_machine": sum(lv) / len(lv), "load_max": max(lv),
                         "load_sd_machines": float(sum((v - sum(lv) / len(lv)) ** 2 for v in lv) / len(lv)) ** 0.5,
                         "agv_busy_now": f["c_agv_busy_share"], "agv_idle_total": idle_tot,
                         "machine_busy_total": mbusy, "agv_path_total": sum(a.path for a in tw.agvs),
                         "agv_trips_total": sum(a.trips for a in tw.agvs), "n_agv": len(tw.agvs)})

        tard, played = ep.play([rule], capture=cap)
        for a, b in zip(rows, rows[1:]):                 # slot k's rates, from its first decision to slot k+1's
            dt = b["t"] - a["t"]
            duty = on_duty_seconds(ep.sched, a["n_agv"], a["t"], b["t"])
            busy = a["n_agv"] * dt - (b["agv_idle_total"] - a["agv_idle_total"])
            a["agv_util"] = busy / duty if duty > 0 else None
            a["machine_util"] = (b["machine_busy_total"] - a["machine_busy_total"]) / (size * dt) if dt > 0 else None
            trips = b["agv_trips_total"] - a["agv_trips_total"]
            a["trips_per_h"] = trips / dt * 3600.0 if dt > 0 else None
            a["path_per_trip"] = (b["agv_path_total"] - a["agv_path_total"]) / trips if trips > 0 else None
        res["rules"][rule] = {"tard": tard, "slots": rows}
    res["seconds"] = time.time() - t
    write(out, res)
    p, q = (res["rules"][r]["tard"] for r in DESC_RULES)
    return f"desc m{size}/{ratio} s{seed}: MDD-PTECT {p:.2f}, MDD-TECT {q:.2f} k job-s, {res['seconds']:.0f} s"


def hash_of(size, ratio, seed):
    rb, _ = load(size, ratio)
    return jobs_hash(rb.sim.scenario("B2", seed))


def check(size):
    """MDD-PTECT at r 0.52 through this adapter = the grid's value (seeds 0-2); the instance (less its AGV fields) is
    the same at every ratio (seeds 0-2, one process per ratio since the fleet wrapper wraps the registry in place)."""
    bad = 0
    rb, ss = load(size, "r052")
    for s in (0, 1, 2):
        v = rb.sim.Episode("B2", s).play(["MDD-PTECT"])[0]
        g = ss.grid_value(f"m{size}", s)
        ok = g is not None and abs(v - g) <= 1e-6 * max(1.0, abs(g)) + 1e-4
        bad += not ok
        print(f"check m{size} s{s}: MDD-PTECT {v:.6f}, grid {g} -> {'OK' if ok else 'MISMATCH'}", flush=True)
    for s in (0, 1, 2):
        hs = {r: subprocess.run([sys.executable, __file__, "hash", str(size), r, str(s)], capture_output=True,
                                text=True, check=True).stdout.strip() for r in RATIOS}
        ok = len(set(hs.values())) == 1
        bad += not ok
        print(f"check m{size} s{s}: job stream {'identical' if ok else 'DIFFERS'} across {sorted(hs)} {hs}",
              flush=True)
    if bad:
        raise SystemExit(1)


def tasks():
    lines = [f"desc {m} {r} {s}" for m in SIZES for r in RATIOS for s in SEEDS]
    for m in SIZES:
        for name in ("v5tect", "v5"):
            for r in RATIOS:
                lines += [f"oracle {m} {r} {name} {s}" for s in SEEDS if not reused(m, r, name, s)]
    return lines


def main(argv):
    cmd = argv[0]
    if cmd == "list":
        print("\n".join(tasks()))
    elif cmd == "check":
        check(int(argv[1].lstrip("m")))
    elif cmd == "hash":
        print(hash_of(int(argv[1]), argv[2], int(argv[3])))
    elif cmd == "desc":
        print(run_desc(int(argv[1]), argv[2], int(argv[3])), flush=True)
    elif cmd == "oracle":
        print(run_oracle(int(argv[1]), argv[2], argv[3], int(argv[4])), flush=True)
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main(sys.argv[1:])
