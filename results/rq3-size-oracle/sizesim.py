"""
@file sizesim.py
@brief Floor-size adapter for the frozen v5 oracle simulator (../rq2-rebase-v5/sim.py, the simulator of the switching
       oracle and of the 84-pair catalog screen ../rq2-portfolio-screen). It plays the B2 window and slots on a
       condition of the floor-size grid (../rq3-size-grid/task.py: m15 = B2 itself; m30 = the dev-size-floors export;
       m50 / m100 = the agv-ratio-scale floors; round(0.52 M) AGVs on duty through rq3-gen-v5's fleet wrapper, B2 load
       x 0.9, arrival cap off), with each instance from evaluate.py's own generator call, so the oracle sees exactly
       the instances the grid scores policies on (checked: `check` in run.py).
@details Call apply() once per process, after the frozen code is first on sys.path (import ../rq2-rebase-v5/run.py
         or ../rq2-portfolio-screen/run.py first): the fleet wrapper wraps scenarios.REGISTRY["randomized"] in place.
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_applied = []


def grid_task():
    spec = importlib.util.spec_from_file_location("size_grid_task", HERE.parent / "rq3-size-grid" / "task.py")
    gt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gt)
    return gt


def register(gt, cond):
    """A floor size the grid does not define (e.g. m1000): the grid's per-machine scaling on the dev-size-floors
    export, M / 5 machines per type, round(0.52 M) AGVs on duty, short blocks half (as ../rq3-size-m1000/task.py)."""
    m = int(cond.lstrip("m"))
    floor = gt.FLOORS / cond / "des_floor.json"
    if not floor.exists():
        raise FileNotFoundError(floor)
    fleet = round(0.52 * m)
    gt.rt.CONDITIONS[cond] = (gt.st.size_params(m // 5, fleet, round(fleet / 2)), floor, None, (fleet, m))


def apply(sim, cond):
    """Point sim (a loaded frozen sim.py module) at condition `cond` of the floor-size grid."""
    if _applied:
        raise RuntimeError(f"sizesim.apply already called for {_applied[0]} in this process")
    _applied.append(cond)
    gt = grid_task()
    if cond not in gt.rt.CONDITIONS:
        register(gt, cond)
    params, floor, layout, fleet = gt.rt.CONDITIONS[cond]
    if layout:
        raise ValueError(f"{cond}: layout overrides are not supported here")
    if fleet is not None:
        gt.rt.fleet_wrapper(*fleet)
    from scenarios import REGISTRY
    from des_twin import Floor
    gen = REGISTRY["randomized"](21600.0, random_warmup=True, warmup_dispatching_rule=None, agv_move_speed=None,
                                 agv_handshake_duration=None, params_overrides=params)
    sim.scenario = lambda setting, seed: gen(seed)          # sim.Episode looks scenario() up at call time
    fl = Floor.load(str(floor))
    sim.sig.floor = lambda: fl                              # sim.Episode / play use sig.floor()
    return gt


def grid_value(cond, seed, policy="MDD-PTECT"):
    """The grid's window tardiness (1000 job-s) for a rule on (cond, seed), from its pdr runs, or None."""
    import pandas as pd
    R = HERE.parent
    if cond == "m15":
        files = [R / "rq3-gen-v5" / "base" / "pdr" / "episodes.csv"]
    elif cond in ("m50", "m100"):
        d = R / "rq3-size-v5" / cond
        files = [d / "pdr" / "episodes.csv"] + sorted(d.glob("pdr_s*-*/episodes.csv"))
    else:
        files = sorted((R / "rq3-size-grid" / cond).glob("pdr_s*-*/episodes.csv"))
    for f in files:
        if f.exists():
            e = pd.read_csv(f, dtype={"config_hash": str, "instance_hash": str})
            r = e[(e.policy == policy) & (e.seed == seed)]
            if len(r):
                return float(r.window_tardiness.iloc[0]) / 1000.0
    return None
