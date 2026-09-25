"""
@file randomized.py
@brief Randomized training scenarios: multi-op jobs over the whole floor, per-episode operation length,
       load that rises and falls, failures on in a share of episodes.

@details
The compound scenarios (compound.py) have 1-2 ops per job, a ~31 s mean op, and send ~70% of work to the
Weld machines, so 12 of 15 machines are background and a policy trained on them sees one narrow regime
(docs/features/SCENARIO_GENERATORS.md §9-10). This generator draws each episode from a family instead:

- **Jobs:** 1-5 operations each (uniform), each on a random machine type (no immediate repeat), eligible on
  every machine of that type. Machines of a type differ in speed (a fixed per-episode factor per machine,
  plus ±10% per job), so machine choice (SMPT vs SRWT vs MMUR) matters.
- **Operation length:** the episode's mean op is log-uniform in `op_mean_seconds` (default 150-360 s,
  2.5-6 min). Longer ops than the compound family, deliberately: at ~64 AGV-seconds per transport move
  (measured on compound at 7 AGVs) 7 AGVs move ~0.11 jobs/s at full load, so short ops make the floor
  transport-bound again, the regime where every rule converged.
- **Load:** the arrival stream is a sequence of segments (20-60 min each), each with its own kind and
  target machine utilization, so queues build up and drain within an episode:
    - `balanced`: Poisson arrivals, uniform type mix
    - `skewed`: one machine type gets 2.5x the op share (a bottleneck group with routing choice inside it)
    - `bimodal`: 70% short ops (0.4x mean) and 30% long ops (2.4x mean) — SPT-vs-starvation pressure
    - `burst`: the segment's whole job count arrives in the first 60 s, then nothing
    - `lull`: low utilization (0.1-0.3), recovery
  Target utilization is floor-average (all 15 machines); overload segments (> 1) are allowed. Arrival rates
  are capped so AGV demand stays under `agv_max_utilization` of the fleet: the regime stays machine-bound.
- **Failures:** on for a seeded `failure_probability` share of episodes (default 2/3), rarer and longer than
  the E1 setting: Weibull k=1.5, λ=7200 s (MTBF ≈ 6500 s ≈ 1.8 h per machine, so about one failure every
  7 min floor-wide at 15 machines) and lognormal repair μ=6.3, σ=0.5 (mean ≈ 620 s ≈ 10 min, several ops long,
  so rerouting around a failure pays off). Availability ≈ 91%.
- **Floor:** 15 machines (3 per type), 7 AGVs, layout D, releasePrevious, lane parking, set in the scenario
  JSON so they do not depend on player CLI overrides.

Every instance is a pure function of its seed. `_phases` lists the segments (for random warm-up cutoffs and
analysis); `_meta` records the per-episode draws.

@par Usage
@code{.sh}
python env/train.py --unity --unity-path linux_server/capstone.x86_64 --no-graphics \\
    --scenario-generator randomized --episode-duration-seconds 5400 --random-warmup \\
    --reward-spec env/config/rewards/flow_time.json --train-seed 0 ...
python env/scenarios/randomized.py --write 0-8 --out linux_server/BatchConfigs/Scenarios/randomized
@endcode
"""

import json
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple

TYPES = ["Mill", "Lathe", "Weld", "Inspect", "Assemble"]

## @brief Rules the warm-up rotates through when no warm-up rule is given (the stronger PDRs), so episode
##        start states are not all shaped by one rule.
WARMUP_RULES = ["SPT_SMPT", "SPT_SRWT", "SRT_SRWT", "SRT_SMPT"]

SEGMENT_KINDS = ("balanced", "skewed", "bimodal", "burst", "lull")


@dataclass(frozen=True)
class RandomizedParams:
    """@brief The generator's knobs. Defaults are the first training run's configuration (2026-09-24)."""

    machines_per_type: int = 3
    agv_count: int = 7
    layout: str = "D"
    reservation_protocol: str = "releasePrevious"
    parking_method: str = "lane"
    routing_trigger: str = "onTransport"                 # see RoutingTrigger.cs; "onReady" = legacy

    horizon_seconds: float = 14400.0                     # span of arrivals (sim-seconds)
    segment_seconds: Tuple[float, float] = (1200.0, 3600.0)
    segment_weights: Tuple[float, ...] = (0.35, 0.2, 0.15, 0.1, 0.2)   # SEGMENT_KINDS order
    utilization: Tuple[float, float] = (0.4, 1.25)        # floor-average target, non-lull segments
    lull_utilization: Tuple[float, float] = (0.1, 0.3)
    op_mean_seconds: Tuple[float, float] = (150.0, 360.0)  # per-episode mean op, log-uniform
    op_sigma: float = 0.35                               # lognormal spread of a single op around the mean
    ops_per_job: Tuple[int, int] = (1, 5)
    machine_speed_spread: float = 0.25                   # per-machine factor log-uniform in [1/1.25, 1.25]
    op_machine_spread: float = 0.10                      # per-op, per-machine factor uniform in [1-x, 1+x]
    skew_weight: float = 2.5
    burst_window_seconds: float = 60.0

    agv_seconds_per_move: float = 64.0                   # measured: AGV busy time per transport move
    agv_max_utilization: float = 0.75

    failure_probability: float = 2.0 / 3.0
    failures: Dict = field(default_factory=lambda: {
        "machineFailuresEnabled": True, "weibullK": 1.5, "weibullLambda": 7200.0,
        "repairLogMu": 6.3, "repairLogSigma": 0.5,
    })


DEFAULT_PARAMS = RandomizedParams()


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _op_types(rng: random.Random, n: int, weights: List[float]) -> List[str]:
    """@brief n machine types drawn by weight, never the same type twice in a row."""
    out: List[str] = []
    for _ in range(n):
        choices = [(t, w) for t, w in zip(TYPES, weights) if not out or t != out[-1]]
        out.append(rng.choices([t for t, _ in choices], [w for _, w in choices])[0])
    return out


def randomized_scenario(seed: int, params: RandomizedParams = DEFAULT_PARAMS) -> Dict:
    """@brief One seeded instance (ScenarioLoader schema), without episode cap or warm-up.

    @param seed    Instance seed; the same seed always returns an identical scenario.
    @param params  Generator knobs (RandomizedParams).
    """
    p = params
    rng = random.Random(seed)
    k = p.machines_per_type
    n_machines = k * len(TYPES)

    op_mean = _log_uniform(rng, *p.op_mean_seconds)
    s = 1.0 + p.machine_speed_spread
    speed = {t: [_log_uniform(rng, 1.0 / s, s) for _ in range(k)] for t in TYPES}
    failures_on = rng.random() < p.failure_probability

    mean_ops = (p.ops_per_job[0] + p.ops_per_job[1]) / 2.0
    # AGV moves per job: one per op plus the move to the output belt.
    transport_cap = p.agv_count * p.agv_max_utilization / p.agv_seconds_per_move   # jobs-moves per second

    jobs: List[Dict] = []
    phases: List[Dict] = []
    t = 0.0
    while t < p.horizon_seconds:
        dur = min(rng.uniform(*p.segment_seconds), p.horizon_seconds - t)
        if dur < 300.0:           # don't leave a sliver segment at the end
            break
        kind = rng.choices(SEGMENT_KINDS, p.segment_weights)[0]
        u = rng.uniform(*(p.lull_utilization if kind == "lull" else p.utilization))

        weights = [1.0] * len(TYPES)
        skew_type = None
        if kind == "skewed":
            skew_type = rng.choice(TYPES)
            weights[TYPES.index(skew_type)] = p.skew_weight
        # bimodal keeps the mean op length: 0.7 * 0.4 + 0.3 * 2.4 = 1.0
        mean_work_per_job = mean_ops * op_mean
        job_rate = u * n_machines / mean_work_per_job                       # jobs per second
        transport_capped = job_rate * (mean_ops + 1.0) > transport_cap
        if transport_capped:
            job_rate = transport_cap / (mean_ops + 1.0)
        n_jobs = max(1, int(round(job_rate * dur)))

        if kind == "burst":
            arrivals = sorted(t + rng.uniform(0.0, p.burst_window_seconds) for _ in range(n_jobs))
        else:
            arrivals, a = [], t
            while True:
                a += rng.expovariate(job_rate)
                if a >= t + dur:
                    break
                arrivals.append(a)

        for arrival in arrivals:
            n_ops = rng.randint(*p.ops_per_job)
            ops = []
            for mtype in _op_types(rng, n_ops, weights):
                if kind == "bimodal":
                    base = op_mean * (0.4 if rng.random() < 0.7 else 2.4)
                else:
                    base = op_mean
                d = base * math.exp(rng.gauss(0.0, p.op_sigma) - p.op_sigma ** 2 / 2.0)
                durations = [round(d * speed[mtype][m] * rng.uniform(1.0 - p.op_machine_spread, 1.0 + p.op_machine_spread), 2)
                             for m in range(k)]
                ops.append({"machineType": mtype, "machineIndex": list(range(k)), "duration": durations})
            jobs.append({"id": len(jobs), "arrivalTime": round(arrival, 2), "operations": ops})

        phases.append({"name": f"{kind}_{len(phases)}", "kind": kind, "start": round(t, 2),
                       "end": round(t + dur, 2), "n_jobs": len(arrivals), "utilization": round(u, 3),
                       "skew_type": skew_type, "transport_capped": transport_capped})
        t += dur

    scenario = {
        "_comment": (f"Randomized training instance seed={seed}: {len(jobs)} jobs over {t:.0f}s in "
                     f"{len(phases)} segments, mean op {op_mean:.0f}s, failures {'on' if failures_on else 'off'}. "
                     f"See env/scenarios/randomized.py."),
        "name": "randomized",
        "seed": seed,
        "agvCount": p.agv_count,
        "machineTypeLayout": [t_ for t_ in TYPES for _ in range(k)],
        "layout": p.layout,
        "reservationProtocol": p.reservation_protocol,
        "parkingMethod": p.parking_method,
        "routingTrigger": p.routing_trigger,
        "jobs": jobs,
        "_phases": phases,
        "_meta": {
            "op_mean_seconds": round(op_mean, 2),
            "machine_speed": {t_: [round(x, 3) for x in v] for t_, v in speed.items()},
            "failures_on": failures_on,
            "params": json.loads(json.dumps(asdict(p))),   # JSON-native (tuples -> lists)
        },
    }
    if failures_on:
        scenario["stochastic"] = dict(p.failures)
    return scenario


def warmup_offsets(scenario: Dict, episode_duration_seconds: Optional[float] = None) -> List[float]:
    """@brief Segment starts usable as warm-up cutoffs: 0 and every later segment start that still
    leaves a full agent window of arrivals (when a window is given)."""
    last_arrival = max(j["arrivalTime"] for j in scenario["jobs"])
    limit = last_arrival - (episode_duration_seconds or 0.0)
    starts = [ph["start"] for ph in scenario["_phases"] if ph["start"] <= limit]
    return starts or [0.0]


def randomized_variant(seed: int, episode_duration_seconds: Optional[float] = None,
                       warmup_seconds: Optional[float] = None,
                       warmup_dispatching_rule: Optional[str] = None,
                       agv_move_speed: Optional[float] = None,
                       agv_handshake_duration: Optional[float] = None,
                       params: RandomizedParams = DEFAULT_PARAMS) -> Dict:
    """@brief randomized_scenario plus the episode-control fields compound_variant also supports.

    @param episode_duration_seconds  Agent-window cap (sim-seconds after the warm-up), truncated not terminated.
    @param warmup_seconds  Heuristic warm-up before the agent takes over (see compound.compound_variant).
    @param warmup_dispatching_rule  Rule for the warm-up. None rotates through WARMUP_RULES by seed.
    """
    scenario = randomized_scenario(seed, params)
    stochastic = dict(scenario.get("stochastic", {}))
    if episode_duration_seconds is not None:
        stochastic["episodeDurationSeconds"] = float(episode_duration_seconds)
    if warmup_seconds is not None:
        stochastic["warmupSeconds"] = float(warmup_seconds)
    if stochastic:
        scenario["stochastic"] = stochastic
    scenario["dispatchingRule"] = warmup_dispatching_rule or WARMUP_RULES[seed % len(WARMUP_RULES)]
    if agv_move_speed is not None:
        scenario["agvMoveSpeed"] = float(agv_move_speed)
    if agv_handshake_duration is not None:
        scenario["agvHandshakeDuration"] = float(agv_handshake_duration)
    return scenario


def randomized_generator(episode_duration_seconds: Optional[float] = None, random_warmup: bool = False,
                         warmup_dispatching_rule: Optional[str] = None,
                         agv_move_speed: Optional[float] = None,
                         agv_handshake_duration: Optional[float] = None,
                         params: RandomizedParams = DEFAULT_PARAMS) -> Callable[[int], Dict]:
    """@brief seed -> scenario callable for UnitySchedulingEnv / VectorizedUnityEnv (same signature as
    compound_generator). With random_warmup, each episode's warm-up cutoff is a segment start drawn from
    the episode's own seed, so the agent starts mid-stream with realistic WIP."""
    def _generate(seed: int) -> Dict:
        warmup = None
        if random_warmup:
            base = randomized_scenario(seed, params)
            warmup = random.Random(seed ^ 0x5EED).choice(warmup_offsets(base, episode_duration_seconds))
        return randomized_variant(seed, episode_duration_seconds, warmup, warmup_dispatching_rule,
                                  agv_move_speed, agv_handshake_duration, params)
    return _generate


def summarize(scenario: Dict) -> Dict:
    """@brief Load statistics of an instance (for tests, docs and sanity checks)."""
    jobs = scenario["jobs"]
    n_machines = len(scenario["machineTypeLayout"])
    ops = [op for j in jobs for op in j["operations"]]
    work_by_type: Dict[str, float] = {}
    for op in ops:
        d = sum(op["duration"]) / len(op["duration"])
        work_by_type[op["machineType"]] = work_by_type.get(op["machineType"], 0.0) + d
    total = sum(work_by_type.values())
    span = scenario["_phases"][-1]["end"]
    return {
        "jobs": len(jobs),
        "ops_per_job": len(ops) / max(len(jobs), 1),
        "mean_op_seconds": total / max(len(ops), 1),
        "work_share": {t: work_by_type.get(t, 0.0) / total for t in TYPES},
        "mean_utilization": total / (n_machines * span),
        "moves_per_second": (len(ops) + len(jobs)) / span,
        "segments": len(scenario["_phases"]),
        "failures_on": scenario["_meta"]["failures_on"],
    }


def _parse_seeds(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Write randomized instances as ScenarioLoader JSON (for PDR sweeps).")
    ap.add_argument("--write", required=True, help="seeds, e.g. 0-8 or 0,3,5")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--prefix", default="randomized", help="file / scenario name prefix: <prefix>_s<seed>.json")
    ap.add_argument("--machine-speed-spread", type=float, default=DEFAULT_PARAMS.machine_speed_spread)
    ap.add_argument("--op-machine-spread", type=float, default=DEFAULT_PARAMS.op_machine_spread)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from dataclasses import replace
    params = replace(DEFAULT_PARAMS, machine_speed_spread=args.machine_speed_spread,
                     op_machine_spread=args.op_machine_spread)
    for seed in _parse_seeds(args.write):
        sc = randomized_variant(seed, params=params)
        sc["name"] = f"{args.prefix}_s{seed}"
        path = os.path.join(args.out, f"{sc['name']}.json")
        with open(path, "w") as f:
            json.dump(sc, f, indent=1)
        st = summarize(sc)
        print(f"{path}: {st['jobs']} jobs, {st['ops_per_job']:.2f} ops/job, mean op {st['mean_op_seconds']:.0f}s, "
              f"util {st['mean_utilization']:.2f}, failures {'on' if st['failures_on'] else 'off'}")
