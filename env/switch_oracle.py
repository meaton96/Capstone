"""
@file switch_oracle.py
@brief Greedy rule-switching oracle: how much can switching (job rule, machine rule) pairs within an
       episode gain over the best fixed pair, with every switch actually simulated?

@details
The episode is split into segments of --segment-seconds simulated time. Stage 1 runs all 12 pairs for
the whole episode (the fixed-pair baselines). Stage k keeps the pairs chosen for segments 1..k-1, tries
each of the 12 pairs from segment k to the end of the episode, and keeps the best. The pair chosen at
stage k-1 is always among stage k's candidates, so the result never gets worse than the best fixed pair
on the seed. Every candidate is a real run, so the final schedule is an achievable result for that
instance (a lower bound on what per-decision switching could do), not an upper-bound estimate like the
earlier segment oracle.

Scoring is the reward's own quantity: the episode return of --reward-spec (flow_time: -(time in system
of every job, finished or not) / 1000). Each stage runs in a fresh player (12 episodes, about 65,000
simulated seconds), which keeps Unity's float32 clock below 131,072 s (AGVController timers; see
docs/experiments/rq4-agvfail_findings_1001.md section 3), so replayed prefixes are reproduced exactly.

Outputs in --out: stages.csv (every candidate run) and result.json (the chosen schedule and the gains).

@par Usage (one seed per call; run seeds in parallel from a shell loop)
@code{.sh}
python env/switch_oracle.py --unity-path linux_server/capstone.x86_64 --seed 3 \
    --scenario-generator randomized --episode-duration-seconds 5400 --segment-seconds 900 \
    --reward-spec env/config/rewards/flow_time.json --base-worker-id 500 --out results/rq2-switch-oracle/s3
@endcode
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import pdr_action
from env_wrappers.unity_env import UnitySchedulingEnv
from evaluate import resolve_pdr_names, run_evaluation


class ScheduledPolicy:
    """@brief Plays a fixed rule pair per time segment: prefix[i] in segment i, then @p tail to the end."""

    kind = "pdr"
    last_probs = None

    def __init__(self, env, prefix: list, tail: str, segment_seconds: float):
        self.env = env
        self._t0 = None   # sim time of this episode's first decision (after any warm-up); segments count from it
        self.prefix = list(prefix)
        self.tail = tail
        self.segment_seconds = segment_seconds
        self.name = "|".join(self.prefix + [tail])
        self._actions = {name: tuple(pdr_action(name)) for name in set(self.prefix + [tail])}

    def __call__(self, obs):
        sim_time = float(self.env.current_metrics.sim_time)
        if self._t0 is None:
            self._t0 = sim_time
        segment = int((sim_time - self._t0) // self.segment_seconds)
        name = self.prefix[segment] if segment < len(self.prefix) else self.tail
        return self._actions[name]


def run_stage(args, scenario_generator, reward_fn, obs_caps, prefix, pairs, stage, out):
    """@brief One fresh player, one episode per candidate tail pair; returns the episode rows."""
    env = UnitySchedulingEnv(
        file_name=args.unity_path,
        obs_caps=obs_caps,
        reward_fn=reward_fn,
        time_scale=args.time_scale,
        worker_id=args.base_worker_id,
        no_graphics=True,
        decision_drain=True,
        log_file=out / f"Player_stage{stage}.log",
    )
    try:
        policies = [ScheduledPolicy(env, prefix, tail, args.segment_seconds) for tail in pairs]
        schedule = [(p, args.seed) for p in range(len(policies))]
        rows = run_evaluation(env, policies, schedule, log=lambda line: print(f"[stage {stage}] {line}", flush=True),
                              scenario_generator=scenario_generator)
    finally:
        env.close()
    for row in rows:
        row["stage"] = stage
        row["tail"] = row["policy"].split("|")[-1]
    return rows


def build_generator(args):
    """@brief seed -> scenario for the regime the arguments describe (generator, params, floor, AGV breakdowns)."""
    from scenarios import REGISTRY, with_agv_failures, with_floor
    if args.params:
        if args.scenario_generator != "randomized":
            raise SystemExit("--params applies to the randomized generator only")
        import dataclasses
        from scenarios.randomized import DEFAULT_PARAMS, randomized_generator
        overrides = json.loads(args.params)
        fields = {f.name: f for f in dataclasses.fields(DEFAULT_PARAMS)}
        unknown = sorted(set(overrides) - set(fields))
        if unknown:
            raise SystemExit(f"--params: unknown RandomizedParams fields {unknown}")
        overrides = {k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
        params = dataclasses.replace(DEFAULT_PARAMS, **overrides)
        generator = randomized_generator(args.episode_duration_seconds, random_warmup=args.random_warmup,
                                         params=params)
    else:
        generator = REGISTRY[args.scenario_generator](args.episode_duration_seconds,
                                                      random_warmup=args.random_warmup)
    if args.agv_failures is not None:
        generator = with_agv_failures(generator, json.loads(args.agv_failures))
    if args.agvs is not None or args.layout is not None:
        generator = with_floor(generator, args.agvs, args.layout)
    return generator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("@par")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unity-path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pdr", default="all", help="Candidate pairs (evaluate.py --pdr syntax)")
    parser.add_argument("--scenario-generator", required=True)
    parser.add_argument("--episode-duration-seconds", type=float, required=True)
    parser.add_argument("--segment-seconds", type=float, default=900.0)
    parser.add_argument("--reward-spec", required=True)
    parser.add_argument("--random-warmup", action="store_true",
                        help="Start each episode mid-stream after a heuristic warm-up (the training distribution)")
    parser.add_argument("--params", default=None, metavar="JSON",
                        help='RandomizedParams overrides, e.g. \'{"utilization": [1.0, 1.8], "ops_per_job": [3, 8]}\'')
    parser.add_argument("--agvs", type=int, default=None)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--agv-failures", default=None, metavar="JSON",
                        help='AGV breakdowns on, with overrides, e.g. \'{"agvWeibullLambda": 1500}\'')
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--base-worker-id", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from scenarios import row_caps_for
    from rewards import load_reward
    scenario_generator = build_generator(args)
    reward_fn = load_reward(args.reward_spec).build()
    obs_caps = row_caps_for([scenario_generator(args.seed)], 0, 0)
    pairs = resolve_pdr_names(args.pdr)
    n_segments = math.ceil(args.episode_duration_seconds / args.segment_seconds)

    prefix, all_rows, history = [], [], []
    start = time.time()
    for stage in range(1, n_segments + 1):
        rows = run_stage(args, scenario_generator, reward_fn, obs_caps, prefix, pairs, stage, out)
        for row in rows:
            row["prefix"] = "|".join(prefix)
        all_rows.extend(rows)
        best = max(rows, key=lambda r: r["return"])
        history.append({"stage": stage, "chosen": best["tail"], "return": best["return"],
                        "total_flow_time": best["total_flow_time"], "mean_flow_time": best["mean_flow_time"],
                        "jobs_exited": best["jobs_exited"]})
        print(f"[stage {stage}] chosen {best['tail']}  return {best['return']:.3f}  ({time.time() - start:.0f}s)",
              flush=True)
        prefix.append(best["tail"])

    fields = ["stage", "prefix", "tail", "policy", "seed", "return", "total_flow_time", "mean_flow_time",
              "jobs_exited", "makespan", "decisions", "deadlock", "timed_out", "truncated", "machine_failures",
              "config_hash", "instance_hash"]
    with open(out / "stages.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    fixed = {r["tail"]: r for r in all_rows if r["stage"] == 1}
    best_fixed = max(fixed.values(), key=lambda r: r["return"])
    final = history[-1]
    # Returns are negative (penalty), so time in system = -return * time_scale; gaps are on that quantity.
    def tis(ret):
        return -ret
    result = {
        "seed": args.seed,
        "regime": {"generator": args.scenario_generator, "params": args.params, "random_warmup": args.random_warmup,
                   "agvs": args.agvs, "layout": args.layout, "agv_failures": args.agv_failures},
        "segment_seconds": args.segment_seconds,
        "episode_duration_seconds": args.episode_duration_seconds,
        "schedule": prefix,
        "stages": history,
        "best_fixed_pair": best_fixed["tail"],
        "best_fixed_return": best_fixed["return"],
        "oracle_return": final["return"],
        "gain_vs_best_fixed_pct": 100.0 * (tis(final["return"]) / tis(best_fixed["return"]) - 1.0),
        "fixed_returns": {k: v["return"] for k, v in fixed.items()},
    }
    with open(out / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"seed {args.seed}: best fixed {best_fixed['tail']} {best_fixed['return']:.3f}, "
          f"oracle {final['return']:.3f} ({result['gain_vs_best_fixed_pct']:+.2f}%), schedule {prefix}")


if __name__ == "__main__":
    main()
