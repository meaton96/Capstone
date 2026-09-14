"""
@file evaluate.py
@brief Compare trained checkpoints and fixed PDR rules on the same reproducible instances.

@details
Every (policy, seed) pair is one episode. All seeds are queued in Unity up front, and
Unity reports which queue position each episode consumed (episode_seed_index), so each
episode is attributed to exactly one policy even though episodes roll over inside Unity.
The episode already running when evaluation starts (index -1) is discarded.

PDR baselines run as constant-action policies through the same wrapper and decision path
as the learned policy, so the comparison differs only in the actions chosen. Evaluation
seeds should stay below unity_env.TRAIN_SEED_LOW so they never coincide with training
instances.

Outputs in --out: episodes.csv (one row per episode), summary.csv, and the Unity log.

@par Usage
@code{.sh}
python env/evaluate.py --unity-path linux_server/capstone.x86_64 --no-graphics \
    --seeds 0-19 --pdr all --checkpoint results/reward_smoke02/checkpoint.pt \
    --device cuda --out results/eval_smoke02
@endcode
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ActorCriticConfig, EncoderConfig, FusionConfig, PDR_ACTIONS
from env_wrappers.unity_env import TRAIN_SEED_LOW, UnitySchedulingEnv

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_seeds(spec: str) -> list:
    """@brief Parse "0-19", "3,7,11", or a mix such as "0-4,10" into a list of seeds."""
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = (int(x) for x in part.split("-", 1))
            if high < low:
                raise ValueError(f"Bad seed range {part!r}")
            seeds.extend(range(low, high + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("No seeds given")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds")
    return seeds


def resolve_pdr_names(spec: str) -> list:
    """@brief "all", "none", or a comma list of rule names (case and -/_ insensitive)."""
    spec = spec.strip()
    if spec.lower() == "all":
        return list(PDR_ACTIONS)
    if spec.lower() in ("", "none"):
        return []
    names = []
    for part in spec.split(","):
        name = part.strip().upper().replace("_", "-")
        if name not in PDR_ACTIONS:
            raise ValueError(f"Unknown PDR rule {part.strip()!r}; choose from {', '.join(PDR_ACTIONS)}")
        names.append(name)
    return names


class ConstantPolicy:
    """@brief A fixed dispatching rule: always the same action index."""

    kind = "pdr"

    def __init__(self, action: int, name: str):
        self.action = action
        self.name = name

    def __call__(self, obs) -> int:
        return self.action


class CheckpointPolicy:
    """@brief A trained SchedulingNetwork loaded from a train.py checkpoint."""

    kind = "checkpoint"

    def __init__(self, path, device: str = "cpu", deterministic: bool = True):
        from models.network import SchedulingNetwork

        path = Path(path)
        checkpoint = torch.load(path, map_location=device)
        self.net = SchedulingNetwork(EncoderConfig(), FusionConfig(), ActorCriticConfig()).to(device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval()
        self.device = device
        self.deterministic = deterministic
        self.name = f"ckpt:{path.parent.name}/{path.stem}"

    def __call__(self, obs) -> int:
        obs_t = {k: torch.tensor(v[None], dtype=torch.float32, device=self.device) for k, v in obs.items()}
        with torch.no_grad():
            action, _, _ = self.net.act(obs_t, deterministic=self.deterministic)
        return int(action.item())


def build_policies(pdr_spec: str, checkpoints, device: str, deterministic: bool) -> list:
    policies = [ConstantPolicy(PDR_ACTIONS.index(name), name) for name in resolve_pdr_names(pdr_spec)]
    policies += [CheckpointPolicy(path, device, deterministic) for path in checkpoints or []]

    seen = {}
    for policy in policies:
        seen[policy.name] = seen.get(policy.name, 0) + 1
        if seen[policy.name] > 1:
            policy.name = f"{policy.name}#{seen[policy.name]}"
    return policies


def run_evaluation(env, policies: list, schedule: list, log=print) -> list:
    """@brief Play one episode per (policy index, seed) in @p schedule, in order.

    @param env       A @ref UnitySchedulingEnv (or anything with queue_seeds / reset / step /
                     current_metrics).
    @param schedule  List of (policy index, seed); position i is queue index i in Unity.
    @return One row dict per completed episode, in schedule order.
    """
    total = len(schedule)
    env.queue_seeds([seed for _, seed in schedule], clear=True)
    obs = env.reset()
    if env.current_metrics is None:
        raise RuntimeError("This Unity build has no reward-metrics sensor; rebuild the player.")

    rows, discarded, start = [], 0, time.time()
    while len(rows) < total:
        index = int(env.current_metrics.episode_seed_index)
        # The episode that was already running before our seeds were queued (index -1) is
        # played out with action 0 and discarded.
        policy = policies[schedule[index][0]] if 0 <= index < total else None
        obs, _, done, info = env.step(policy(obs) if policy is not None else 0)
        if not done:
            continue

        episode = info["episode"]
        index = episode["seed_index"]
        if not 0 <= index < total:
            discarded += 1
            if discarded > 2:
                raise RuntimeError("Unity keeps starting unseeded episodes; the seed queue is not being applied.")
            continue

        policy_index, seed = schedule[index]
        if episode["seed"] != seed:
            raise RuntimeError(f"Episode at queue index {index} used seed {episode['seed']}, expected {seed}")
        policy = policies[policy_index]
        rows.append({
            "policy": policy.name,
            "kind": policy.kind,
            "seed": seed,
            "seed_index": index,
            "makespan": episode["makespan"],
            "mean_flow_time": episode["mean_flow_time"],
            "total_flow_time": episode["total_flow_time"],
            "return": episode["return"],
            "decisions": episode["length"],
            "jobs_exited": episode["jobs_exited"],
            "deadlock": episode["deadlock"],
            "timed_out": episode["timed_out"],
        })
        log(f"[{len(rows):4d}/{total}] seed {seed:5d}  {policy.name:28s} "
            f"makespan {episode['makespan']:7.1f}  total flow {episode['total_flow_time']:8.1f}  "
            f"({time.time() - start:5.0f}s)")
    return rows


def summarize(rows: list) -> list:
    """@brief Per-policy means, plus paired gaps against the best PDR rule on each seed.

    @details The gap for a policy is the mean over seeds of (policy − best PDR on that seed) /
    best PDR on that seed, in percent: negative beats every rule on that instance, 0 matches
    the per-seed oracle choice among the rules.
    """
    best = {}
    for row in rows:
        if row["kind"] == "pdr":
            makespan, flow = best.get(row["seed"], (math.inf, math.inf))
            best[row["seed"]] = (min(makespan, row["makespan"]), min(flow, row["total_flow_time"]))

    by_policy = {}
    for row in rows:
        by_policy.setdefault(row["policy"], []).append(row)

    summary = []
    for name, policy_rows in by_policy.items():
        makespans = np.array([r["makespan"] for r in policy_rows])
        entry = {
            "policy": name,
            "kind": policy_rows[0]["kind"],
            "episodes": len(policy_rows),
            "makespan_mean": float(makespans.mean()),
            "makespan_std": float(makespans.std()),
            "total_flow_mean": float(np.mean([r["total_flow_time"] for r in policy_rows])),
            "mean_flow_time_mean": float(np.mean([r["mean_flow_time"] for r in policy_rows])),
            "deadlocks": int(sum(bool(r["deadlock"]) for r in policy_rows)),
            "timeouts": int(sum(bool(r["timed_out"]) for r in policy_rows)),
        }
        paired = [r for r in policy_rows if r["seed"] in best]
        if paired:
            entry["makespan_gap_pct"] = float(100 * np.mean(
                [(r["makespan"] - best[r["seed"]][0]) / best[r["seed"]][0] for r in paired]))
            entry["flow_gap_pct"] = float(100 * np.mean(
                [(r["total_flow_time"] - best[r["seed"]][1]) / best[r["seed"]][1] for r in paired]))
        summary.append(entry)
    return sorted(summary, key=lambda e: e["makespan_mean"])


def print_summary(summary: list):
    def gap(value):
        return f"{value:+7.2f}" if value is not None else " " * 7

    header = (f"{'policy':30s} {'n':>4s} {'makespan':>17s} {'gap%':>7s} "
              f"{'total flow':>11s} {'gap%':>7s} {'deadlk':>6s}")
    print()
    print(header)
    print("-" * len(header))
    for e in summary:
        print(f"{e['policy']:30s} {e['episodes']:4d} {e['makespan_mean']:9.1f} ±{e['makespan_std']:6.1f} "
              f"{gap(e.get('makespan_gap_pct'))} {e['total_flow_mean']:11.1f} {gap(e.get('flow_gap_pct'))} "
              f"{e['deadlocks']:6d}")
    print("gap% = mean per-seed gap to the best PDR rule on that seed (lower is better).")


def write_csv(path: Path, rows: list):
    if not rows:
        return
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate policies on reproducible instances")
    parser.add_argument("--unity-path", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0-19",
                        help="Evaluation seeds, e.g. '0-19' or '1,5,9' (keep below %d)" % TRAIN_SEED_LOW)
    parser.add_argument("--pdr", type=str, default="all",
                        help="PDR baselines: 'all', 'none', or a comma list of rule names")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="train.py checkpoint to evaluate (repeatable)")
    parser.add_argument("--stochastic-policy", action="store_true",
                        help="Sample checkpoint actions instead of taking the argmax")
    parser.add_argument("--reward-spec", type=str, default=None,
                        help="Also report episode return under this reward spec")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "results" / f"eval-{datetime.now():%Y%m%d-%H%M%S}"))
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--no-graphics", action="store_true")
    parser.add_argument("--no-decision-drain", action="store_true")
    parser.add_argument("--base-worker-id", type=int, default=0)
    args = parser.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    if max(seeds) >= TRAIN_SEED_LOW:
        print(f"Warning: seeds >= {TRAIN_SEED_LOW} can coincide with training instances.")

    policies = build_policies(args.pdr, args.checkpoint, args.device, not args.stochastic_policy)
    if not policies:
        parser.error("Nothing to evaluate: pass --checkpoint and/or --pdr")

    # Seed-major order, so an interrupted run still has every policy on the seeds it finished.
    schedule = [(p, seed) for seed in seeds for p in range(len(policies))]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {len(policies)} policies × {len(seeds)} seeds = {len(schedule)} episodes -> {out}")

    reward_fn = None
    if args.reward_spec:
        from rewards import load_reward
        reward_fn = load_reward(args.reward_spec).build()

    env = UnitySchedulingEnv(
        file_name=args.unity_path,
        reward_fn=reward_fn,
        time_scale=args.time_scale,
        worker_id=args.base_worker_id,
        no_graphics=args.no_graphics,
        decision_drain=not args.no_decision_drain,
        log_file=out / "Player.log",
    )
    rows = []
    try:
        # Flush each progress line so it still shows up when stdout is piped or redirected.
        rows = run_evaluation(env, policies, schedule, log=lambda line: print(line, flush=True))
    finally:
        env.close()
        write_csv(out / "episodes.csv", rows)

    summary = summarize(rows)
    write_csv(out / "summary.csv", summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
