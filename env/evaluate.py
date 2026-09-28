"""
@file evaluate.py
@brief Compare trained checkpoints and fixed PDR rules on the same reproducible instances.

@details
Every (policy, seed) pair is one episode. All seeds are queued in Unity up front, and
Unity reports which queue position each episode consumed (episode_seed_index), so each
episode is attributed to exactly one policy even though episodes roll over inside Unity.
The episode already running when evaluation starts (index -1) is discarded.

PDR baselines run as constant-action policies through the same wrapper and decision path
as the learned policy, so the comparison differs only in the actions chosen. Only the
(job head, machine head) pairs are reachable this way (config.PDR_ACTIONS); other catalog
rules run through the batch runner (linux_server/run_experiment_queue.py). Evaluation
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

from config import (ActorCriticConfig, EncoderConfig, FusionConfig, PDR_ACTIONS, pdr_action,
                    ACTION_BRANCHES, JOB_HEAD_RULES, MACHINE_HEAD_RULES)
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


## @brief Per-head probability columns of decisions.csv.
HEAD_PROB_FIELDS = [f"p_job_{r}" for r in JOB_HEAD_RULES] + [f"p_machine_{r}" for r in MACHINE_HEAD_RULES]

DECISION_FIELDS = (
    ["policy", "kind", "seed", "seed_index", "step", "sim_time", "decision_count", "wip",
     "jobs_exited", "job_head", "machine_head", "job_head_used", "machine_head_used", "rule",
     "entropy", "chosen_prob"]
    + HEAD_PROB_FIELDS
)


def heads_used(action_mask) -> list:
    """@brief Per branch, whether Unity left more than action 0 enabled (the head could change the outcome)."""
    if action_mask is None:
        return [True] * len(ACTION_BRANCHES)
    used, offset = [], 0
    for size in ACTION_BRANCHES:
        used.append(bool(np.asarray(action_mask)[offset + 1:offset + size].sum() > 0))
        offset += size
    return used


def decision_row(policy, seed: int, seed_index: int, step: int, metrics, action, action_mask=None) -> dict:
    """@brief One decisions.csv row: the (job head, machine head) chosen at a decision, which heads could
    change it, and, for checkpoints, the policy's per-head distributions (summed entropy, probability of
    the chosen action over the heads used, p_job_<rule> / p_machine_<rule>)."""
    job, machine = int(action[0]), int(action[1])
    used = heads_used(action_mask)
    row = {
        "policy": policy.name, "kind": policy.kind, "seed": seed, "seed_index": seed_index,
        "step": step, "sim_time": round(metrics.sim_time, 3),
        "decision_count": int(metrics.decision_count), "wip": int(metrics.wip),
        "jobs_exited": int(metrics.jobs_exited), "job_head": job, "machine_head": machine,
        "job_head_used": int(used[0]), "machine_head_used": int(used[1]),
        "rule": f"{JOB_HEAD_RULES[job]}-{MACHINE_HEAD_RULES[machine]}",
    }
    probs = getattr(policy, "last_probs", None)
    if probs is not None:
        entropy, chosen = 0.0, 1.0
        for head, (p_head, a, is_used) in enumerate(zip(probs, (job, machine), used)):
            p = np.clip(p_head, 1e-12, 1.0)
            entropy += float(-(p_head * np.log(p)).sum())
            if is_used:
                chosen *= float(p_head[a])
        row["entropy"] = round(entropy, 5)
        row["chosen_prob"] = round(chosen, 5)
        row.update({name: round(float(v), 5) for name, v in zip(HEAD_PROB_FIELDS, np.concatenate(probs))})
    return row


class ConstantPolicy:
    """@brief A fixed dispatching rule: always the same (job head, machine head)."""

    kind = "pdr"
    last_probs = None

    def __init__(self, action, name: str):
        self.action = tuple(action)
        self.name = name

    def __call__(self, obs):
        return self.action


class CheckpointPolicy:
    """@brief A trained SchedulingNetwork loaded from a train.py checkpoint."""

    kind = "checkpoint"

    def __init__(self, path, device: str = "cpu", deterministic: bool = True):
        from models.network import SchedulingNetwork

        path = Path(path)
        checkpoint = torch.load(path, map_location=device)
        from train import check_action_layout, check_obs_schema
        check_obs_schema(checkpoint, path)
        check_action_layout(checkpoint, path)
        self.net = SchedulingNetwork(EncoderConfig(), FusionConfig(), ActorCriticConfig()).to(device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval()
        self.device = device
        self.deterministic = deterministic
        self.name = f"ckpt:{path.parent.name}/{path.stem}"
        ## @brief Per-head action probabilities from the latest call (for decision logging).
        self.last_probs = None

    def __call__(self, obs):
        obs_t = {k: torch.tensor(v[None], dtype=torch.float32, device=self.device) for k, v in obs.items()}
        with torch.no_grad():
            # Same distributions as SchedulingNetwork.act (masked per head), kept for logging.
            probs = [d.probs[0] for d in self.net.distributions(obs_t)]
            action = tuple(int(p.argmax()) if self.deterministic else int(torch.multinomial(p, 1))
                           for p in probs)
        self.last_probs = [p.cpu().numpy() for p in probs]
        return action


def build_policies(pdr_spec: str, checkpoints, device: str, deterministic: bool) -> list:
    policies = [ConstantPolicy(pdr_action(name), name) for name in resolve_pdr_names(pdr_spec)]
    policies += [CheckpointPolicy(path, device, deterministic) for path in checkpoints or []]

    seen = {}
    for policy in policies:
        seen[policy.name] = seen.get(policy.name, 0) + 1
        if seen[policy.name] > 1:
            policy.name = f"{policy.name}#{seen[policy.name]}"
    return policies


def run_evaluation(env, policies: list, schedule: list, log=print, decision_writer=None,
                   scenario_generator=None) -> list:
    """@brief Play one episode per (policy index, seed) in @p schedule, in order.

    @param env              A @ref UnitySchedulingEnv (or anything with queue_seeds /
                            queue_scenarios / reset / step / current_metrics).
    @param schedule         List of (policy index, seed); position i is queue index i in Unity.
    @param decision_writer  Optional csv.DictWriter (fields @ref DECISION_FIELDS) receiving one
                            row per decision of every scheduled episode.
    @param scenario_generator  Optional seed -> scenario dict (see scenarios/); when set, each
                               seed in @p schedule also queues that seed's scripted-scenario
                               variant, in lockstep with the seed queue.
    @return One row dict per completed episode, in schedule order.
    """
    total = len(schedule)
    seeds = [seed for _, seed in schedule]
    env.queue_seeds(seeds, clear=True)
    if scenario_generator is not None:
        env.queue_scenarios([scenario_generator(seed) for seed in seeds], clear=True)
    obs = env.reset()
    if env.current_metrics is None:
        raise RuntimeError("This Unity build has no reward-metrics sensor; rebuild the player.")

    rows, discarded, start, episode_step = [], 0, time.time(), 0
    while len(rows) < total:
        metrics = env.current_metrics
        index = int(metrics.episode_seed_index)
        # The episode that was already running before our seeds were queued (index -1) is
        # played out with action (0, 0) and discarded.
        policy = policies[schedule[index][0]] if 0 <= index < total else None
        action = policy(obs) if policy is not None else (0,) * len(ACTION_BRANCHES)
        if decision_writer is not None and policy is not None:
            decision_writer.writerow(decision_row(policy, schedule[index][1], index, episode_step, metrics,
                                                  action, obs.get("action_mask")))
        episode_step += 1
        obs, _, done, info = env.step(action)
        if not done:
            continue
        episode_step = 0

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
            "truncated": episode["truncated"],
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
                        help="PDR baselines: 'all', 'none', or a comma list of JOB-MACHINE names from the "
                             "RL heads (e.g. SPT-ECT)")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="train.py checkpoint to evaluate (repeatable)")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Scripted scenario JSON (ScenarioLoader schema) to evaluate on instead of "
                             "the generated default config; seeds then only label repeats "
                             "(mutually exclusive with --scenario-generator)")
    from scenarios import REGISTRY as _SCENARIO_REGISTRY
    parser.add_argument("--scenario-generator", type=str, default=None,
                        choices=sorted(_SCENARIO_REGISTRY),
                        help="Evaluate each seed on its own generated scripted-scenario variant "
                             "instead of a fixed instance (see env/scenarios)")
    parser.add_argument("--episode-duration-seconds", type=float, default=0.0,
                        help="With --scenario-generator, cap each episode at this many sim-seconds "
                             "(steady-state mode); 0 runs the scenario to its natural length")
    parser.add_argument("--machine-flexibility", type=float, default=0.0,
                        help="With --scenario-generator: probability that a machine can also run each other "
                             "operation type (0 = fully typed; see FJSSPConfig.MachineFlexibilityProbability)")
    parser.add_argument("--secondary-time-multiplier", type=float, default=1.0,
                        help="With --machine-flexibility: processing-time factor on a machine's secondary types")
    parser.add_argument("--decision-log", action="store_true",
                        help="Also write decisions.csv: every decision's chosen rule, plus the "
                             "action probabilities for checkpoints")
    parser.add_argument("--unity-decision-log", action="store_true",
                        help="Have Unity also write decision_log.csv (candidate counts, degenerate "
                             "flags) into --out; needs a player built with -decisionlogdir support")
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
    parser.add_argument("--obs-max-machines", type=int, default=0,
                        help="Observation machine rows (0 = fit the largest evaluated floor)")
    parser.add_argument("--obs-max-jobs", type=int, default=0,
                        help="Observation job rows (0 = 256 per 15 machines of the largest floor)")
    args = parser.parse_args(argv)

    if args.scenario and args.scenario_generator:
        parser.error("--scenario and --scenario-generator are mutually exclusive")

    scenario_generator = None
    if args.scenario_generator:
        from scenarios import REGISTRY
        duration = args.episode_duration_seconds if args.episode_duration_seconds > 0 else None
        scenario_generator = REGISTRY[args.scenario_generator](
            duration, machine_flexibility=args.machine_flexibility,
            secondary_time_multiplier=args.secondary_time_multiplier)

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

    # Row caps fit the evaluated floors (checkpoints don't depend on them; see config.obs_row_caps).
    from scenarios import row_caps_for
    if args.scenario:
        planned = [args.scenario]
    elif scenario_generator is not None:
        planned = [scenario_generator(seed) for seed in seeds]
    else:
        planned = []
    obs_caps = row_caps_for(planned, args.obs_max_machines, args.obs_max_jobs)
    print(f"Observation row caps: {obs_caps[0]} machines, {obs_caps[1]} jobs")

    env = UnitySchedulingEnv(
        file_name=args.unity_path,
        obs_caps=obs_caps,
        reward_fn=reward_fn,
        time_scale=args.time_scale,
        worker_id=args.base_worker_id,
        no_graphics=args.no_graphics,
        decision_drain=not args.no_decision_drain,
        log_file=out / "Player.log",
        extra_args=["-decisionlogdir", str(out.resolve())] if args.unity_decision_log else None,
    )
    if args.scenario:
        env.load_scenario(args.scenario)

    rows = []
    decision_file = open(out / "decisions.csv", "w", newline="") if args.decision_log else None
    decision_writer = None
    if decision_file is not None:
        decision_writer = csv.DictWriter(decision_file, fieldnames=DECISION_FIELDS)
        decision_writer.writeheader()
    try:
        # Flush each progress line so it still shows up when stdout is piped or redirected.
        rows = run_evaluation(env, policies, schedule, log=lambda line: print(line, flush=True),
                              decision_writer=decision_writer, scenario_generator=scenario_generator)
    finally:
        env.close()
        write_csv(out / "episodes.csv", rows)
        if decision_file is not None:
            decision_file.close()

    summary = summarize(rows)
    write_csv(out / "summary.csv", summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
