"""
@file evaluate.py
@brief Compare trained checkpoints and fixed PDR rules on the same reproducible instances.

@details
Every (policy, seed) pair is one episode. Seeds (and scenario variants) are queued in Unity
in schedule order, a few episodes ahead of the running one (inline scenarios are large, and
queueing a whole schedule at once can exceed gRPC's 4 MB message cap), and Unity reports
which queue position each episode consumed (episode_seed_index), so each episode is
attributed to exactly one policy even though episodes roll over inside Unity.
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
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (ActorCriticConfig, FusionConfig, PDR_ACTIONS, pdr_action,
                    ACTION_BRANCHES, JOB_HEAD_RULES, MACHINE_HEAD_RULES)
from env_wrappers.unity_env import SEED_BUFFER, TRAIN_SEED_LOW, UnitySchedulingEnv

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
        from models.network import SchedulingNetwork, ac_config_for, encoder_config_for

        path = Path(path)
        from train import check_action_layout, check_obs_schema, load_checkpoint
        checkpoint = load_checkpoint(path, device)
        check_obs_schema(checkpoint, path)
        check_action_layout(checkpoint, path)
        # The policy uses the actor only; a look-ahead critic (ac_config_for) is loaded but never called here.
        self.net = SchedulingNetwork(encoder_config_for(checkpoint["model_state_dict"]), FusionConfig(),
                                     ac_config_for(checkpoint["model_state_dict"])).to(device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval()
        self.device = device
        self.deterministic = deterministic
        self.name = f"ckpt:{path.parent.name}/{path.stem}" + ("" if deterministic else "~sampled")
        ## @brief What the policy was trained on (train.run_settings; {} for checkpoints before 2026-10-07) and its
        ##        PPO config, so main() can take slot_seconds and the discount from them and warn on mismatches.
        self.settings = dict(checkpoint.get("run_settings") or {})
        self.ppo = dict((checkpoint.get("config") or {}).get("ppo") or {})
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
                   scenario_generator=None, scenario_dir=None, gamma_per_second=None) -> list:
    """@brief Play one episode per (policy index, seed) in @p schedule, in order.

    @param env              A @ref UnitySchedulingEnv (or anything with queue_seeds /
                            queue_scenarios / reset / step / current_metrics).
    @param schedule         List of (policy index, seed); position i is queue index i in Unity.
                            Items are queued incrementally, SEED_BUFFER ahead of the running
                            episode, never all at once.
    @param decision_writer  Optional csv.DictWriter (fields @ref DECISION_FIELDS) receiving one
                            row per decision of every scheduled episode.
    @param scenario_generator  Optional seed -> scenario dict (see scenarios/); when set, each
                               seed in @p schedule also queues that seed's scripted-scenario
                               variant, in lockstep with the seed queue.
    @param scenario_dir     If set, each seed's scenario is written once to scenario_dir/s<seed>.json
                            and queued as that path, not inline: Unity's gRPC receive limit is
                            4 MiB per message, and a few linked-floor scenarios exceed it.
    @param gamma_per_second  If set, each row also gets discounted_return: sum gamma_s^(tau - tau_0) r over the
                            window, the quantity training maximizes (without its bootstrap past the window end), next
                            to the undiscounted return that is scored (audit M3, 2026-10-07). A slot env must discount
                            within its slots with the same gamma_s (SlotActionEnv gamma_per_second).
    @return One row dict per completed episode, in schedule order.
    """
    total = len(schedule)
    queued = 0   # schedule items sent to Unity so far; item i is Unity queue index i

    def top_up(consumed: int):
        """Keep SEED_BUFFER items queued beyond the @p consumed ones Unity has started. Queues are
        cleared only on the first call, so Unity's index keeps counting from schedule position 0."""
        nonlocal queued
        end = min(total, consumed + SEED_BUFFER)
        if end <= queued and queued > 0:
            return
        seeds = [seed for _, seed in schedule[queued:end]]
        env.queue_seeds(seeds, clear=queued == 0)
        if scenario_generator is not None:
            env.queue_scenarios([scenario_item(seed) for seed in seeds], clear=queued == 0)
        queued = end

    def scenario_item(seed: int):
        if scenario_dir is None:
            return scenario_generator(seed)
        path = Path(scenario_dir) / f"s{seed}.json"
        if path not in written:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(scenario_generator(seed)))
            written.add(path)
        return str(path)

    written = set()

    top_up(0)
    obs = env.reset()
    if env.current_metrics is None:
        raise RuntimeError("This Unity build has no reward-metrics sensor; rebuild the player.")

    rows, discarded, start, episode_step, consumed = [], 0, time.time(), 0, 0
    disc_return, discount = 0.0, 1.0
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
        obs, reward, done, info = env.step(action)
        if gamma_per_second is not None:
            disc_return += discount * float(reward)
            discount *= gamma_per_second ** float(info.get("dt") or 0.0)
        if not done:
            continue
        episode_step = 0
        episode_disc_return = disc_return if gamma_per_second is not None else None
        disc_return, discount = 0.0, 1.0
        # Unity has already started the next episode, consuming queue index started_index; refill
        # now so the following items arrive (with the next step) before this new episode ends.
        started_index = int(env.current_metrics.episode_seed_index)
        consumed = max(consumed, started_index + 1)
        top_up(consumed)

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
            "discounted_return": episode_disc_return,
            "decisions": episode["length"],
            "jobs_exited": episode["jobs_exited"],
            "deadlock": episode["deadlock"],
            "timed_out": episode["timed_out"],
            "truncated": episode["truncated"],
            "tick_error": episode.get("tick_error", False),
            "machine_failures": episode.get("machine_failures"),
            "agv_failures": episode.get("agv_failures"),
            "agv_repair_time": episode.get("agv_repair_time"),
            "agv_blocked_by_failure_time": episode.get("agv_blocked_by_failure_time"),
            "routed_moves": episode.get("routed_moves"),
            "cross_tile_moves": episode.get("cross_tile_moves"),
            "tiles_crossed": episode.get("tiles_crossed"),
            "agv_idle_fraction": episode.get("agv_idle_fraction"),
            "release_counts": episode.get("release_counts"),
            "travel_price": episode.get("travel_price"),
            "input_buffer_capacity": episode.get("input_buffer_capacity"),
            "output_buffer_capacity": episode.get("output_buffer_capacity"),
            "output_blocked_machine_seconds": episode.get("output_blocked_machine_seconds"),
            "buffer_wait_job_seconds": episode.get("buffer_wait_job_seconds"),
            # Both objectives over the agent window, whatever the reward (rewards.metrics.window_outcomes); the due-date
            # fields are None from a player older than reward metrics v5.
            "window_time_in_system": episode.get("window_time_in_system"),
            "window_tardiness": episode.get("window_tardiness"),
            "tardiness_exited_sum": episode.get("tardiness_exited_sum"),
            "jobs_exited_late": episode.get("jobs_exited_late"),
            "jobs_with_due_date": episode.get("jobs_with_due_date"),
            "config_hash": episode.get("config_hash"),
            "instance_hash": episode.get("instance_hash"),
        })
        log(f"[{len(rows):4d}/{total}] seed {seed:5d}  {policy.name:28s} "
            f"makespan {episode['makespan']:7.1f}  total flow {episode['total_flow_time']:8.1f}  "
            f"({time.time() - start:5.0f}s)")
    return rows


def _gap_pct(values, best):
    """@brief Mean relative gap in percent; seeds whose best is 0 are skipped (None if none are left)."""
    gaps = [(v - b) / abs(b) for v, b in zip(values, best) if b != 0]
    return float(100 * np.mean(gaps)) if gaps else None


def _cost_change_pct(policy_costs, ref_costs):
    """@brief 100 * (mean policy cost / mean reference cost - 1) over paired seeds: < 0 beats the reference."""
    ref = float(np.mean(ref_costs))
    return float(100 * (np.mean(policy_costs) / ref - 1)) if ref != 0 else None


def summarize(rows: list, scored: bool = True, oracle: dict = None, best_fixed: str = None) -> list:
    """@brief Per-policy means, plus paired gaps against the best PDR rule on each seed.

    @details The gap for a policy is the mean over seeds of (policy − best PDR on that seed) /
    |best PDR on that seed|, in percent: negative beats every rule on that instance, 0 matches
    the per-seed oracle choice among the rules. The best PDR comes from episodes that neither
    deadlocked nor timed out (both stop the clock early).

    The score is `return` (with a reward spec: time in system of every job, finished or not),
    so its gap is on -return: > 0 is worse. Total flow and makespan are censored by jobs still
    in the system (and, with random warm-up, count jobs that finished during the warm-up); their
    gaps are kept for reference only. Rows sort by the return gap when @p scored, else by name.

    The three comparisons every policy result reports (user, 2026-10-07; as results/rq2-realizable/analyze.py), each
    the change in mean cost (-return) over the seeds both have, < 0 better:
      - vs_best_fixed_pct: the best fixed pair overall (@p best_fixed, e.g. chosen on training seeds; else the PDR
        with the best mean return on these seeds, which favours the baseline);
      - vs_hindsight_fixed_pct: each instance's own best fixed pair (the per-seed best PDR above);
      - vs_oracle_pct: the switching oracle (@p oracle, seed -> oracle return), and oracle_gain_captured_pct, the
        share of the oracle's gain over the best fixed pair the policy realizes.
    """
    def clean(r):
        return not (r["deadlock"] or r["timed_out"] or r.get("tick_error"))

    best = {}
    for row in rows:
        if row["kind"] == "pdr" and clean(row):
            ret, makespan, flow = best.get(row["seed"], (-math.inf, math.inf, math.inf))
            best[row["seed"]] = (max(ret, row["return"]), min(makespan, row["makespan"]),
                                 min(flow, row["total_flow_time"]))

    by_policy = {}
    for row in rows:
        by_policy.setdefault(row["policy"], []).append(row)

    # The best fixed pair overall: named, or the PDR with the best mean return on the seeds every PDR finished cleanly.
    fixed_returns = None
    if scored:
        pdr = {name: {r["seed"]: r["return"] for r in rs if clean(r)}
               for name, rs in by_policy.items() if rs[0]["kind"] == "pdr"}
        if best_fixed is not None and best_fixed not in pdr:
            raise ValueError(f"best fixed pair {best_fixed!r} was not evaluated (pass it in --pdr)")
        if best_fixed is None and pdr:
            common = set.intersection(*(set(v) for v in pdr.values()))
            if common:
                best_fixed = max(pdr, key=lambda n: np.mean([pdr[n][s] for s in common]))
        fixed_returns = pdr.get(best_fixed)

    summary = []
    for name, policy_rows in by_policy.items():
        makespans = np.array([r["makespan"] for r in policy_rows])
        entry = {
            "policy": name,
            "kind": policy_rows[0]["kind"],
            "episodes": len(policy_rows),
            "return_mean": float(np.mean([r["return"] for r in policy_rows])) if scored else None,
            "jobs_exited_mean": float(np.mean([r["jobs_exited"] for r in policy_rows])),
            "makespan_mean": float(makespans.mean()),
            "makespan_std": float(makespans.std()),
            "total_flow_censored_mean": float(np.mean([r["total_flow_time"] for r in policy_rows])),
            "mean_flow_time_censored_mean": float(np.mean([r["mean_flow_time"] for r in policy_rows])),
            "deadlocks": int(sum(bool(r["deadlock"]) for r in policy_rows)),
            "timeouts": int(sum(bool(r["timed_out"]) for r in policy_rows)),
        }
        discounted = [r.get("discounted_return") for r in policy_rows]
        if scored and discounted and all(d is not None for d in discounted):
            entry["discounted_return_mean"] = float(np.mean(discounted))
        # Deadlocked / timed-out episodes stop the clock early, so they are counted above, not gapped.
        paired = [r for r in policy_rows if r["seed"] in best and clean(r)]
        if paired:
            ref = [best[r["seed"]] for r in paired]
            entry["return_gap_pct"] = _gap_pct([-r["return"] for r in paired], [-b[0] for b in ref]) \
                if scored else None
            entry["makespan_gap_pct"] = _gap_pct([r["makespan"] for r in paired], [b[1] for b in ref])
            entry["flow_censored_gap_pct"] = _gap_pct([r["total_flow_time"] for r in paired], [b[2] for b in ref])
        if scored:
            mine = {r["seed"]: r["return"] for r in policy_rows if clean(r)}
            if fixed_returns:
                seeds = sorted(set(mine) & set(fixed_returns))
                if seeds:
                    entry["best_fixed"] = best_fixed
                    entry["vs_best_fixed_pct"] = _cost_change_pct([-mine[s] for s in seeds],
                                                                  [-fixed_returns[s] for s in seeds])
                    entry["wins_vs_best_fixed"] = int(sum(mine[s] > fixed_returns[s] + 1e-9 for s in seeds))
                    entry["losses_vs_best_fixed"] = int(sum(mine[s] < fixed_returns[s] - 1e-9 for s in seeds))
            seeds = sorted(set(mine) & set(best))
            if seeds:
                entry["vs_hindsight_fixed_pct"] = _cost_change_pct([-mine[s] for s in seeds],
                                                                   [-best[s][0] for s in seeds])
            if oracle:
                seeds = sorted(set(mine) & set(oracle))
                if seeds:
                    entry["oracle_seeds"] = len(seeds)
                    entry["vs_oracle_pct"] = _cost_change_pct([-mine[s] for s in seeds], [-oracle[s] for s in seeds])
                    if fixed_returns and set(seeds) <= set(fixed_returns):
                        gain = np.mean([fixed_returns[s] - oracle[s] for s in seeds])   # <= 0 in return units
                        mine_gain = np.mean([fixed_returns[s] - mine[s] for s in seeds])
                        entry["oracle_gain_captured_pct"] = float(100 * mine_gain / gain) if gain != 0 else None
        summary.append(entry)
    if scored:
        return sorted(summary, key=lambda e: (e.get("return_gap_pct") is None, e.get("return_gap_pct") or 0.0))
    return sorted(summary, key=lambda e: e["policy"])


def load_oracle(paths) -> tuple:
    """@brief (seed -> switching-oracle return, seed -> {pair: fixed-pair return}) from result JSON files (globs
    allowed): env/switch_oracle.py's result.json ("seed", "oracle_return", "fixed_returns") or the twin oracle tasks
    ("seed", "oracle" and "fixed" {pair: {"tard"}}, costs = -return). The fixed-pair returns let main() check that the
    oracle was run on the same instances and window as this evaluation."""
    import glob
    oracle, fixed = {}, {}
    for pattern in paths:
        files = sorted(glob.glob(pattern)) or [pattern]
        for f in files:
            data = json.loads(Path(f).read_text())
            seed = int(data["seed"])
            if "oracle_return" in data:
                oracle[seed] = float(data["oracle_return"])
                fixed[seed] = {k: float(v) for k, v in (data.get("fixed_returns") or {}).items()}
            elif "oracle" in data:
                oracle[seed] = -float(data["oracle"])
                fixed[seed] = {k: -float(v["tard"]) for k, v in (data.get("fixed") or {}).items()}
            else:
                raise ValueError(f"{f}: no 'oracle_return' or 'oracle' field")
    return oracle, fixed


def check_oracle(rows: list, oracle_fixed: dict, tol: float = 1e-3) -> None:
    """@brief Warn when this evaluation's fixed-pair returns differ from the oracle run's own fixed-pair returns on
    the same seeds: then the oracle saw other instances, another window or another model, and vs_oracle is invalid."""
    diffs = [abs(r["return"] - oracle_fixed[r["seed"]][r["policy"]]) / max(abs(r["return"]), 1e-9)
             for r in rows if r["kind"] == "pdr" and r["policy"] in oracle_fixed.get(r["seed"], {})]
    if not diffs:
        print("[NOTE] the oracle results carry no fixed pairs that were evaluated here: vs_oracle is unchecked.")
    elif max(diffs) > tol:
        print(f"[WARNING] fixed-pair returns differ from the oracle run's own by up to {100 * max(diffs):.2f}% "
              f"({sum(d > tol for d in diffs)}/{len(diffs)} episodes): the oracle is not on these instances or this "
              "window, so vs_oracle is not comparable.")
    else:
        print(f"Oracle check: {len(diffs)} fixed-pair episodes match the oracle run's (max rel. diff {max(diffs):.1e}).")


def print_summary(summary: list):
    def gap(value):
        return f"{value:+7.2f}" if value is not None else " " * 7

    scored = any(e.get("return_mean") is not None for e in summary)
    header = (f"{'policy':30s} {'n':>4s} {'return':>10s} {'gap%':>7s} {'jobs out':>8s} "
              f"{'total flow*':>11s} {'deadlk':>6s} {'timeout':>7s}")
    print()
    print(header)
    print("-" * len(header))
    for e in summary:
        ret = f"{e['return_mean']:10.2f}" if e.get("return_mean") is not None else " " * 10
        print(f"{e['policy']:30s} {e['episodes']:4d} {ret} {gap(e.get('return_gap_pct'))} "
              f"{e['jobs_exited_mean']:8.1f} {e['total_flow_censored_mean']:11.1f} "
              f"{e['deadlocks']:6d} {e['timeouts']:7d}")
    if scored:
        print("gap% = mean per-seed gap in return to the best PDR rule on that seed (> 0 worse); "
              "deadlocked / timed-out episodes are left out of it.")
        print_comparisons(summary)
    else:
        print("No --reward-spec: return is Unity's pass-through (0), so nothing is ranked.")
    print("* total flow of exited jobs only (censored, read it with jobs out); with --random-warmup both also "
          "count jobs that finished during the warm-up.")


def print_comparisons(summary: list):
    """@brief The three references per policy (summarize): change in mean cost, < 0 better."""
    def pct(value):
        return f"{value:+8.2f}" if value is not None else " " * 8

    ref = next((e.get("best_fixed") for e in summary if e.get("best_fixed")), None)
    has_oracle = any("vs_oracle_pct" in e for e in summary)
    has_disc = any("discounted_return_mean" in e for e in summary)
    header = (f"{'policy':30s} {'vs best':>8s} {'vs hind':>8s}" + (f" {'vs orcl':>8s} {'captured':>8s}" if has_oracle else "")
              + f" {'W/L vs best':>11s}" + (f" {'disc ret':>10s}" if has_disc else ""))
    print()
    print(header)
    print("-" * len(header))
    for e in summary:
        wl = (f"{e['wins_vs_best_fixed']}/{e['losses_vs_best_fixed']}" if "wins_vs_best_fixed" in e else "")
        line = f"{e['policy']:30s} {pct(e.get('vs_best_fixed_pct'))} {pct(e.get('vs_hindsight_fixed_pct'))}"
        if has_oracle:
            line += f" {pct(e.get('vs_oracle_pct'))} {pct(e.get('oracle_gain_captured_pct'))}"
        line += f" {wl:>11s}"
        if has_disc:
            d = e.get("discounted_return_mean")
            line += f" {d:10.3f}" if d is not None else " " * 11
        print(line)
    print(f"Change in mean cost (-return) in %, < 0 better: vs the best fixed pair overall ({ref}), vs each instance's "
          "own best fixed pair (hindsight)" + (", vs the switching oracle; captured = share of the oracle's gain over "
                                              "the best fixed pair" if has_oracle else "")
          + (". disc ret = discounted window return (training's gamma, no bootstrap)" if has_disc else "") + ".")


def floor_fields(scenario: dict) -> dict:
    """@brief The floor a generated scenario runs on, as episodes.csv columns (fleet, layout, tiling, TECT price)."""
    tiling = scenario.get("tiling") or {}
    return {
        "agv_count": scenario.get("agvCount"),
        "layout": scenario.get("layout"),
        "tiles": tiling.get("tiles", 1),
        "job_scope": tiling.get("jobScope", "tile"),
        "agv_assignment": tiling.get("agvAssignment", "tile"),
        "release_rule": tiling.get("releaseRule", "roundRobin"),
        "release_weights": ";".join(f"{w:g}" for w in tiling.get("releaseWeights", [])),
        "scenario_travel_price": scenario.get("travelPrice", 0.0),
        "scenario_input_buffer_capacity": scenario.get("inputBufferCapacity", 0),
        "scenario_output_buffer_capacity": scenario.get("outputBufferCapacity", 0),
        # Due dates: the generator's allowance c (RandomizedParams.due_date_allowance; 0 = none) and how many jobs carry one.
        "scenario_due_date_allowance": ((scenario.get("_meta") or {}).get("params") or {}).get("due_date_allowance", 0.0),
        "scenario_jobs_with_due_date": sum(1 for j in scenario.get("jobs", []) if j.get("dueDate") is not None),
    }


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


## @brief run_settings keys evaluate.py compares with its own flags (train.RUN_SETTING_KEYS), with the flag's default.
CHECKED_SETTINGS = {"twin_transport": "kinematic", "scenario_generator": None, "params": None,
                    "episode_duration_seconds": 0.0, "random_warmup": False, "machine_flexibility": 0.0}


def resolve_checkpoint_settings(args, policies) -> tuple:
    """@brief (slot_seconds, gamma_per_second) for this evaluation, from the flags or the checkpoints (audit m2, M3).

    @details A slot-trained checkpoint run per decision is out of distribution (per-decision masks, a different action
    cadence) and nothing used to say so. Unset --slot-seconds / --discount-horizon-s take the checkpoints' own values
    (they must agree, since one wrapper serves every policy); a flag that disagrees with a checkpoint is kept, with a
    warning, as is any difference in the scenario settings. Checkpoints before 2026-10-07 record no run settings.
    """
    ckpts = [p for p in policies if p.kind == "checkpoint"]

    def from_ckpts(get, what):
        values = {p.name: get(p) for p in ckpts}
        known = {v for v in values.values() if v is not None}
        if len(known) > 1:
            raise SystemExit(f"error: the checkpoints were trained with different {what} ({values}); pass it explicitly "
                             "or evaluate them separately")
        return known.pop() if known else None

    slot = from_ckpts(lambda p: p.settings.get("slot_seconds"), "slot_seconds")
    if args.slot_seconds is None:
        if slot is None and any(not p.settings for p in ckpts):
            print("[NOTE] checkpoint(s) from before 2026-10-07 record no slot setting: evaluating per decision. Pass "
                  "--slot-seconds if they were trained with slot actions.")
        slot_seconds = float(slot or 0.0)
        if slot:
            print(f"Slot actions: {slot_seconds:g} s, as the checkpoints were trained")
    else:
        slot_seconds = float(args.slot_seconds)
        for p in ckpts:
            trained = p.settings.get("slot_seconds")
            if trained is not None and float(trained) != slot_seconds:
                print(f"[WARNING] {p.name} was trained with slot_seconds={trained}, evaluating with {slot_seconds:g}: "
                      "its actions are out of distribution")

    horizon = args.discount_horizon_s
    if horizon is None:
        horizon = from_ckpts(lambda p: p.ppo.get("discount_horizon_s"), "discount_horizon_s")
    gamma_per_second = 1.0 - 1.0 / horizon if horizon and horizon > 1.0 else None
    if gamma_per_second is not None:
        print(f"Discounted return: horizon {horizon:,.0f} s (gamma_s = {gamma_per_second:.6f})")

    for p in ckpts:
        for key, default in CHECKED_SETTINGS.items():
            if key not in p.settings:
                continue
            trained, used = p.settings[key], getattr(args, key, default)
            if key == "params":
                trained, used = (json.loads(x) if x else None for x in (trained, used))
            if (trained if trained is not None else default) != (used if used is not None else default):
                print(f"[WARNING] {p.name} was trained with {key}={trained!r}, evaluating with {used!r}")
    return slot_seconds, gamma_per_second


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate policies on reproducible instances")
    parser.add_argument("--unity-path", type=str, default=None, help="player (required unless --twin)")
    parser.add_argument("--twin", type=str, default=None, metavar="DES_FLOOR_JSON",
                        help="evaluate on the event-based twin instead of a player (2026-10-05); the floor export must "
                             "carry the observation frame and match the scenarios' layout and fleet")
    parser.add_argument("--twin-transport", default="kinematic", choices=("instant", "geometric", "kinematic"))
    parser.add_argument("--slot-seconds", type=float, default=None,
                        help="hold each action for this many sim-seconds (slot actions); default: the checkpoints' own "
                             "training setting (recorded since 2026-10-07), else 0 = per decision")
    parser.add_argument("--discount-horizon-s", type=float, default=None,
                        help="also report each episode's discounted return with gamma_s = 1 - 1/H (training's "
                             "objective); default: the checkpoints' own horizon; 0 = off")
    parser.add_argument("--oracle", action="append", default=[], metavar="JSON_GLOB",
                        help="switching-oracle results for the evaluated seeds (switch_oracle.py result.json or twin "
                             "oracle task JSONs; repeatable, globs allowed): adds the vs-oracle comparison")
    parser.add_argument("--best-fixed", type=str, default=None, metavar="JOB-MACHINE",
                        help="the best fixed pair overall, chosen elsewhere (e.g. on training seeds); default: the PDR "
                             "with the best mean return on the evaluated seeds")
    # No effect since 2026-10-02, when the launch-time check against BUILD_MANIFEST.json was removed
    # (see env/player_manifest.py); still accepted so older commands run.
    parser.add_argument("--allow-unverified-player", action="store_true", help=argparse.SUPPRESS)
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
    parser.add_argument("--params", type=str, default=None, metavar="JSON",
                        help="With --scenario-generator randomized: RandomizedParams overrides (env/scenarios/"
                             "randomized.py), e.g. a linked 7-tile floor '{\"tiles\": 7, \"job_scope\": \"open\", "
                             "\"agv_assignment\": \"pooled\", \"agv_count\": 49}' or TECT's travel price "
                             "'{\"travel_price\": 4}'. Applied after --machine-flexibility")
    parser.add_argument("--random-warmup", action="store_true",
                        help="With --scenario-generator: start each episode mid-stream after a heuristic warm-up "
                             "(the training distribution; the warm-up cutoff is drawn from the seed)")
    parser.add_argument("--agv-failures", nargs="?", const="{}", default=None, metavar="JSON",
                        help="With --scenario-generator: add AGV breakdowns to every instance (same instance "
                             "otherwise). Bare flag = scenarios.AGV_FAILURE_DEFAULTS; a JSON object overrides "
                             "fields, e.g. '{\"agvWeibullLambda\": 6000, \"agvRepairLogMu\": 4.2}'")
    parser.add_argument("--agvs", type=int, default=None, metavar="N",
                        help="With --scenario-generator: run every instance with N AGVs. The jobs are the "
                             "generator's own (its arrival cap stays at its default fleet), so seeds stay "
                             "paired across fleet sizes")
    parser.add_argument("--layout", type=str, default=None, metavar="X",
                        help="With --scenario-generator: run every instance on floor layout X (A-O); the "
                             "jobs are unchanged")
    parser.add_argument("--input-buffer", type=int, default=None, metavar="N",
                        help="With --scenario-generator: machine input buffer size (0 = unbounded); same jobs")
    parser.add_argument("--output-buffer", type=int, default=None, metavar="N",
                        help="With --scenario-generator: machine output buffer size (0 = unbounded; a full "
                             "buffer blocks the machine); same jobs")
    parser.add_argument("--decision-log", action="store_true",
                        help="Also write decisions.csv: every decision's chosen rule, plus the "
                             "action probabilities for checkpoints")
    parser.add_argument("--unity-decision-log", action="store_true",
                        help="Have Unity also write decision_log.csv (candidate counts, degenerate "
                             "flags) into --out; needs a player built with -decisionlogdir support")
    parser.add_argument("--stochastic-policy", action="store_true",
                        help="Sample checkpoint actions instead of taking the argmax (policy names get a "
                             "'~sampled' suffix in the CSVs)")
    parser.add_argument("--policy-seed", type=int, default=0,
                        help="torch seed for --stochastic-policy sampling, so sampled runs are reproducible")
    parser.add_argument("--warmup-dispatching-rule", type=str, default=None,
                        help="As train.py: DispatchingRule driving the --random-warmup window")
    parser.add_argument("--agv-move-speed", type=float, default=None,
                        help="As train.py: with --scenario-generator, overrides AGV travel speed")
    parser.add_argument("--agv-handshake-duration", type=float, default=None,
                        help="As train.py: with --scenario-generator, overrides the AGV handshake time")
    parser.add_argument("--reward-spec", type=str, default=str(REPO_ROOT / "env" / "config" / "rewards" / "flow_time.json"),
                        help="Reward spec whose episode return is the score (default flow_time: time in system of "
                             "every job); 'none' passes Unity's reward through (always 0) and ranks nothing")
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
    if not args.unity_path and not args.twin:
        parser.error("pass --unity-path (a player) or --twin (a des_floor.json)")

    if args.scenario and args.scenario_generator:
        parser.error("--scenario and --scenario-generator are mutually exclusive")
    if args.agv_failures is not None and not args.scenario_generator:
        parser.error("--agv-failures needs --scenario-generator")
    if (args.agvs is not None or args.layout is not None) and not args.scenario_generator:
        parser.error("--agvs / --layout need --scenario-generator")
    if (args.input_buffer is not None or args.output_buffer is not None) and not args.scenario_generator:
        parser.error("--input-buffer / --output-buffer need --scenario-generator")
    if (args.params is not None or args.random_warmup) and not args.scenario_generator:
        parser.error("--params / --random-warmup need --scenario-generator")
    if args.params is not None and args.scenario_generator != "randomized":
        parser.error("--params applies to the randomized generator only")

    scenario_generator = None
    if args.scenario_generator:
        from scenarios import REGISTRY
        duration = args.episode_duration_seconds if args.episode_duration_seconds > 0 else None
        extra = {}
        if args.params is not None:
            try:
                extra["params_overrides"] = json.loads(args.params)
                if not isinstance(extra["params_overrides"], dict):
                    raise ValueError("expected a JSON object")
                print(f"Generator overrides: {extra['params_overrides']}")
            except ValueError as exc:
                parser.error(f"--params: {exc}")
        # The same generator arguments as train.py, so a checkpoint is evaluated on its training distribution.
        try:
            scenario_generator = REGISTRY[args.scenario_generator](
                duration, random_warmup=args.random_warmup, warmup_dispatching_rule=args.warmup_dispatching_rule,
                agv_move_speed=args.agv_move_speed, agv_handshake_duration=args.agv_handshake_duration,
                machine_flexibility=args.machine_flexibility,
                secondary_time_multiplier=args.secondary_time_multiplier, **extra)
        except ValueError as exc:
            parser.error(f"--params: {exc}")
        if args.agv_failures is not None:
            from scenarios import agv_failure_block, with_agv_failures
            try:
                overrides = json.loads(args.agv_failures)
                if not isinstance(overrides, dict):
                    raise ValueError("expected a JSON object")
                print(f"AGV breakdowns on: {agv_failure_block(overrides)}")
            except (ValueError, KeyError) as exc:
                parser.error(f"--agv-failures: {exc}")
            scenario_generator = with_agv_failures(scenario_generator, overrides)
        if args.agvs is not None or args.layout is not None:
            from scenarios import floor_override_fields, with_floor
            try:
                fields = floor_override_fields(args.agvs, args.layout)
            except ValueError as exc:
                parser.error(f"--agvs / --layout: {exc}")
            print(f"Floor override: {fields}")
            scenario_generator = with_floor(scenario_generator, args.agvs, args.layout)
        if args.input_buffer is not None or args.output_buffer is not None:
            from scenarios import buffer_override_fields, with_buffers
            try:
                fields = buffer_override_fields(args.input_buffer, args.output_buffer)
            except ValueError as exc:
                parser.error(f"--input-buffer / --output-buffer: {exc}")
            print(f"Machine buffers: {fields}")
            scenario_generator = with_buffers(scenario_generator, args.input_buffer, args.output_buffer)

    seeds = parse_seeds(args.seeds)
    if max(seeds) >= TRAIN_SEED_LOW:
        print(f"Warning: seeds >= {TRAIN_SEED_LOW} can coincide with training instances.")

    if args.stochastic_policy:
        torch.manual_seed(args.policy_seed)
    policies = build_policies(args.pdr, args.checkpoint, args.device, not args.stochastic_policy)
    if not policies:
        parser.error("Nothing to evaluate: pass --checkpoint and/or --pdr")

    slot_seconds, gamma_per_second = resolve_checkpoint_settings(args, policies)
    oracle, oracle_fixed = load_oracle(args.oracle) if args.oracle else (None, None)
    best_fixed = args.best_fixed.strip().upper().replace("_", "-") if args.best_fixed else None

    # Seed-major order, so an interrupted run still has every policy on the seeds it finished.
    schedule = [(p, seed) for seed in seeds for p in range(len(policies))]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {len(policies)} policies × {len(seeds)} seeds = {len(schedule)} episodes -> {out}")

    reward_fn = None
    if args.reward_spec and args.reward_spec.lower() != "none":
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

    if args.twin:
        # Event-based twin (2026-10-05): same queue interface, episodes simulated in this process.
        from env_wrappers.twin_env import TwinSchedulingEnv
        env = TwinSchedulingEnv(args.twin, transport=args.twin_transport, reward_fn=reward_fn, obs_caps=obs_caps,
                                queued=True)
    else:
        env = None
    env = env or UnitySchedulingEnv(
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
    if slot_seconds > 0:
        # Hold each policy's action for a slot (env_wrappers.slot_env); a fixed pair is unchanged by it. Its reward
        # is discounted within the slot with training's gamma, for the discounted_return column.
        from env_wrappers.slot_env import SlotActionEnv
        env = SlotActionEnv(env, slot_seconds, gamma_per_second)
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
                              decision_writer=decision_writer, scenario_generator=scenario_generator,
                              scenario_dir=out / "scenarios", gamma_per_second=gamma_per_second)
    finally:
        env.close()
        # Record each episode's floor so per-condition runs (--agvs / --layout / --params) stay self-describing.
        if scenario_generator is not None:
            floor_by_seed = {seed: floor_fields(s) for seed, s in zip(seeds, planned)}
            for row in rows:
                row.update(floor_by_seed.get(row["seed"], {}))
        write_csv(out / "episodes.csv", rows)
        if decision_file is not None:
            decision_file.close()

    if oracle_fixed is not None:
        check_oracle(rows, oracle_fixed)
    summary = summarize(rows, scored=reward_fn is not None, oracle=oracle, best_fixed=best_fixed)
    write_csv(out / "summary.csv", summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
