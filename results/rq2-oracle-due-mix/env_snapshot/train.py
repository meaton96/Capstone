"""
@file train.py
@brief PPO Training Loop for the DRL Scheduling Network.

@details
Supports two environment backends:
  --unity          Connect to Unity Editor or built executable.
  (default)        Use the placeholder synthetic environment.

With --unity the reward is computed in Python by the function named in --reward-spec
(see env/rewards), so rewards can be changed or compared without rebuilding Unity. Each
run writes to results/<run-id>/: TensorBoard logs (losses, episode metrics, per-term
reward totals), a copy of the reward spec and source, Unity player logs, and checkpoints.

Domain randomization (noise/dropout) is handled on the C# side by
ObservationBuilder.ApplyDomainRandomization, so no Python-side sensor
corruption wrapper is needed.

@par Usage
@code{.sh}
# Placeholder (no Unity needed):
python env/train.py --total-timesteps 50000 --num-envs 4

# Built Unity executable with the flow-time reward:
python env/train.py --unity --unity-path linux_server/capstone.x86_64 --no-graphics \
    --reward-spec env/config/rewards/flow_time.json --run-id flow01 --num-envs 1 --device cuda

tensorboard --logdir results
@endcode
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    EncoderConfig, FusionConfig, ActorCriticConfig, PPOConfig, OBS_SHAPES, OBS_LAYOUT,
    ACTION_BRANCHES, ACTION_LAYOUT, obs_shapes as shapes_for_caps,
)
from models.network import SchedulingNetwork, encoder_config_for
from notify import Watchdog, notify, run_label
from rollout_buffer import RolloutBuffer

ENV_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ENV_ROOT.parent


def obs_to_torch(obs: dict, device: str) -> dict:
    """@brief Convert a numpy observation dict to a dict of PyTorch tensors."""
    return {
        k: torch.tensor(v, dtype=torch.float32).to(device)
        for k, v in obs.items()
    }


def build_env(args, ppo_cfg, run_dir: Path, reward, scenario_generator=None, seed_stream: int = 0):
    """@brief Factory function to create the appropriate vectorized env.

    @param args     Parsed CLI arguments (checked for --unity flag).
    @param ppo_cfg  PPO config with num_envs.
    @param run_dir  Run output directory (Unity player logs go here).
    @param reward   Loaded reward spec for the Unity backend.
    @param scenario_generator  Optional seed -> scenario callable (see scenarios/); each env
                               then gets a fresh scripted-scenario variant every episode.
    @param seed_stream  Instance-seed stream: 0 fresh, the resumed global step on a resume.
    @return Tuple of (vec_env, obs_shapes_dict).
    """
    obs_shapes = dict(OBS_SHAPES)

    if args.twin:
        # Event-based twin (env/des_twin): same interface, episodes simulated in this process.
        from env_wrappers.twin_env import VectorizedTwinEnv
        from scenarios import row_caps_for
        if args.scenario:
            planned = [args.scenario]
        elif scenario_generator is not None:
            planned = [scenario_generator(10_000 + i) for i in range(8)]
        else:
            raise ValueError("--twin needs --scenario or --scenario-generator (the twin has no default job set)")
        obs_caps = row_caps_for(planned, args.obs_max_machines, args.obs_max_jobs)
        obs_shapes = shapes_for_caps(*obs_caps)
        print(f"Observation row caps: {obs_caps[0]} machines, {obs_caps[1]} jobs")
        scenario = None
        if args.scenario:
            with open(args.scenario) as f:
                scenario = json.load(f)
        vec_env = VectorizedTwinEnv(
            num_envs=ppo_cfg.num_envs,
            floor=args.twin,
            transport=args.twin_transport,
            reward_spec=reward,
            train_seed=None if args.train_seed < 0 else args.train_seed,
            scenario_generator=scenario_generator,
            scenario=scenario,
            obs_caps=obs_caps,
            instant_fleet=args.twin_instant_fleet,
            seed_stream=seed_stream,
        )
        shutil.copy2(args.twin, run_dir / "des_floor.json")
    elif args.unity:
        from env_wrappers.unity_env import TRAIN_SEED_LOW, VectorizedUnityEnv
        from scenarios import row_caps_for
        # Row caps fit the floors this run will see, so a small floor isn't padded to the largest one.
        # A generator's floor is fixed per generator; sample a few training seeds to be safe.
        if args.scenario:
            planned = [args.scenario]
        elif scenario_generator is not None:
            planned = [scenario_generator(TRAIN_SEED_LOW + i) for i in range(8)]
        else:
            planned = []   # the player's own default floor: size unknown here, use the defaults
        obs_caps = row_caps_for(planned, args.obs_max_machines, args.obs_max_jobs)
        obs_shapes = shapes_for_caps(*obs_caps)
        print(f"Observation row caps: {obs_caps[0]} machines, {obs_caps[1]} jobs")
        vec_env = VectorizedUnityEnv(
            num_envs=ppo_cfg.num_envs,
            file_name=args.unity_path,
            reward_spec=reward,
            time_scale=args.time_scale,
            base_worker_id=args.base_worker_id,
            no_graphics=args.no_graphics,
            decision_drain=not args.no_decision_drain,
            log_dir=run_dir,
            train_seed=None if args.train_seed < 0 else args.train_seed,
            parallel=not args.sequential_envs,
            scenario_generator=scenario_generator,
            obs_caps=obs_caps,
            seed_stream=seed_stream,
        )
    else:
        from env_wrappers.placeholder_env import VectorizedPlaceholderEnv
        vec_env = VectorizedPlaceholderEnv(num_envs=ppo_cfg.num_envs)

    return vec_env, obs_shapes


EPISODE_CSV_FIELDS = [
    "global_step", "env", "seed", "seed_index", "return", "length", "makespan",
    "mean_flow_time", "total_flow_time", "jobs_exited", "deadlock", "timed_out", "truncated",
    "config_hash", "instance_hash", "tick_error",
]


def log_episodes(writer: SummaryWriter, infos, global_step: int, recent: deque,
                 episode_csv: csv.DictWriter = None, applied_configs=None) -> int:
    """@brief Write every episode that finished this step to TensorBoard (and episodes.csv).

    @param applied_configs  Open text file (applied_configs.jsonl): one line per config hash the first time a
                            player applies it, so every config_hash in episodes.csv resolves to its config.

    @return Number of episodes that finished.
    """
    finished = 0
    for env_index, info in enumerate(infos):
        episode = info.get("episode") if isinstance(info, dict) else None
        if not episode:
            continue
        finished += 1
        recent.append(episode)
        if episode_csv is not None:
            episode_csv.writerow({"global_step": global_step, "env": env_index,
                                  **{k: episode.get(k) for k in EPISODE_CSV_FIELDS[2:]}})
        if applied_configs is not None and episode.get("applied_config"):
            applied_configs.write(json.dumps({"config_hash": episode.get("config_hash"), "env": env_index,
                                              "global_step": global_step, "config": episode["applied_config"]}) + "\n")
        for key in ("return", "length", "makespan", "mean_flow_time", "jobs_exited"):
            value = episode.get(key)
            if value is not None and not math.isnan(value):
                writer.add_scalar(f"episode/{key}", value, global_step)
        for key in ("deadlock", "timed_out", "truncated", "interrupted", "tick_error"):
            if key in episode:
                writer.add_scalar(f"episode/{key}", float(episode[key]), global_step)
        for name, total in episode.get("reward_terms", {}).items():
            writer.add_scalar(f"reward_terms/{name}", total, global_step)
    return finished


def compute_truncation_bootstrap(net, truncateds: np.ndarray, infos, device: str) -> np.ndarray:
    """@brief V(terminal_obs) for each truncated env this step, 0 for every other env.

    @details Unity has already auto-reset a done env by the time step() returns, so obs/next_obs
    is the new episode's first frame — not the state to bootstrap the ended (truncated) one from.
    info["terminal_obs"] (Unity and twin backends) carries that ended episode's last decision
    observation instead (before 2026-10-01 the Unity wrapper passed the player's zero-padded terminal
    observation, so every truncated run bootstrapped from V(zeros)); this is the only place it's used (see rollout_buffer.RolloutBuffer.add /
    compute_gae). The placeholder backend has no "terminal_obs", so its truncations get no
    correction (0, the same as before this existed) — a reasonable fallback, not the backend this
    was built for. A per-env loop is fine here: truncation is rare (only at a scenario's
    time-limit cutoff), typically 0-1 envs per step, not the hot path a batched call would be
    worth optimizing.
    """
    bootstrap = np.zeros(len(truncateds), dtype=np.float32)
    if not truncateds.any():
        return bootstrap
    with torch.no_grad():
        for i in np.flatnonzero(truncateds):
            term_obs = infos[i].get("terminal_obs")
            if term_obs is None:
                continue
            term_obs_t = {k: torch.tensor(v[None], dtype=torch.float32, device=device)
                         for k, v in term_obs.items()}
            _, _, value = net.act(term_obs_t)
            bootstrap[i] = value.item()
    return bootstrap


def step_durations(infos) -> np.ndarray:
    """@brief Simulated seconds each env's step took (info["dt"]), for discounting over simulated time.

    @details Every backend reports it: Unity and twin from the reward-metrics sim_time, the placeholder a
    nominal constant. It is None only when the Unity wrapper has no metrics snapshot (no reward function),
    which cannot train with time discounting, so that is an error rather than a silent per-decision fallback.
    """
    dts = [info.get("dt") for info in infos]
    if any(dt is None for dt in dts):
        raise RuntimeError("Discounting over simulated time needs info['dt'] from every env, but a step "
                           "returned none (Unity without a reward spec?). Pass --discount-horizon-s 0 to "
                           "discount per decision instead.")
    return np.asarray(dts, dtype=np.float32)


def entropy_coef_at(ppo_cfg: PPOConfig, global_step: int) -> float:
    """@brief Entropy coefficient at global_step: constant, or linear decay to entropy_coef_final."""
    if ppo_cfg.entropy_coef_final is None:
        return ppo_cfg.entropy_coef
    frac = min(max(global_step / max(ppo_cfg.total_timesteps, 1), 0.0), 1.0)
    return ppo_cfg.entropy_coef + frac * (ppo_cfg.entropy_coef_final - ppo_cfg.entropy_coef)


def check_obs_schema(ckpt: dict, path) -> None:
    """@brief Refuse a checkpoint whose observation layout this network cannot read, with a clear message.

    @details Compares OBS_LAYOUT (column meanings and per-row widths). Row caps may differ: a network
    trained with 100 machine rows loads into a player built with more. Checkpoints saved before the
    layout was recorded are schema v1 (13,328-float observation) and are always refused.
    """
    saved = ckpt.get("obs_layout")
    if saved is None:
        raise ValueError(
            f"{path}: checkpoint uses observation schema v1 (no obs_layout recorded), this code is "
            f"v{OBS_LAYOUT['schema_version']}. Networks do not transfer across schemas (see env/config.py); "
            f"train a new run.")
    diff = {k: (saved.get(k), v) for k, v in OBS_LAYOUT.items() if saved.get(k) != v}
    if diff:
        detail = ", ".join(f"{k}: checkpoint {a} vs code {b}" for k, (a, b) in diff.items())
        raise ValueError(f"{path}: observation layout differs ({detail}); the network cannot read it.")


def load_checkpoint(path, device) -> dict:
    """@brief Load a checkpoint as tensors and plain containers only (thesis section 7.2).

    @details Plain torch.load unpickles arbitrary objects in the pinned PyTorch (the default only became
    weights_only=True in 2.6), so a shared checkpoint could run code on the machine that loads it.
    save_checkpoint writes only dicts, lists, numbers, strings and tensors, which weights_only accepts.
    """
    return torch.load(path, map_location=device, weights_only=True)


def check_action_layout(ckpt: dict, path) -> None:
    """@brief Refuse a checkpoint trained on a different action space (see config.ACTION_LAYOUT).

    @details Checkpoints saved before action schema v2 have no action_layout: they have a single 8-way
    actor output and cannot drive the two-branch player.
    """
    saved = ckpt.get("action_layout")
    if saved is None:
        raise ValueError(
            f"{path}: checkpoint uses action schema v1 (one 8-way composite-rule branch), this code is "
            f"v{ACTION_LAYOUT['schema_version']} (job head x machine head, see env/config.py); train a new run.")
    if saved != ACTION_LAYOUT:
        raise ValueError(f"{path}: action layout differs (checkpoint {saved} vs code {ACTION_LAYOUT}).")


def save_checkpoint(path: Path, net, optimizer, global_step: int, ppo_cfg: PPOConfig,
                    reward_name, row_caps):
    """@brief Save network, optimizer, and config so a run can be resumed or evaluated.

    @details Keep every value a tensor, dict, list, tuple, str, number, bool or None: load_checkpoint uses
    weights_only=True and refuses anything else (e.g. a Path or a dataclass instance). Written to a temp file and
    renamed, so a kill mid-write never leaves a corrupt checkpoint for a resume to pick up.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "reward": reward_name,
        "obs_layout": dict(OBS_LAYOUT),
        "action_layout": dict(ACTION_LAYOUT),
        "obs_row_caps": {"max_machines": row_caps[0], "max_jobs": row_caps[1]},   # informational only
        "config": {
            "encoder": dict((getattr(net, "encoder_cfg", None) or EncoderConfig()).__dict__),
            "fusion": FusionConfig().__dict__,
            "actor_critic": ActorCriticConfig().__dict__,
            "ppo": ppo_cfg.__dict__,
        },
    }, tmp)
    os.replace(tmp, path)


## @brief Live run status for the watchdog and the crash handler in __main__ (label, step, total, phase).
RUN_STATUS = {"label": "", "step": 0, "total": 0, "phase": "not started"}


def write_progress(run_dir: Path, progress: dict) -> None:
    """@brief Atomically write run_dir/progress.json (read by slurm/train_status.sh)."""
    tmp = run_dir / "progress.json.tmp"
    tmp.write_text(json.dumps(progress, indent=1))
    os.replace(tmp, run_dir / "progress.json")


def progress_line(p: dict) -> str:
    """@brief One-line summary of a progress.json dict."""
    ret = f", recent return {p['recent_return']:.2f}" if p.get("recent_return") is not None else ""
    flow = f", recent mean flow {p['recent_mean_flow']:.0f} s (exited jobs, censored)" \
        if p.get("recent_mean_flow") is not None else ""
    return (f"step {p['global_step']:,}/{p['total_timesteps']:,} "
            f"({100 * p['global_step'] / max(p['total_timesteps'], 1):.1f}%), {p['sps']:.1f} SPS, "
            f"ETA {p['eta_hours']:.1f} h, {p['episodes']} episodes{ret}{flow}")


def train(ppo_cfg: PPOConfig, args, device: str = "cpu"):
    """@brief Main PPO training loop.

    @param ppo_cfg  PPOConfig dataclass with all training hyperparameters.
    @param args     Parsed CLI arguments (env backend, paths, etc.).
    @param device   Torch device string for network and tensor allocation.
    @return The trained SchedulingNetwork instance.
    """
    print("=" * 60)
    print("DRL Scheduling Network — PPO Training")
    print("=" * 60)

    run_dir = Path(args.results_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Progress shared with the watchdog, the heartbeat messages, progress.json and the crash handler in __main__.
    label = run_label(args.run_id)
    RUN_STATUS.update(label=label, step=0, total=ppo_cfg.total_timesteps, phase="starting")
    watchdog = Watchdog(args.stall_minutes, label,
                        status=lambda: f"Last phase: {RUN_STATUS['phase']}, step {RUN_STATUS['step']:,}.")

    reward = None
    if args.unity or args.twin:
        from rewards import load_reward
        reward = load_reward(args.reward_spec)
        reward.archive(run_dir)
        print(f"\nReward: {reward.name} ({reward.spec['entry']})")

    # A resumed checkpoint decides the grid normalization (BatchNorm before 2026-10-01, GroupNorm after).
    ckpt = load_checkpoint(args.resume_from, device) if args.resume_from else None
    # A resume keys its instance seeds, action sampling and minibatch order on its resumed step; with the
    # bare train seed every Slurm leg would replay the first leg's instances in order (audit P1.1).
    seed_stream = int(ckpt["global_step"]) if ckpt else 0

    if args.train_seed >= 0:
        # Same seed -> same initial weights and minibatch order, so runs that differ only in
        # their reward start from an identical policy (GPU kernels can still add small noise).
        rng_seed = args.train_seed if seed_stream == 0 else \
            int(np.random.SeedSequence([args.train_seed, seed_stream]).generate_state(1)[0])
        torch.manual_seed(rng_seed)
        np.random.seed(rng_seed)

    # ---- Initialize network ----
    encoder_cfg = encoder_config_for(ckpt["model_state_dict"]) if ckpt else EncoderConfig()
    if encoder_cfg.grid_norm == "batch":
        print("\n[NOTE] the checkpoint's grid CNN uses BatchNorm (before 2026-10-01). It now stays on its running "
              "statistics during updates too, so its statistics are frozen; the old runs normalized updates with "
              "minibatch statistics.")
    net = SchedulingNetwork(
        encoder_cfg=encoder_cfg,
        fusion_cfg=FusionConfig(),
        ac_cfg=ActorCriticConfig(),
    ).to(device)

    param_summary = net.get_param_summary()
    print("\nParameter counts:")
    for name, count in param_summary.items():
        print(f"  {name:30s} {count:>10,}")

    optimizer = torch.optim.Adam(net.parameters(), lr=ppo_cfg.lr, eps=1e-5)

    resumed_global_step = 0
    if args.resume_from:
        check_obs_schema(ckpt, args.resume_from)
        check_action_layout(ckpt, args.resume_from)
        net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        resumed_global_step = ckpt["global_step"]
        ckpt_reward = ckpt.get("reward")
        print(f"\nResumed from {args.resume_from} at step {resumed_global_step:,} "
              f"(checkpoint reward: {ckpt_reward})")
        ckpt_horizon = ckpt.get("config", {}).get("ppo", {}).get("discount_horizon_s")   # absent = per-decision (before 10-01)
        if ckpt_horizon != ppo_cfg.discount_horizon_s:
            print(f"[WARNING] the checkpoint was trained with discount_horizon_s={ckpt_horizon} "
                  f"(None = per-decision gamma) but this run uses {ppo_cfg.discount_horizon_s} -- "
                  "continuing anyway, but this changes the objective mid-training.")
        if (args.unity or args.twin) and ckpt_reward is not None and ckpt_reward != reward.name:
            print(f"[WARNING] --reward-spec is '{reward.name}' but the checkpoint was trained "
                  f"with '{ckpt_reward}' -- continuing anyway, but this changes the objective "
                  "mid-training.")

    # ---- Scripted scenario: a fixed file, a generator of fresh seeded variants, or neither ----
    scenario_generator = None
    if (args.unity or args.twin) and args.scenario_generator:
        if args.scenario:
            raise ValueError("--scenario and --scenario-generator are mutually exclusive")
        if args.train_seed < 0:
            raise ValueError("--scenario-generator needs --train-seed >= 0, to draw variant seeds")
        from scenarios import REGISTRY
        duration = args.episode_duration_seconds if args.episode_duration_seconds > 0 else None
        extra = {}
        if args.params is not None:
            if args.scenario_generator != "randomized":
                raise ValueError("--params applies to the randomized generator only")
            extra["params_overrides"] = json.loads(args.params)
            if not isinstance(extra["params_overrides"], dict):
                raise ValueError("--params: expected a JSON object")
        scenario_generator = REGISTRY[args.scenario_generator](
            duration, random_warmup=args.random_warmup,
            warmup_dispatching_rule=args.warmup_dispatching_rule,
            agv_move_speed=args.agv_move_speed, agv_handshake_duration=args.agv_handshake_duration,
            machine_flexibility=args.machine_flexibility, secondary_time_multiplier=args.secondary_time_multiplier,
            **extra)
        print(f"\nScenario generator: {args.scenario_generator}"
              + (f" (episode_duration_seconds={args.episode_duration_seconds})"
                 if args.episode_duration_seconds else "")
              + (f" (random_warmup, rule={args.warmup_dispatching_rule or 'default'})"
                 if args.random_warmup else "")
              + (f" (agv_move_speed={args.agv_move_speed})" if args.agv_move_speed else "")
              + (f" (agv_handshake_duration={args.agv_handshake_duration})"
                 if args.agv_handshake_duration else "")
              + (f" (machine_flexibility={args.machine_flexibility}, secondary_time_multiplier="
                 f"{args.secondary_time_multiplier})" if args.machine_flexibility else "")
              + (f" (params {args.params})" if args.params else ""))

    # ---- Initialize environments ----
    vec_env, obs_shapes = build_env(args, ppo_cfg, run_dir, reward, scenario_generator, seed_stream)
    row_caps = (obs_shapes["machine_table"][0], obs_shapes["job_table"][0])
    if (args.unity or args.twin) and args.scenario:
        if args.unity:
            vec_env.load_scenario_all(args.scenario)
        shutil.copy2(args.scenario, run_dir / "scenario.json")
        print(f"Scenario: {args.scenario}")
    obs, infos = vec_env.reset()
    watchdog.beat()
    RUN_STATUS["phase"] = "collecting the first rollout"

    buffer = RolloutBuffer(
        rollout_length=ppo_cfg.rollout_length,
        num_envs=ppo_cfg.num_envs,
        obs_shapes=obs_shapes,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        device=device,
        action_shape=(len(ACTION_BRANCHES),),
        gamma_per_second=ppo_cfg.gamma_per_second,
    )
    if ppo_cfg.gamma_per_second is None:
        print(f"Discount: gamma = {ppo_cfg.gamma} per decision")
    else:
        print(f"Discount: gamma_s = {ppo_cfg.gamma_per_second:.6f} per simulated second "
              f"(horizon {ppo_cfg.discount_horizon_s:,.0f} s)")
    writer = SummaryWriter(log_dir=str(run_dir))

    # ---- Training loop ----
    # total_timesteps is always the target CUMULATIVE step count, resumed or not -- resuming at
    # step 300k with --total-timesteps 500k trains 200k more, not 500k more.
    steps_per_update = ppo_cfg.rollout_length * ppo_cfg.num_envs
    remaining_steps = max(0, ppo_cfg.total_timesteps - resumed_global_step)
    num_updates = remaining_steps // steps_per_update
    if not args.resume_from:
        num_updates = max(1, num_updates)   # unresumed runs always do at least one update
    global_step = resumed_global_step
    RUN_STATUS["step"] = global_step
    last_heartbeat = time.time()
    episodes_done = 0
    recent_episodes = deque(maxlen=20)
    start_time = time.time()
    reward_name = reward.name if reward is not None else None
    ckpt_path = run_dir / "checkpoint.pt"
    if not args.resume_from:
        # Untrained starting policy, as a reference point for evaluation. Skipped when resuming
        # -- that file already exists from the original run and still means "untrained".
        save_checkpoint(run_dir / "checkpoint_init.pt", net, optimizer, 0, ppo_cfg, reward_name, row_caps)

    backend = (f"twin ({args.twin_transport}, {args.twin})" if args.twin
               else "Unity" if args.unity else "Placeholder")
    print(f"\nBackend: {backend}")
    print(f"Run directory: {run_dir}")
    if args.resume_from:
        print(f"Resuming at step {global_step:,} — {remaining_steps:,} more of "
              f"{ppo_cfg.total_timesteps:,} target timesteps")
    else:
        print(f"Training for {ppo_cfg.total_timesteps:,} timesteps")
    print(f"  {num_updates} updates × {ppo_cfg.rollout_length} steps "
          f"× {ppo_cfg.num_envs} envs")
    print(f"  Device: {device}")
    print()
    if num_updates == 0:
        print("Already at or past --total-timesteps -- nothing to do. Pass a larger "
              "--total-timesteps to train further.\n")

    # Append when resuming into the same run_dir (keeps prior episode history); write mode
    # otherwise. Header only if the file is new/empty either way.
    episode_file_path = run_dir / "episodes.csv"
    resuming_existing_log = bool(args.resume_from) and episode_file_path.exists() \
        and episode_file_path.stat().st_size > 0
    # A log resumed from before a column existed keeps its own header (new columns are dropped, not misaligned).
    fieldnames = EPISODE_CSV_FIELDS
    if resuming_existing_log:
        with open(episode_file_path, newline="") as f:
            fieldnames = next(csv.reader(f))
    episode_file = open(episode_file_path, "a" if args.resume_from else "w", newline="")
    episode_csv = csv.DictWriter(episode_file, fieldnames=fieldnames, extrasaction="ignore")
    if not resuming_existing_log:
        episode_csv.writeheader()
    # Canonical text of every config Unity applied, keyed by the config_hash column of episodes.csv (thesis 7.1).
    applied_configs_file = open(run_dir / "applied_configs.jsonl", "a" if args.resume_from else "w")

    try:
        for update in range(1, num_updates + 1):
            update_start = time.time()

            # ---- Collect rollout ----
            net.eval()
            buffer.reset()

            for step in range(ppo_cfg.rollout_length):
                global_step += ppo_cfg.num_envs
                obs_t = obs_to_torch(obs, device)

                with torch.no_grad():
                    actions, log_probs, values = net.act(obs_t)

                actions_np = actions.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()
                values_np = values.cpu().numpy()

                next_obs, rewards, terminateds, truncateds, infos = vec_env.step(
                    actions_np
                )
                dones = np.logical_or(terminateds, truncateds).astype(np.float32)
                bootstrap_values = compute_truncation_bootstrap(net, truncateds, infos, device)
                dts = step_durations(infos) if ppo_cfg.gamma_per_second is not None else None

                buffer.add(obs, actions_np, log_probs_np, rewards, values_np, dones, bootstrap_values, dts)
                obs = next_obs
                watchdog.beat()
                finished_now = log_episodes(writer, infos, global_step, recent_episodes, episode_csv,
                                            applied_configs_file)
                if finished_now:
                    # On the cluster's parallel filesystem Python buffers a whole block (MBs), so without this
                    # episodes.csv stays empty for days and its rows are lost if the job is hard-killed.
                    episode_file.flush()
                    applied_configs_file.flush()
                episodes_done += finished_now

            # Bootstrap value for GAE
            with torch.no_grad():
                obs_t = obs_to_torch(obs, device)
                _, _, last_values = net.act(obs_t)
                last_values = last_values.cpu().numpy()

            buffer.compute_gae(last_values)

            # ---- PPO update ----
            net.train()
            total_pg_loss = 0.0
            total_v_loss = 0.0
            total_entropy = 0.0
            n_batches = 0

            ent_coef = entropy_coef_at(ppo_cfg, global_step)
            for epoch in range(ppo_cfg.num_epochs):
                for batch in buffer.get_batches(ppo_cfg.batch_size):
                    new_log_probs, new_values, entropy = net.evaluate(
                        batch["obs"], batch["actions"]
                    )

                    ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                    surr1 = ratio * batch["advantages"]
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1.0 - ppo_cfg.clip_epsilon,
                            1.0 + ppo_cfg.clip_epsilon,
                        )
                        * batch["advantages"]
                    )
                    pg_loss = -torch.min(surr1, surr2).mean()

                    v_loss = nn.functional.mse_loss(new_values, batch["returns"])
                    ent_loss = -entropy.mean()

                    loss = (
                        pg_loss
                        + ppo_cfg.value_coef * v_loss
                        + ent_coef * ent_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        net.parameters(), ppo_cfg.max_grad_norm
                    )
                    optimizer.step()

                    total_pg_loss += pg_loss.item()
                    total_v_loss += v_loss.item()
                    total_entropy += -ent_loss.item()
                    n_batches += 1

            # ---- Logging ----
            elapsed = time.time() - start_time
            # global_step - resumed_global_step, not global_step: SPS is this invocation's
            # throughput, and global_step alone (absolute, including a resumed run's prior
            # steps) divided by only this invocation's elapsed time would be inflated.
            sps = (global_step - resumed_global_step) / elapsed
            update_time = time.time() - update_start

            avg_reward = buffer.rewards.mean()
            avg_return = buffer.returns.mean()

            writer.add_scalar("losses/policy", total_pg_loss / n_batches, global_step)
            writer.add_scalar("losses/value", total_v_loss / n_batches, global_step)
            writer.add_scalar("losses/entropy", total_entropy / n_batches, global_step)
            writer.add_scalar("charts/entropy_coef", ent_coef, global_step)
            writer.add_scalar("charts/step_reward_mean", avg_reward, global_step)
            writer.add_scalar("charts/sps", sps, global_step)
            writer.add_scalar("charts/episodes", episodes_done, global_step)
            # Share of this rollout's decisions each head could change (Unity leaves more than action 0 enabled).
            offset = 0
            for name, size in zip(("job", "machine"), ACTION_BRANCHES):
                used = buffer.obs_buffers["action_mask"][..., offset + 1:offset + size].sum(-1) > 0
                writer.add_scalar(f"charts/{name}_head_used", float(used.mean()), global_step)
                offset += size

            if update % 5 == 0 or update == 1:
                episode_stats = ""
                if recent_episodes:
                    ep_return = np.mean([e["return"] for e in recent_episodes])
                    makespans = [e["makespan"] for e in recent_episodes if "makespan" in e]
                    episode_stats = f"Ep {episodes_done:5d} | EpRet {ep_return:+.3f} | "
                    if makespans:
                        episode_stats += f"Makespan {np.mean(makespans):6.0f} | "
                print(
                    f"Update {update:4d}/{num_updates} | "
                    f"Step {global_step:>8,} | "
                    f"SPS {sps:6.0f} | "
                    f"{episode_stats}"
                    f"R_avg {avg_reward:+.4f} | "
                    f"Ret {avg_return:+.3f} | "
                    f"PG {total_pg_loss/n_batches:.4f} | "
                    f"VL {total_v_loss/n_batches:.4f} | "
                    f"Ent {total_entropy/n_batches:.3f} | "
                    f"T {update_time:.2f}s"
                )

            if args.save_every > 0 and update % args.save_every == 0:
                save_checkpoint(run_dir / f"checkpoint_step{global_step}.pt", net, optimizer,
                                global_step, ppo_cfg, reward_name, row_caps)

            # ---- Progress file, heartbeat messages, watchdog ----
            watchdog.beat()
            RUN_STATUS.update(step=global_step, phase=f"training (update {update}/{num_updates})")
            eta_h = (ppo_cfg.total_timesteps - global_step) / sps / 3600 if sps > 0 else float("nan")
            flows = [e["mean_flow_time"] for e in recent_episodes
                     if isinstance(e.get("mean_flow_time"), float) and not math.isnan(e["mean_flow_time"])]
            progress = {
                "run_id": args.run_id, "label": label, "update": update, "num_updates": num_updates,
                "global_step": global_step, "total_timesteps": ppo_cfg.total_timesteps,
                "sps": round(sps, 2), "eta_hours": round(eta_h, 2), "episodes": episodes_done,
                "recent_return": round(float(np.mean([e["return"] for e in recent_episodes])), 3)
                if recent_episodes else None,
                "recent_mean_flow": round(float(np.mean(flows)), 1) if flows else None,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            write_progress(run_dir, progress)
            if update == 1 or time.time() - last_heartbeat >= args.notify_every_hours * 3600:
                head = ":white_check_mark: **training started**" if update == 1 else ":hourglass: progress"
                notify(f"{head} {label}: {progress_line(progress)}")
                last_heartbeat = time.time()
    finally:
        # Save and shut Unity down even on Ctrl-C, so long runs keep their progress.
        save_checkpoint(ckpt_path, net, optimizer, global_step, ppo_cfg, reward_name, row_caps)
        writer.close()
        episode_file.close()
        applied_configs_file.close()
        if (args.unity or args.twin) and hasattr(vec_env, 'close'):
            vec_env.close()

    elapsed = time.time() - start_time
    watchdog.stop()
    RUN_STATUS["phase"] = "finished"
    notify(f":checkered_flag: **finished** {label}: step {global_step:,} of {ppo_cfg.total_timesteps:,} "
           f"in {elapsed / 3600:.1f} h. Checkpoint: {ckpt_path}")
    print(f"\nCheckpoint saved to {ckpt_path}")
    print(f"Total training time: {elapsed:.1f}s")
    print(f"Average SPS: {(global_step - resumed_global_step) / elapsed:.0f}")

    return net


if __name__ == "__main__":
    # Line-buffer stdout so progress lines still appear when output is redirected to a file.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Train DRL Scheduling Agent")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-id", type=str, default=datetime.now().strftime("%Y%m%d-%H%M%S"),
                        help="Output subdirectory name under --results-dir")
    parser.add_argument("--results-dir", type=str, default=str(REPO_ROOT / "results"))
    parser.add_argument("--save-every", type=int, default=0,
                        help="Also checkpoint every N updates (0 = only at the end)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to a checkpoint .pt (e.g. results/<run>/checkpoint.pt) to "
                             "resume from: loads model/optimizer state and continues global_step "
                             "from where it left off, training up to --total-timesteps total "
                             "(not +total-timesteps more). RNG state isn't saved/restored, so the "
                             "resumed run's minibatch order isn't a bitwise continuation of the "
                             "original -- only the learned weights and optimizer state are.")

    # Unity-specific flags
    parser.add_argument("--unity", action="store_true",
                        help="Connect to Unity instead of using placeholder env")
    parser.add_argument("--unity-path", type=str, default=None,
                        help="Path to built Unity executable (None = Editor)")
    # No effect since 2026-10-02, when the launch-time check against BUILD_MANIFEST.json was removed
    # (see env/player_manifest.py); still accepted so older commands run.
    parser.add_argument("--allow-unverified-player", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--twin", type=str, default=None, metavar="DES_FLOOR_JSON",
                        help="Train on the event-based twin (env/des_twin) instead of Unity: the floor exported by a "
                             "player with -destrace (built after 2026-10-01) for the layout and fleet size trained "
                             "on. Needs --scenario or --scenario-generator; failures and flexibility are refused")
    parser.add_argument("--twin-transport", type=str, default="kinematic",
                        choices=["instant", "geometric", "kinematic"],
                        help="Twin transport model: instant (DES-0), geometric (DES-1g) or kinematic (DES-1k)")
    parser.add_argument("--twin-instant-fleet", type=int, default=None,
                        help="AGVs a DES-0 observation reports, parked and idle (default: the floor's fleet)")
    parser.add_argument("--time-scale", type=float, default=100.0,
                        help="Unity simulation speed multiplier")
    parser.add_argument("--reward-spec", type=str,
                        default=str(ENV_ROOT / "config" / "rewards" / "flow_time.json"),
                        help="JSON reward spec (see env/rewards)")
    parser.add_argument("--no-graphics", action="store_true",
                        help="Run the Unity player without rendering")
    parser.add_argument("--no-decision-drain", action="store_true",
                        help="Use ML-Agents' per-FixedUpdate stepping instead of one step per decision")
    parser.add_argument("--train-seed", type=int, default=0,
                        help="Seed for per-episode instance seeds, making training instances "
                             "reproducible; -1 sends no seeds, so every episode replays the config's "
                             "own seed (Unity rebuilds the factory each episode)")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Scripted scenario JSON (ScenarioLoader schema) to replay every episode "
                             "(mutually exclusive with --scenario-generator)")
    from scenarios import REGISTRY as _SCENARIO_REGISTRY
    parser.add_argument("--scenario-generator", type=str, default=None,
                        choices=sorted(_SCENARIO_REGISTRY),
                        help="Generate a fresh seeded scripted-scenario variant every episode "
                             "(see env/scenarios); needs --train-seed >= 0")
    parser.add_argument("--params", type=str, default=None, metavar="JSON",
                        help="With --scenario-generator randomized: RandomizedParams overrides (env/scenarios/"
                             "randomized.py), as in evaluate.py; e.g. '{\"failure_probability\": 0}' for --twin, "
                             "which refuses machine-failure scenarios")
    parser.add_argument("--episode-duration-seconds", type=float, default=0.0,
                        help="With --scenario-generator, cap each episode at this many sim-seconds "
                             "(steady-state mode: in-flight jobs censored, episode truncated not "
                             "terminated); 0 runs the scenario to its natural length")
    parser.add_argument("--random-warmup", action="store_true",
                        help="With --scenario-generator, run each episode's floor forward under "
                             "--warmup-dispatching-rule to a random phase-boundary offset (drawn "
                             "from the episode's own seed) before the RL agent takes over, instead "
                             "of always starting at t=0 -- gives phase-diverse exposure at cheap "
                             "episode length; combine with --episode-duration-seconds for a random "
                             "phase-aligned window instead of always phases 1-3")
    parser.add_argument("--warmup-dispatching-rule", type=str, default=None,
                        help="DispatchingRule name (e.g. SPT_SMPT) driving the --random-warmup "
                             "window; unset: compound uses the scenario's own rule (SRT_SRWT), randomized "
                             "rotates SPT_SMPT / SPT_SRWT / SRT_SRWT / SRT_SMPT by seed % 4")
    parser.add_argument("--agv-move-speed", type=float, default=None,
                        help="With --scenario-generator, overrides AGV travel speed (units/sim-"
                             "second; prefab default 3.5). Faster AGVs shrink physical transit "
                             "time relative to job processing time without changing the scripted "
                             "arrival rate, so dispatch decisions are less often degenerate (<=1 "
                             "real candidate) purely from AGV transit spacing jobs out -- see "
                             "dispatch-degeneracy-regime-dependence")
    parser.add_argument("--agv-handshake-duration", type=float, default=None,
                        help="With --scenario-generator, overrides AGV pickup/dropoff handshake "
                             "time (sim-seconds; prefab default 1.5)")
    parser.add_argument("--machine-flexibility", type=float, default=0.0,
                        help="With --scenario-generator: probability that a machine can also run each other "
                             "operation type (0 = fully typed; see FJSSPConfig.MachineFlexibilityProbability)")
    parser.add_argument("--secondary-time-multiplier", type=float, default=1.0,
                        help="With --machine-flexibility: processing-time factor on a machine's secondary types")
    parser.add_argument("--sequential-envs", action="store_true",
                        help="Step Unity envs one after another instead of concurrently")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--discount-horizon-s", type=float, default=PPOConfig.discount_horizon_s,
                        help="Discount over simulated time: gamma_s = 1 - 1/H per second (SMDP). "
                             "0 = per-decision gamma 0.99, as in every run before 2026-10-01")
    parser.add_argument("--ent-coef-final", type=float, default=None,
                        help="If set, decay the entropy coefficient linearly to this value over --total-timesteps")
    parser.add_argument("--torch-threads", type=int, default=0,
                        help="torch.set_num_threads (0 = torch default); keep low when sharing a node with Unity players")
    parser.add_argument("--base-worker-id", type=int, default=0,
                        help="Unity port offset (5005 + id); change to run several trainings at once")
    parser.add_argument("--stall-minutes", type=float, default=45.0,
                        help="Exit with code 3 (after a webhook alert) when no env step or update happens for this "
                             "long, startup included; 0 disables. See env/notify.py")
    parser.add_argument("--notify-every-hours", type=float, default=6.0,
                        help="Webhook progress message interval (plus one after the first update, and on finish, "
                             "stop or crash). The webhook is $NOTIFY_WEBHOOK_URL or ~/.capstone_webhook; none = off")
    parser.add_argument("--obs-max-machines", type=int, default=0,
                        help="Observation machine rows (0 = fit the largest floor in the run's scenarios)")
    parser.add_argument("--obs-max-jobs", type=int, default=0,
                        help="Observation job rows (0 = 256 per 15 machines of the largest floor)")

    args = parser.parse_args()
    if args.twin and args.unity:
        parser.error("--twin and --unity are mutually exclusive")

    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        lr=args.lr,
        entropy_coef=args.ent_coef,
        entropy_coef_final=args.ent_coef_final,
        discount_horizon_s=args.discount_horizon_s if args.discount_horizon_s > 0 else None,
    )
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    # Slurm ends a job with SIGTERM (at --time, or early with #SBATCH --signal). Python's default SIGTERM
    # exits without running train()'s finally block, so the final checkpoint.pt would be lost; raising
    # KeyboardInterrupt takes the same save-and-close path as Ctrl-C.
    import signal

    def _terminate(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _terminate)
    try:
        train(cfg, args, device=args.device)
    except KeyboardInterrupt as e:
        # SIGTERM from Slurm (time limit / scancel) or Ctrl-C: train()'s finally already saved checkpoint.pt.
        notify(f":pause_button: **stopped** {RUN_STATUS['label']} ({e}) at step {RUN_STATUS['step']:,} of "
               f"{RUN_STATUS['total']:,}; checkpoint saved. Resubmit the same command to resume.")
        raise
    except BaseException:
        import traceback
        tail = "".join(traceback.format_exc().splitlines(keepends=True)[-12:])
        notify(f":x: **crashed** {RUN_STATUS['label']} during {RUN_STATUS['phase']} at step "
               f"{RUN_STATUS['step']:,}:\n```\n{tail}```")
        raise
