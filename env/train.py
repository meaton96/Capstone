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
import math
import os
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
    EncoderConfig, FusionConfig, ActorCriticConfig, PPOConfig,
)
from models.network import SchedulingNetwork
from rollout_buffer import RolloutBuffer

ENV_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ENV_ROOT.parent


def obs_to_torch(obs: dict, device: str) -> dict:
    """@brief Convert a numpy observation dict to a dict of PyTorch tensors."""
    return {
        k: torch.tensor(v, dtype=torch.float32).to(device)
        for k, v in obs.items()
    }


def build_env(args, ppo_cfg, run_dir: Path, reward):
    """@brief Factory function to create the appropriate vectorized env.

    @param args     Parsed CLI arguments (checked for --unity flag).
    @param ppo_cfg  PPO config with num_envs.
    @param run_dir  Run output directory (Unity player logs go here).
    @param reward   Loaded reward spec for the Unity backend.
    @return Tuple of (vec_env, obs_shapes_dict).
    """
    obs_shapes = {
        "factory_grid":   (3, 64, 64),
        "sched_matrix":   (3, 20, 16),
        "global_scalars": (10,),
        "distance_matrix": (64,),
        "event_flags":    (6,),
    }

    if args.unity:
        from env_wrappers.unity_env import VectorizedUnityEnv
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
        )
    else:
        from env_wrappers.placeholder_env import VectorizedPlaceholderEnv
        vec_env = VectorizedPlaceholderEnv(num_envs=ppo_cfg.num_envs)

    return vec_env, obs_shapes


EPISODE_CSV_FIELDS = [
    "global_step", "env", "seed", "seed_index", "return", "length", "makespan",
    "mean_flow_time", "total_flow_time", "jobs_exited", "deadlock", "timed_out",
]


def log_episodes(writer: SummaryWriter, infos, global_step: int, recent: deque,
                 episode_csv: csv.DictWriter = None) -> int:
    """@brief Write every episode that finished this step to TensorBoard (and episodes.csv).

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
        for key in ("return", "length", "makespan", "mean_flow_time", "jobs_exited"):
            value = episode.get(key)
            if value is not None and not math.isnan(value):
                writer.add_scalar(f"episode/{key}", value, global_step)
        for key in ("deadlock", "timed_out", "interrupted"):
            if key in episode:
                writer.add_scalar(f"episode/{key}", float(episode[key]), global_step)
        for name, total in episode.get("reward_terms", {}).items():
            writer.add_scalar(f"reward_terms/{name}", total, global_step)
    return finished


def save_checkpoint(path: Path, net, optimizer, global_step: int, ppo_cfg: PPOConfig,
                    reward_name):
    """@brief Save network, optimizer, and config so a run can be resumed or evaluated."""
    torch.save({
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "reward": reward_name,
        "config": {
            "encoder": EncoderConfig().__dict__,
            "fusion": FusionConfig().__dict__,
            "actor_critic": ActorCriticConfig().__dict__,
            "ppo": ppo_cfg.__dict__,
        },
    }, path)


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

    reward = None
    if args.unity:
        from rewards import load_reward
        reward = load_reward(args.reward_spec)
        reward.archive(run_dir)
        print(f"\nReward: {reward.name} ({reward.spec['entry']})")

    # ---- Initialize network ----
    net = SchedulingNetwork(
        encoder_cfg=EncoderConfig(),
        fusion_cfg=FusionConfig(),
        ac_cfg=ActorCriticConfig(),
    ).to(device)

    param_summary = net.get_param_summary()
    print("\nParameter counts:")
    for name, count in param_summary.items():
        print(f"  {name:30s} {count:>10,}")

    optimizer = torch.optim.Adam(net.parameters(), lr=ppo_cfg.lr, eps=1e-5)

    # ---- Initialize environments ----
    vec_env, obs_shapes = build_env(args, ppo_cfg, run_dir, reward)
    obs, infos = vec_env.reset()

    buffer = RolloutBuffer(
        rollout_length=ppo_cfg.rollout_length,
        num_envs=ppo_cfg.num_envs,
        obs_shapes=obs_shapes,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        device=device,
    )
    writer = SummaryWriter(log_dir=str(run_dir))

    # ---- Training loop ----
    num_updates = max(1, ppo_cfg.total_timesteps // (
        ppo_cfg.rollout_length * ppo_cfg.num_envs
    ))
    global_step = 0
    episodes_done = 0
    recent_episodes = deque(maxlen=20)
    start_time = time.time()
    reward_name = reward.name if reward is not None else None
    ckpt_path = run_dir / "checkpoint.pt"

    backend = "Unity" if args.unity else "Placeholder"
    print(f"\nBackend: {backend}")
    print(f"Run directory: {run_dir}")
    print(f"Training for {ppo_cfg.total_timesteps:,} timesteps")
    print(f"  {num_updates} updates × {ppo_cfg.rollout_length} steps "
          f"× {ppo_cfg.num_envs} envs")
    print(f"  Device: {device}")
    print()

    episode_file = open(run_dir / "episodes.csv", "w", newline="")
    episode_csv = csv.DictWriter(episode_file, fieldnames=EPISODE_CSV_FIELDS)
    episode_csv.writeheader()

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

                buffer.add(obs, actions_np, log_probs_np, rewards, values_np, dones)
                obs = next_obs
                episodes_done += log_episodes(writer, infos, global_step, recent_episodes, episode_csv)

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
                        + ppo_cfg.entropy_coef * ent_loss
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
            sps = global_step / elapsed
            update_time = time.time() - update_start

            avg_reward = buffer.rewards.mean()
            avg_return = buffer.returns.mean()

            writer.add_scalar("losses/policy", total_pg_loss / n_batches, global_step)
            writer.add_scalar("losses/value", total_v_loss / n_batches, global_step)
            writer.add_scalar("losses/entropy", total_entropy / n_batches, global_step)
            writer.add_scalar("charts/step_reward_mean", avg_reward, global_step)
            writer.add_scalar("charts/sps", sps, global_step)
            writer.add_scalar("charts/episodes", episodes_done, global_step)

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
                save_checkpoint(ckpt_path, net, optimizer, global_step, ppo_cfg, reward_name)
    finally:
        # Save and shut Unity down even on Ctrl-C, so long runs keep their progress.
        save_checkpoint(ckpt_path, net, optimizer, global_step, ppo_cfg, reward_name)
        writer.close()
        episode_file.close()
        if args.unity and hasattr(vec_env, 'close'):
            vec_env.close()

    elapsed = time.time() - start_time
    print(f"\nCheckpoint saved to {ckpt_path}")
    print(f"Total training time: {elapsed:.1f}s")
    print(f"Average SPS: {global_step / elapsed:.0f}")

    return net


if __name__ == "__main__":
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

    # Unity-specific flags
    parser.add_argument("--unity", action="store_true",
                        help="Connect to Unity instead of using placeholder env")
    parser.add_argument("--unity-path", type=str, default=None,
                        help="Path to built Unity executable (None = Editor)")
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
                             "reproducible; -1 lets Unity continue its own random stream")
    parser.add_argument("--base-worker-id", type=int, default=0,
                        help="Unity port offset (5005 + id); change to run several trainings at once")

    args = parser.parse_args()

    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    train(cfg, args, device=args.device)
