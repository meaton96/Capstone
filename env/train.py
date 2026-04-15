"""
@file train.py
@brief PPO Training Loop for the DRL Scheduling Network.

@details
Supports two environment backends:
  --unity          Connect to Unity Editor or built executable.
  (default)        Use the placeholder synthetic environment.

Domain randomization (noise/dropout) is handled on the C# side by
ObservationBuilder.ApplyDomainRandomization, so no Python-side sensor
corruption wrapper is needed.

@par Usage
@code{.sh}
# Placeholder (no Unity needed):
python train.py --total-timesteps 50000 --num-envs 4

# Unity Editor (single env):
python train.py --unity --num-envs 1

# Built Unity executable (parallel):
python train.py --unity --unity-path ./Build/Factory.x86_64 --num-envs 4
@endcode
"""

import argparse
import time
import sys
import os

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    EncoderConfig, FusionConfig, ActorCriticConfig, PPOConfig, PDR_ACTIONS,
)
from models.network import SchedulingNetwork
from rollout_buffer import RolloutBuffer


def obs_to_torch(obs: dict, device: str) -> dict:
    """@brief Convert a numpy observation dict to a dict of PyTorch tensors."""
    return {
        k: torch.tensor(v, dtype=torch.float32).to(device)
        for k, v in obs.items()
    }


def build_env(args, ppo_cfg):
    """@brief Factory function to create the appropriate vectorized env.

    @param args     Parsed CLI arguments (checked for --unity flag).
    @param ppo_cfg  PPO config with num_envs.
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
        from env.unity_env import VectorizedUnityEnv
        vec_env = VectorizedUnityEnv(
            num_envs=ppo_cfg.num_envs,
            file_name=args.unity_path,
            time_scale=args.time_scale,
        )
    else:
        from env.placeholder_env import VectorizedPlaceholderEnv
        vec_env = VectorizedPlaceholderEnv(num_envs=ppo_cfg.num_envs)

    return vec_env, obs_shapes


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
    vec_env, obs_shapes = build_env(args, ppo_cfg)
    obs, infos = vec_env.reset()

    buffer = RolloutBuffer(
        rollout_length=ppo_cfg.rollout_length,
        num_envs=ppo_cfg.num_envs,
        obs_shapes=obs_shapes,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        device=device,
    )

    # ---- Training loop ----
    num_updates = ppo_cfg.total_timesteps // (
        ppo_cfg.rollout_length * ppo_cfg.num_envs
    )
    global_step = 0
    start_time = time.time()

    backend = "Unity" if args.unity else "Placeholder"
    print(f"\nBackend: {backend}")
    print(f"Training for {ppo_cfg.total_timesteps:,} timesteps")
    print(f"  {num_updates} updates × {ppo_cfg.rollout_length} steps "
          f"× {ppo_cfg.num_envs} envs")
    print(f"  Device: {device}")
    print()

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

        # Bootstrap value for GAE
        with torch.no_grad():
            obs_t = obs_to_torch(obs, device)
            _, _, last_values = net.act(obs_t)
            last_values = last_values.cpu().numpy()

        last_dones = np.zeros(ppo_cfg.num_envs, dtype=np.float32)
        buffer.compute_gae(last_values, last_dones)

        # ---- PPO update ----
        net.train()
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        total_loss_val = 0.0
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
                total_loss_val += loss.item()
                n_batches += 1

        # ---- Logging ----
        elapsed = time.time() - start_time
        sps = global_step / elapsed
        update_time = time.time() - update_start

        avg_reward = buffer.rewards.mean()
        avg_return = buffer.returns.mean()

        if update % 5 == 0 or update == 1:
            print(
                f"Update {update:4d}/{num_updates} | "
                f"Step {global_step:>8,} | "
                f"SPS {sps:6.0f} | "
                f"R_avg {avg_reward:+.4f} | "
                f"Ret {avg_return:+.3f} | "
                f"PG {total_pg_loss/n_batches:.4f} | "
                f"VL {total_v_loss/n_batches:.4f} | "
                f"Ent {total_entropy/n_batches:.3f} | "
                f"T {update_time:.2f}s"
            )

    # ---- Save checkpoint ----
    ckpt_path = "checkpoint.pt"
    torch.save({
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "config": {
            "encoder": EncoderConfig().__dict__,
            "fusion": FusionConfig().__dict__,
            "actor_critic": ActorCriticConfig().__dict__,
            "ppo": ppo_cfg.__dict__,
        },
    }, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")
    print(f"Total training time: {elapsed:.1f}s")
    print(f"Average SPS: {global_step / elapsed:.0f}")

    # Clean up Unity if active.
    if args.unity and hasattr(vec_env, 'close'):
        vec_env.close()

    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DRL Scheduling Agent")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")

    # Unity-specific flags
    parser.add_argument("--unity", action="store_true",
                        help="Connect to Unity instead of using placeholder env")
    parser.add_argument("--unity-path", type=str, default=None,
                        help="Path to built Unity executable (None = Editor)")
    parser.add_argument("--time-scale", type=float, default=20.0,
                        help="Unity simulation speed multiplier")

    args = parser.parse_args()

    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    train(cfg, args, device=args.device)