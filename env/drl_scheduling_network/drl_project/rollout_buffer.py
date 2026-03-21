"""
Rollout buffer for PPO: stores transitions and computes GAE advantages.
"""

import torch
import numpy as np
from typing import Generator


class RolloutBuffer:
    """
    Stores rollout data and computes Generalized Advantage Estimation (GAE).
    """

    def __init__(self, rollout_length: int, num_envs: int,
                 obs_shapes: dict, gamma: float = 0.99,
                 gae_lambda: float = 0.95, device: str = "cpu"):
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.pos = 0

        # Pre-allocate observation buffers
        self.obs_buffers = {}
        for key, shape in obs_shapes.items():
            self.obs_buffers[key] = np.zeros(
                (rollout_length, num_envs, *shape), dtype=np.float32
            )

        # Scalar buffers
        self.actions = np.zeros((rollout_length, num_envs), dtype=np.int64)
        self.log_probs = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.rewards = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_length, num_envs), dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros((rollout_length, num_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_length, num_envs), dtype=np.float32)

    def add(self, obs: dict, actions, log_probs, rewards, values, dones):
        """Store one timestep of data from all envs."""
        for key in self.obs_buffers:
            self.obs_buffers[key][self.pos] = obs[key]
        self.actions[self.pos] = actions
        self.log_probs[self.pos] = log_probs
        self.rewards[self.pos] = rewards
        self.values[self.pos] = values
        self.dones[self.pos] = dones
        self.pos += 1

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray):
        """
        Compute GAE advantages and discounted returns.

        Args:
            last_values: V(s_{T+1}) bootstrap values, shape (num_envs,)
            last_dones:  done flags at T+1, shape (num_envs,)
        """
        gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_values = last_values
                next_nonterminal = 1.0 - last_dones
            else:
                next_values = self.values[t + 1]
                next_nonterminal = 1.0 - self.dones[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_values * next_nonterminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int) -> Generator:
        """
        Yield randomized minibatches as tensors.
        Flattens (rollout_length * num_envs) then shuffles.
        """
        total = self.rollout_length * self.num_envs
        indices = np.random.permutation(total)

        # Flatten all buffers
        flat_obs = {
            k: torch.tensor(
                v.reshape(total, *v.shape[2:]), dtype=torch.float32
            ).to(self.device)
            for k, v in self.obs_buffers.items()
        }
        flat_actions = torch.tensor(
            self.actions.reshape(total), dtype=torch.long
        ).to(self.device)
        flat_log_probs = torch.tensor(
            self.log_probs.reshape(total), dtype=torch.float32
        ).to(self.device)
        flat_advantages = torch.tensor(
            self.advantages.reshape(total), dtype=torch.float32
        ).to(self.device)
        flat_returns = torch.tensor(
            self.returns.reshape(total), dtype=torch.float32
        ).to(self.device)

        # Normalize advantages
        flat_advantages = (
            (flat_advantages - flat_advantages.mean())
            / (flat_advantages.std() + 1e-8)
        )

        for start in range(0, total, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            yield {
                "obs": {k: v[idx] for k, v in flat_obs.items()},
                "actions": flat_actions[idx],
                "old_log_probs": flat_log_probs[idx],
                "advantages": flat_advantages[idx],
                "returns": flat_returns[idx],
            }

    def reset(self):
        self.pos = 0
