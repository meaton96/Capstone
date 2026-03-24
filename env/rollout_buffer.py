"""
@file rollout_buffer.py
@brief Rollout buffer for PPO: stores transitions and computes GAE advantages.

@details
Collects experience from parallel environments over a fixed rollout
window, then computes Generalized Advantage Estimation (GAE) and
yields randomised mini-batches for the PPO update epochs.
"""

import torch
import numpy as np
from typing import Generator


class RolloutBuffer:
    """@brief Stores rollout data and computes Generalized Advantage Estimation (GAE).

    @details
    Pre-allocates numpy arrays for observations, actions, log-probs,
    rewards, values, and done flags over a fixed
    (@ref rollout_length × @ref num_envs) grid.  After a full rollout,
    @ref compute_gae fills the @ref advantages and @ref returns arrays,
    and @ref get_batches yields shuffled mini-batches as PyTorch tensors
    for the PPO optimisation loop.
    """

    def __init__(self, rollout_length: int, num_envs: int,
                 obs_shapes: dict, gamma: float = 0.99,
                 gae_lambda: float = 0.95, device: str = "cpu"):
        """@brief Construct the rollout buffer and pre-allocate storage.

        @param rollout_length  Number of environment steps per rollout.
        @param num_envs        Number of parallel environments.
        @param obs_shapes      Dict mapping observation keys to their
                               per-environment shapes (excluding the
                               batch dimension).
        @param gamma           Discount factor for future rewards.
        @param gae_lambda      Lambda parameter for GAE smoothing.
        @param device          Torch device string used when yielding
                               mini-batch tensors.
        """
        ## @brief Number of environment steps collected per rollout.
        self.rollout_length = rollout_length
        ## @brief Number of parallel environments.
        self.num_envs = num_envs
        ## @brief Discount factor γ.
        self.gamma = gamma
        ## @brief GAE smoothing parameter λ.
        self.gae_lambda = gae_lambda
        ## @brief Torch device for mini-batch tensors.
        self.device = device
        ## @brief Write cursor into the time dimension (0-indexed).
        self.pos = 0

        ## @brief Dict of pre-allocated observation arrays,
        ##        each of shape (rollout_length, num_envs, *obs_shape).
        self.obs_buffers = {}
        for key, shape in obs_shapes.items():
            self.obs_buffers[key] = np.zeros(
                (rollout_length, num_envs, *shape), dtype=np.float32
            )

        # ---- Scalar buffers (rollout_length, num_envs) ----

        ## @brief Selected action indices (int64).
        self.actions = np.zeros((rollout_length, num_envs), dtype=np.int64)
        ## @brief Log-probabilities of the selected actions under the
        ##        collection policy.
        self.log_probs = np.zeros((rollout_length, num_envs), dtype=np.float32)
        ## @brief Per-step rewards from the environment.
        self.rewards = np.zeros((rollout_length, num_envs), dtype=np.float32)
        ## @brief State-value estimates V(s) at collection time.
        self.values = np.zeros((rollout_length, num_envs), dtype=np.float32)
        ## @brief Done flags (1.0 = episode ended at this step).
        self.dones = np.zeros((rollout_length, num_envs), dtype=np.float32)

        # ---- Computed after rollout ----

        ## @brief GAE advantage estimates, filled by @ref compute_gae.
        self.advantages = np.zeros((rollout_length, num_envs), dtype=np.float32)
        ## @brief Discounted returns (advantages + values), filled by
        ##        @ref compute_gae.
        self.returns = np.zeros((rollout_length, num_envs), dtype=np.float32)

    def add(self, obs: dict, actions, log_probs, rewards, values, dones):
        """@brief Store one timestep of data from all parallel environments.

        @param obs        Observation dict with arrays of shape (num_envs, ...).
        @param actions    Action indices, shape (num_envs,).
        @param log_probs  Log-probabilities, shape (num_envs,).
        @param rewards    Rewards, shape (num_envs,).
        @param values     Value estimates, shape (num_envs,).
        @param dones      Done flags, shape (num_envs,).
        """
        for key in self.obs_buffers:
            self.obs_buffers[key][self.pos] = obs[key]
        self.actions[self.pos] = actions
        self.log_probs[self.pos] = log_probs
        self.rewards[self.pos] = rewards
        self.values[self.pos] = values
        self.dones[self.pos] = dones
        self.pos += 1

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray):
        """@brief Compute GAE advantages and discounted returns.

        @details
        Iterates backwards through the rollout, computing the temporal-
        difference residuals δ_t and accumulating the exponentially
        weighted advantage estimates.  After completion, @ref returns
        is set to @ref advantages + @ref values.

        @param last_values  Bootstrap values V(s_{T+1}), shape (num_envs,).
        @param last_dones   Done flags at T+1, shape (num_envs,).
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
        """@brief Yield randomised mini-batches as PyTorch tensors.

        @details
        Flattens the (rollout_length × num_envs) transitions into a
        single array, shuffles them, normalises the advantages to zero
        mean and unit variance, and yields dicts of index-sliced tensors
        on @ref device.

        Each yielded dict contains the keys: @c obs, @c actions,
        @c old_log_probs, @c advantages, and @c returns.

        @param batch_size  Number of transitions per mini-batch.
        @return Generator of mini-batch dicts.
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
        """@brief Reset the write cursor so the buffer can be reused
        for the next rollout.

        @note Array contents are not zeroed; they will be overwritten
              by subsequent @ref add calls.
        """
        self.pos = 0