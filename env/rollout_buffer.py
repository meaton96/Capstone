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
from typing import Generator, Optional


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
                 gae_lambda: float = 0.95, device: str = "cpu",
                 action_shape: tuple = (), gamma_per_second: Optional[float] = None,
                 gae_lambda_time_s: Optional[float] = None):
        """@brief Construct the rollout buffer and pre-allocate storage.

        @param rollout_length  Number of environment steps per rollout.
        @param num_envs        Number of parallel environments.
        @param obs_shapes      Dict mapping observation keys to their
                               per-environment shapes (excluding the
                               batch dimension).
        @param gamma           Discount factor per decision (used when @p gamma_per_second is None).
        @param gae_lambda      Lambda parameter for GAE smoothing.
        @param device          Torch device string used when yielding
                               mini-batch tensors.
        @param action_shape    Per-step action shape: () for one discrete
                               action, (n_branches,) for a branched action.
        @param gamma_per_second  If set, discount over simulated time instead (SMDP form): step t
                               is discounted by gamma_per_second ** dts[t]. See compute_gae.
        @param gae_lambda_time_s  If set (needs @p gamma_per_second), λ decays over simulated time too: step t uses
                               gae_lambda ** (dts[t] / gae_lambda_time_s) (audit M2, 2026-10-07). See compute_gae.
        """
        if gae_lambda_time_s is not None and (gamma_per_second is None or gae_lambda_time_s <= 0):
            raise ValueError("gae_lambda_time_s needs gamma_per_second (discounting over simulated time) and must be > 0")
        ## @brief Number of environment steps collected per rollout.
        self.rollout_length = rollout_length
        ## @brief Number of parallel environments.
        self.num_envs = num_envs
        ## @brief Discount factor γ.
        self.gamma = gamma
        ## @brief Discount per simulated second, or None to discount per decision.
        self.gamma_per_second = gamma_per_second
        ## @brief GAE smoothing parameter λ.
        self.gae_lambda = gae_lambda
        ## @brief Seconds per λ factor when λ decays over simulated time, or None for λ per step.
        self.gae_lambda_time_s = gae_lambda_time_s
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

        ## @brief Per-step action shape (see __init__).
        self.action_shape = tuple(action_shape)
        ## @brief Selected action indices (int64), shape (rollout_length, num_envs, *action_shape).
        self.actions = np.zeros((rollout_length, num_envs, *self.action_shape), dtype=np.int64)
        ## @brief Log-probabilities of the selected actions under the
        ##        collection policy.
        self.log_probs = np.zeros((rollout_length, num_envs), dtype=np.float32)
        ## @brief Per-step rewards from the environment.
        self.rewards = np.zeros((rollout_length, num_envs), dtype=np.float32)
        ## @brief State-value estimates V(s) at collection time.
        self.values = np.zeros((rollout_length, num_envs), dtype=np.float32)
        ## @brief Done flags (1.0 = episode ended at this step, terminated or truncated).
        self.dones = np.zeros((rollout_length, num_envs), dtype=np.float32)
        ## @brief Value-bootstrap correction for a step whose episode ended by truncation (a
        ##        deliberate time-limit cutoff) rather than a real terminal — see compute_gae.
        ##        0 everywhere else.
        self.bootstrap_values = np.zeros((rollout_length, num_envs), dtype=np.float32)
        ## @brief Simulated seconds from this step's decision to the next one (Δτ). Used only with
        ##        gamma_per_second; 1.0 is a neutral filler otherwise.
        self.dts = np.ones((rollout_length, num_envs), dtype=np.float32)

        # ---- Computed after rollout ----

        ## @brief GAE advantage estimates, filled by @ref compute_gae.
        self.advantages = np.zeros((rollout_length, num_envs), dtype=np.float32)
        ## @brief Discounted returns (advantages + values), filled by
        ##        @ref compute_gae.
        self.returns = np.zeros((rollout_length, num_envs), dtype=np.float32)

    def add(self, obs: dict, actions, log_probs, rewards, values, dones, bootstrap_values=None, dts=None):
        """@brief Store one timestep of data from all parallel environments.

        @param obs        Observation dict with arrays of shape (num_envs, ...).
        @param actions    Action indices, shape (num_envs, *action_shape).
        @param log_probs  Log-probabilities, shape (num_envs,).
        @param rewards    Rewards, shape (num_envs,).
        @param values     Value estimates, shape (num_envs,).
        @param dones      Done flags, shape (num_envs,) — set for both terminated and truncated
                          episode ends (either way the next stored observation is a new episode).
        @param bootstrap_values  Per-env value estimate V(terminal_obs) (shape (num_envs,)),
                                 meaningful only where this step's episode just ended by
                                 truncation (see train.py); compute_gae discounts and applies it
                                 there and ignores it everywhere else, including real terminations,
                                 so 0 (the default) is a safe filler for every other case.
        @param dts        Simulated seconds this step took, shape (num_envs,). Required when the
                          buffer discounts over simulated time; ignored otherwise.
        """
        if self.gamma_per_second is not None and dts is None:
            raise ValueError("RolloutBuffer discounts over simulated time but add() got no dts")
        for key in self.obs_buffers:
            self.obs_buffers[key][self.pos] = obs[key]
        self.actions[self.pos] = actions
        self.log_probs[self.pos] = log_probs
        self.rewards[self.pos] = rewards
        self.values[self.pos] = values
        self.dones[self.pos] = dones
        self.bootstrap_values[self.pos] = 0.0 if bootstrap_values is None else bootstrap_values
        self.dts[self.pos] = 1.0 if dts is None else dts
        self.pos += 1

    def compute_gae(self, last_values: np.ndarray):
        """@brief Compute GAE advantages and discounted returns.

        @details
        Iterates backwards through the rollout, computing the temporal-
        difference residuals δ_t and accumulating the exponentially
        weighted advantage estimates.  After completion, @ref returns
        is set to @ref advantages + @ref values.

        @c dones[t] is the done flag returned by the step taken from
        observation t, so when it is set the next stored observation
        already belongs to a new episode and must not be bootstrapped from.

        A step whose episode ended by truncation (a deliberate time-limit cutoff, not a real
        terminal) still needs a bootstrap value, but @c self.values[t+1] would be the value of
        the wrong state — the *next* episode's first observation, since Unity has already
        auto-reset by the time the wrapper returns. @ref bootstrap_values[t] carries V(terminal_obs)
        for exactly those steps (0 elsewhere, including real terminations) and substitutes for the
        zeroed-out @c next_values term there, the standard TimeLimit-bootstrap correction (as in
        SB3/CleanRL) generalized to an auto-resetting env.

        Discounting. Decisions are not evenly spaced in simulated time, while the reward already
        integrates over the time between them, so this is a semi-MDP. With @ref gamma_per_second
        set, each step uses its own discount g_t = gamma_per_second ** dts[t] in the bootstrap term
        and the GAE recursion; otherwise g_t = @ref gamma for every step. λ is per step unless
        @ref gae_lambda_time_s is set, when step t uses λ_t = gae_lambda ** (dts[t] / gae_lambda_time_s). Per decision,
        λ = 0.95 spans about 20 decisions (~240 s at ~12 s a decision) against a 3,000-10,800 s discount horizon, so
        the advantage leans on the critic (audit M2); over time it spans the same seconds at any decision rate.

        @param last_values  Bootstrap values V(s_T) for the observation that
                            follows the final stored step, shape (num_envs,).
        """
        gae = np.zeros(self.num_envs, dtype=np.float32)
        if self.gamma_per_second is None:
            discounts = np.full_like(self.dts, self.gamma)
        else:
            discounts = np.power(self.gamma_per_second, self.dts, dtype=np.float64).astype(np.float32)
        if self.gae_lambda_time_s is None:
            lambdas = np.full_like(self.dts, self.gae_lambda)
        else:
            lambdas = np.power(self.gae_lambda, self.dts / self.gae_lambda_time_s, dtype=np.float64).astype(np.float32)
        for t in reversed(range(self.rollout_length)):
            next_values = last_values if t == self.rollout_length - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            g = discounts[t]

            delta = (
                self.rewards[t]
                + g * next_values * next_nonterminal
                + g * self.bootstrap_values[t]
                - self.values[t]
            )
            gae = delta + g * lambdas[t] * next_nonterminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def apply_shared_baseline(self, keys: np.ndarray, groups: np.ndarray, mode: str = "residual") -> dict:
        """@brief Shared-instance baseline (2026-10-07, credit trace item 3; Mao et al. ICLR 2019, Decima): replace each
        step's advantage by a leave-one-out comparison with the other envs that played the same slot of the same
        instance. Call after @ref compute_gae.

        @details Envs of one group draw the same instance seeds (twin_env.VectorizedTwinEnv instance_group), and the twin
        is deterministic given the instance and the actions, so the group's envs differ only in their sampled
        actions. Returns spread about 68% of their mean across instances, and the critic cannot see the instance's
        future arrivals; the other envs on the same instance can. Their returns do not depend on this env's action,
        so subtracting them keeps the policy gradient unbiased.
          - "residual" (baseline next to V): A_i = A_i - mean_{j != i} A_j, where A = λ-return - V is the GAE
            advantage. V keeps what the state says; the group removes what the instance's future adds.
          - "return": A_i = G_i - mean_{j != i} G_j with G the λ-return (the critic is not used for the advantage).
        Steps whose (group, key) has no other member in this rollout keep their GAE advantage. @ref returns (the
        critic's target) is unchanged.

        @param keys    (rollout_length, num_envs) int64: the same value for the same slot of the same instance within
                       a group (train.py: episode ordinal * 2**20 + slot index in the episode); -1 = no key.
        @param groups  (num_envs,) group id of each env.
        @param mode    "residual" or "return".
        @return Stats for logging: share of steps that got a shared baseline, and the advantage variance before and
                after (over those steps).
        """
        if mode not in ("residual", "return"):
            raise ValueError(f"shared baseline mode must be 'residual' or 'return', got {mode!r}")
        values = self.advantages if mode == "residual" else self.returns
        buckets = {}
        for t in range(self.rollout_length):
            for e in range(self.num_envs):
                if keys[t, e] >= 0:
                    buckets.setdefault((int(groups[e]), int(keys[t, e])), []).append((t, e))
        new = self.advantages.copy()
        covered = []
        for members in buckets.values():
            if len(members) < 2:
                continue
            vals = np.array([values[t, e] for t, e in members], dtype=np.float64)
            loo = (vals.sum() - vals) / (len(vals) - 1)
            for (t, e), v, b in zip(members, vals, loo):
                new[t, e] = v - b
                covered.append((t, e))
        stats = {"shared_baseline_share": len(covered) / max(self.rollout_length * self.num_envs, 1)}
        if covered:
            idx = tuple(np.array(covered).T)
            stats["adv_var_before"] = float(np.var(self.advantages[idx]))
            stats["adv_var_after"] = float(np.var(new[idx]))
        self.advantages = new.astype(np.float32)
        return stats

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
            self.actions.reshape(total, *self.action_shape), dtype=torch.long
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