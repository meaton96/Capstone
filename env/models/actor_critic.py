"""
@file heads.py
@brief Fusion head and Actor-Critic output heads.

@details
Pipeline: 560-D concat → Fusion (256-D) → Actor (job head + machine head) + Critic (V).

The actor is factorized: one Categorical per action branch (config.ACTION_BRANCHES). Unity masks a
head that cannot change a decision down to its action 0; a masked head has log-prob 0 and entropy 0,
so each decision's log-prob and entropy count only the heads it used.

The fusion head projects the concatenated encoder features down to a
shared 256-D representation consumed by both the actor and critic
networks.

@note Sensor corruption (dropout, noise for sim-to-real transfer) is
handled by @ref SensorCorruptionWrapper at the observation level, not
inside the network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence

from torch.distributions import Categorical


def branch_distributions(logits: torch.Tensor, branches: Sequence[int],
                         action_mask: Optional[torch.Tensor] = None) -> List[Categorical]:
    """@brief One Categorical per action branch, with masked actions at -1e9.

    @param logits       (B, sum(branches)) concatenated branch logits.
    @param action_mask  (B, sum(branches)), 1 = enabled; None enables every action.
    """
    parts = torch.split(logits, list(branches), dim=-1)
    masks = torch.split(action_mask > 0.5, list(branches), dim=-1) if action_mask is not None else [None] * len(parts)
    return [Categorical(logits=part if mask is None else part.masked_fill(~mask, -1e9))
            for part, mask in zip(parts, masks)]


class FusionHead(nn.Module):
    """@brief Fusion MLP: projects concatenated encoder features to a
    shared representation.

    @details
    Architecture: Linear(@p input_dim, @p hidden_dim) → LayerNorm → SiLU
    → Linear(@p hidden_dim, @p output_dim) → LayerNorm → SiLU.

    Default dimensionality: 560-D → 256-D.
    """

    def __init__(self, input_dim: int = 560, hidden_dim: int = 512,
                 output_dim: int = 256):
        """@brief Construct the fusion head.

        @param input_dim     Dimensionality of the concatenated encoder output.
        @param hidden_dim    Width of the intermediate fully-connected layer.
        @param output_dim    Dimensionality of the shared representation
                             fed to actor and critic heads.
        """
        super().__init__()

        ## @brief Two-layer MLP with LayerNorm and SiLU activations.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(inplace=True),
        )

        ## @brief Output dimensionality exposed for downstream heads.
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """@brief Project concatenated features through the MLP.

        @param x  Concatenated encoder output of shape (B, @p input_dim).
        @return Fused representation of shape (B, @ref output_dim).
        """
        return self.net(x)


class ActorHead(nn.Module):
    """@brief Actor network: one categorical distribution per action branch.

    @details
    Architecture: Linear → LayerNorm → SiLU → Linear(sum(branches)).
    Output logits are unnormalized log-probabilities, branches concatenated.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256,
                 branches: Sequence[int] = (4, 3)):
        """@brief Construct the actor head.

        @param input_dim    Dimensionality of the fused feature vector.
        @param hidden_dim   Width of the hidden layer.
        @param branches     Size of each discrete action branch (job head, machine head).
        """
        num_actions = sum(branches)
        super().__init__()

        ## @brief Single-hidden-layer MLP producing action logits.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )

        ## @brief Size of each action branch.
        self.branches = tuple(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """@brief Compute raw action logits.

        @param x  Fused feature vector of shape (B, @p input_dim).
        @return Action logits of shape (B, sum(@ref branches)).
        """
        return self.net(x)

    def get_distributions(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> List[Categorical]:
        """@brief One (masked) Categorical per branch.

        @param x            Fused feature vector of shape (B, @p input_dim).
        @param action_mask  (B, sum(branches)), 1 = enabled; None enables every action.
        """
        return branch_distributions(self.forward(x), self.branches, action_mask)


class CriticHead(nn.Module):
    """@brief Critic network: estimates the state value V(s).

    @details
    Architecture mirrors ActorHead but outputs a single scalar per
    batch element.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        """@brief Construct the critic head.

        @param input_dim   Dimensionality of the fused feature vector.
        @param hidden_dim  Width of the hidden layer.
        """
        super().__init__()

        ## @brief Single-hidden-layer MLP producing a scalar value estimate.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """@brief Compute the state-value estimate.

        @param x  Fused feature vector of shape (B, @p input_dim).
        @return Value estimate of shape (B, 1).
        """
        return self.net(x)


class ActorCritic(nn.Module):
    """@brief Combined Actor-Critic module using shared fusion features.

    @details
    Wraps an ActorHead and a CriticHead that both consume the same
    256-D fused representation.  Provides convenience methods for
    action selection (@ref act) and PPO-style evaluation (@ref evaluate).
    An action is one index per branch, shape (B, len(branches)); its
    log-prob and entropy are sums over the branches.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256,
                 branches: Sequence[int] = (4, 3)):
        """@brief Construct the combined actor-critic.

        @param input_dim    Dimensionality of the fused feature vector.
        @param hidden_dim   Width of hidden layers in both heads.
        @param branches     Size of each discrete action branch.
        """
        super().__init__()

        ## @brief Policy (actor) head producing concatenated branch logits.
        self.actor = ActorHead(input_dim, hidden_dim, branches)
        ## @brief Value (critic) head producing V(s).
        self.critic = CriticHead(input_dim, hidden_dim)

    def forward(self, features: torch.Tensor):
        """@brief Forward pass returning both action logits and value.

        @param features  Fused representation of shape (B, @p input_dim).
        @return Tuple of (action_logits, value) with shapes (B, sum(branches)) and (B, 1).
        """
        return self.actor(features), self.critic(features)

    def act(self, features: torch.Tensor, action_mask: Optional[torch.Tensor] = None,
            deterministic: bool = False):
        """@brief Select an action and return associated quantities.

        @param features       Fused representation of shape (B, 256).
        @param action_mask    (B, sum(branches)), 1 = enabled; None enables every action.
        @param deterministic  If True, take each branch's argmax instead of sampling.
        @return Tuple of:
                - @c action   (B, n_branches) — index per branch.
                - @c log_prob (B,) — summed log-probability of the selected action.
                - @c value    (B,) — state-value estimate (squeezed).
        """
        dists = self.actor.get_distributions(features, action_mask)
        value = self.critic(features).squeeze(-1)

        if deterministic:
            action = torch.stack([d.logits.argmax(dim=-1) for d in dists], dim=-1)
        else:
            action = torch.stack([d.sample() for d in dists], dim=-1)

        log_prob = sum(d.log_prob(action[:, i]) for i, d in enumerate(dists))
        return action, log_prob, value

    def evaluate(self, features: torch.Tensor, actions: torch.Tensor,
                 action_mask: Optional[torch.Tensor] = None):
        """@brief Evaluate previously taken actions for the PPO update step.

        @param features     Fused representation of shape (B, 256).
        @param actions      Previously selected actions of shape (B, n_branches).
        @param action_mask  The masks those actions were taken under (B, sum(branches)).
        @return Tuple of:
                - @c log_probs (B,) — summed log-probability of @p actions under
                  the current policy.
                - @c values    (B,) — state-value estimates (squeezed).
                - @c entropy   (B,) — summed branch entropy for the entropy bonus
                  (0 for a masked head).
        """
        dists = self.actor.get_distributions(features, action_mask)
        value = self.critic(features).squeeze(-1)

        log_probs = sum(d.log_prob(actions[:, i]) for i, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
        return log_probs, value, entropy
