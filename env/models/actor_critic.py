"""
@file heads.py
@brief Fusion head and Actor-Critic output heads.

@details
Pipeline: 464-D concat → Fusion (256-D) → Actor (8 PDR actions) + Critic (V).

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
from torch.distributions import Categorical


class FusionHead(nn.Module):
    """@brief Fusion MLP: projects concatenated encoder features to a
    shared representation.

    @details
    Architecture: Linear(@p input_dim, @p hidden_dim) → LayerNorm → SiLU
    → Linear(@p hidden_dim, @p output_dim) → LayerNorm → SiLU.

    Default dimensionality: 464-D → 256-D.
    """

    def __init__(self, input_dim: int = 464, hidden_dim: int = 512,
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
    """@brief Actor network: produces a categorical distribution over PDR rules.

    @details
    Architecture: Linear → LayerNorm → SiLU → Linear(num_actions).
    Output logits are unnormalized log-probabilities.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256,
                 num_actions: int = 8):
        """@brief Construct the actor head.

        @param input_dim    Dimensionality of the fused feature vector.
        @param hidden_dim   Width of the hidden layer.
        @param num_actions  Number of discrete PDR rule actions.
        """
        super().__init__()

        ## @brief Single-hidden-layer MLP producing action logits.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )

        ## @brief Number of discrete actions (PDR rules).
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """@brief Compute raw action logits.

        @param x  Fused feature vector of shape (B, @p input_dim).
        @return Action logits of shape (B, @ref num_actions).
        """
        return self.net(x)

    def get_distribution(self, x: torch.Tensor) -> Categorical:
        """@brief Build a Categorical distribution from the action logits.

        @param x  Fused feature vector of shape (B, @p input_dim).
        @return torch.distributions.Categorical over the PDR actions.
        """
        logits = self.forward(x)
        return Categorical(logits=logits)


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
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256,
                 num_actions: int = 8):
        """@brief Construct the combined actor-critic.

        @param input_dim    Dimensionality of the fused feature vector.
        @param hidden_dim   Width of hidden layers in both heads.
        @param num_actions  Number of discrete PDR rule actions.
        """
        super().__init__()

        ## @brief Policy (actor) head producing action logits.
        self.actor = ActorHead(input_dim, hidden_dim, num_actions)
        ## @brief Value (critic) head producing V(s).
        self.critic = CriticHead(input_dim, hidden_dim)

    def forward(self, features: torch.Tensor):
        """@brief Forward pass returning both action logits and value.

        @param features  Fused representation of shape (B, @p input_dim).
        @return Tuple of (action_logits, value) with shapes (B, 8) and (B, 1).
        """
        return self.actor(features), self.critic(features)

    def act(self, features: torch.Tensor, deterministic: bool = False):
        """@brief Select an action and return associated quantities.

        @param features       Fused representation of shape (B, 256).
        @param deterministic  If True, take argmax instead of sampling
                              from the categorical distribution.
        @return Tuple of:
                - @c action   (B,) — selected PDR rule index.
                - @c log_prob (B,) — log-probability of the selected action.
                - @c value    (B,) — state-value estimate (squeezed).
        """
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, features: torch.Tensor, actions: torch.Tensor):
        """@brief Evaluate previously taken actions for the PPO update step.

        @param features  Fused representation of shape (B, 256).
        @param actions   Previously selected action indices of shape (B,).
        @return Tuple of:
                - @c log_probs (B,) — log-probability of @p actions under
                  the current policy.
                - @c values    (B,) — state-value estimates (squeezed).
                - @c entropy   (B,) — categorical distribution entropy for
                  the entropy bonus.
        """
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value, entropy