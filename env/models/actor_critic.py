"""
Fusion head (with domain randomization) and Actor-Critic output heads.

Pipeline: 464-D concat -> Fusion (256-D) -> Actor (8 PDR actions) + Critic (V)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class DomainRandomization(nn.Module):
    """
    Domain randomization layer: Dropout + Gaussian noise injection.
    Only active during training to improve sim-to-real transfer.
    """

    def __init__(self, dropout_rate: float = 0.1, noise_std: float = 0.01):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return x


class FusionHead(nn.Module):
    """
    Fusion MLP: 464-D -> 256-D with domain randomization.
    """

    def __init__(self, input_dim: int = 464, hidden_dim: int = 512,
                 output_dim: int = 256, dropout_rate: float = 0.1,
                 noise_std: float = 0.01):
        super().__init__()
        self.domain_rand = DomainRandomization(dropout_rate, noise_std)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(inplace=True),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.domain_rand(x)
        return self.net(x)


class ActorHead(nn.Module):
    """
    Actor network: produces a categorical distribution over 8 PDR rules.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256,
                 num_actions: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action logits (B, num_actions)."""
        return self.net(x)

    def get_distribution(self, x: torch.Tensor) -> Categorical:
        logits = self.forward(x)
        return Categorical(logits=logits)


class CriticHead(nn.Module):
    """
    Critic network: estimates state value V(s).
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns value estimate (B, 1)."""
        return self.net(x)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic using shared fusion features.

    Forward pass returns:
        - action_logits: (B, 8)
        - value: (B, 1)
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256,
                 num_actions: int = 8):
        super().__init__()
        self.actor = ActorHead(input_dim, hidden_dim, num_actions)
        self.critic = CriticHead(input_dim, hidden_dim)

    def forward(self, features: torch.Tensor):
        return self.actor(features), self.critic(features)

    def act(self, features: torch.Tensor, deterministic: bool = False):
        """
        Select an action and return (action, log_prob, value).

        Args:
            features: (B, 256) fusion output
            deterministic: if True, take argmax instead of sampling

        Returns:
            action:   (B,) selected PDR rule index
            log_prob: (B,) log probability of selected action
            value:    (B,) state value estimate
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
        """
        Evaluate given actions for PPO update.

        Returns:
            log_probs: (B,) log prob of the given actions
            values:    (B,) state value estimates
            entropy:   (B,) distribution entropy
        """
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value, entropy
