"""
Full DRL Scheduling Network.

Combines:
  MultiModalEncoder (5 modalities -> 464-D)
  FusionHead        (464-D -> 256-D with domain randomization)
  ActorCritic       (256-D -> action logits + value)
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import MultiModalEncoder
from models.actor_critic import FusionHead, ActorCritic
from config import EncoderConfig, FusionConfig, ActorCriticConfig


class SchedulingNetwork(nn.Module):
    """
    End-to-end DRL network for job-shop scheduling.

    Observation dict -> Encoder (464-D) -> Fusion (256-D) -> Actor + Critic

    Usage:
        net = SchedulingNetwork()
        obs = placeholder_env.reset()
        action, log_prob, value = net.act(obs)
    """

    def __init__(
        self,
        encoder_cfg: EncoderConfig = None,
        fusion_cfg: FusionConfig = None,
        ac_cfg: ActorCriticConfig = None,
    ):
        super().__init__()
        encoder_cfg = encoder_cfg or EncoderConfig()
        fusion_cfg = fusion_cfg or FusionConfig()
        ac_cfg = ac_cfg or ActorCriticConfig()

        self.encoder = MultiModalEncoder(encoder_cfg)
        self.fusion = FusionHead(
            input_dim=encoder_cfg.concat_dim,
            hidden_dim=fusion_cfg.hidden_dim,
            output_dim=fusion_cfg.output_dim,
            dropout_rate=fusion_cfg.dropout_rate,
            noise_std=fusion_cfg.noise_std,
        )
        self.actor_critic = ActorCritic(
            input_dim=fusion_cfg.output_dim,
            hidden_dim=ac_cfg.hidden_dim,
            num_actions=ac_cfg.num_actions,
        )

    def forward(self, obs: dict):
        """
        Full forward pass.

        Returns:
            action_logits: (B, 8) raw logits over PDR rules
            value:         (B, 1) state value estimate
        """
        encoded = self.encoder(obs)       # (B, 464)
        fused = self.fusion(encoded)      # (B, 256)
        return self.actor_critic(fused)

    def act(self, obs: dict, deterministic: bool = False):
        """
        Select action for environment stepping.

        Returns:
            action:   (B,) PDR index
            log_prob: (B,) log probability
            value:    (B,) value estimate
        """
        encoded = self.encoder(obs)
        fused = self.fusion(encoded)
        return self.actor_critic.act(fused, deterministic=deterministic)

    def evaluate(self, obs: dict, actions: torch.Tensor):
        """
        Evaluate stored actions for PPO loss computation.

        Returns:
            log_probs: (B,)
            values:    (B,)
            entropy:   (B,)
        """
        encoded = self.encoder(obs)
        fused = self.fusion(encoded)
        return self.actor_critic.evaluate(fused, actions)

    def get_param_summary(self) -> dict:
        """Parameter count summary for each submodule."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "encoder_factory_cnn": count(self.encoder.factory_encoder),
            "encoder_sched_cnn": count(self.encoder.sched_encoder),
            "encoder_global_mlp": count(self.encoder.global_mlp),
            "encoder_distance_mlp": count(self.encoder.distance_mlp),
            "encoder_event_embed": count(self.encoder.event_embed),
            "fusion_head": count(self.fusion),
            "actor": count(self.actor_critic.actor),
            "critic": count(self.actor_critic.critic),
            "total": count(self),
        }
