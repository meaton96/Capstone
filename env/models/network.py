"""
@file network.py
@brief Full DRL Scheduling Network.

@details
Combines the three major components into an end-to-end pipeline:

  1. @ref MultiModalEncoder — five observation modalities → 464-D
  2. @ref FusionHead — 464-D → 256-D with domain randomization
  3. @ref ActorCritic — 256-D → action logits + state value
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.encoder import MultiModalEncoder
from models.actor_critic import FusionHead, ActorCritic
from config import EncoderConfig, FusionConfig, ActorCriticConfig


class SchedulingNetwork(nn.Module):
    """@brief End-to-end DRL network for job-shop scheduling.

    @details
    Observation dict → Encoder (464-D) → Fusion (256-D) → Actor + Critic.

    The network exposes three entry points depending on the call-site:
    - @ref forward — raw logits and value (general use / debugging).
    - @ref act     — action selection during environment rollouts.
    - @ref evaluate — log-prob / value / entropy re-computation for the
      PPO update step.

    @par Example usage
    @code{.py}
    net = SchedulingNetwork()
    obs, info = placeholder_env.reset()
    action, log_prob, value = net.act(obs)
    @endcode
    """

    def __init__(
        self,
        encoder_cfg: EncoderConfig = None,
        fusion_cfg: FusionConfig = None,
        ac_cfg: ActorCriticConfig = None,
    ):
        """@brief Construct the scheduling network.

        @param encoder_cfg  Encoder dimension config.
                            Defaults to EncoderConfig() if None.
        @param fusion_cfg   Fusion-head config (hidden dim, dropout, noise).
                            Defaults to FusionConfig() if None.
        @param ac_cfg       Actor-critic config (hidden dim, num actions).
                            Defaults to ActorCriticConfig() if None.
        """
        super().__init__()
        encoder_cfg = encoder_cfg or EncoderConfig()
        fusion_cfg = fusion_cfg or FusionConfig()
        ac_cfg = ac_cfg or ActorCriticConfig()

        ## @brief Multi-modal encoder producing a 464-D concatenated embedding.
        self.encoder = MultiModalEncoder(encoder_cfg)

        ## @brief Fusion MLP projecting encoder output (464-D → 256-D).
        self.fusion = FusionHead(
            input_dim=encoder_cfg.concat_dim,
            hidden_dim=fusion_cfg.hidden_dim,
            output_dim=fusion_cfg.output_dim,
        )

        ## @brief Shared actor-critic heads consuming the 256-D fused features.
        self.actor_critic = ActorCritic(
            input_dim=fusion_cfg.output_dim,
            hidden_dim=ac_cfg.hidden_dim,
            num_actions=ac_cfg.num_actions,
        )

    def forward(self, obs: dict):
        """@brief Full forward pass through encoder, fusion, and actor-critic.

        @param obs  Observation dict with keys matching the state-space spec
                    (see @ref MultiModalEncoder).
        @return Tuple of:
                - @c action_logits (B, 8) — raw logits over PDR rules.
                - @c value         (B, 1) — state-value estimate.
        """
        encoded = self.encoder(obs)       # (B, 464)
        fused = self.fusion(encoded)      # (B, 256)
        return self.actor_critic(fused)

    def act(self, obs: dict, deterministic: bool = False):
        """@brief Select an action for environment stepping.

        @param obs            Observation dict from the environment.
        @param deterministic  If True, take argmax instead of sampling.
        @return Tuple of:
                - @c action   (B,) — selected PDR rule index.
                - @c log_prob (B,) — log-probability of the selected action.
                - @c value    (B,) — state-value estimate.
        """
        encoded = self.encoder(obs)
        fused = self.fusion(encoded)
        return self.actor_critic.act(fused, deterministic=deterministic)

    def evaluate(self, obs: dict, actions: torch.Tensor):
        """@brief Re-evaluate stored actions for the PPO loss computation.

        @param obs      Observation dict corresponding to the stored
                        transitions.
        @param actions  Previously selected action indices of shape (B,).
        @return Tuple of:
                - @c log_probs (B,) — log-probability of @p actions under
                  the current policy.
                - @c values    (B,) — state-value estimates.
                - @c entropy   (B,) — distribution entropy for the
                  entropy bonus.
        """
        encoded = self.encoder(obs)
        fused = self.fusion(encoded)
        return self.actor_critic.evaluate(fused, actions)

    def get_param_summary(self) -> dict:
        """@brief Parameter-count summary for each submodule.

        @return Dict mapping human-readable submodule names to their
                total number of learnable parameters, plus a @c total key.
        """
        def count(module):
            """@brief Count total parameters in a module.

            @param module  Any nn.Module.
            @return Integer parameter count.
            """
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