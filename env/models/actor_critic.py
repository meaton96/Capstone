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

@note The network applies no sensor corruption (dropout, noise); none is
implemented today (env_wrappers/sensor_corruption.py is commented out).
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


def prior_log_probs(pair: str, prob: float, heads: Sequence[Sequence[str]]) -> torch.Tensor:
    """@brief Log-probabilities of the action prior p0, branches concatenated (action-prior branch, 2026-10-06).

    @param pair   Default pair "JOB-MACHINE", one rule per head (e.g. "MDD-TECT").
    @param prob   Probability each head puts on the pair's rule; the rest is spread evenly over the head's others.
    @param heads  Rule names of each head (config.JOB_HEAD_RULES, config.MACHINE_HEAD_RULES).
    """
    rules = pair.split("-")
    if len(rules) != len(heads):
        raise ValueError(f"prior pair {pair!r} needs one rule per head ({len(heads)})")
    if not 0.0 < prob < 1.0:
        raise ValueError(f"prior_prob must be in (0, 1), got {prob}")
    out = []
    for rule, names in zip(rules, heads):
        if rule not in names:
            raise ValueError(f"prior rule {rule!r} is not in head {list(names)}")
        p = torch.full((len(names),), (1.0 - prob) / (len(names) - 1))
        p[list(names).index(rule)] = prob
        out.append(p.log())
    return torch.cat(out)


def kl_to_prior(dists: List[Categorical], prior_logp: torch.Tensor, branches: Sequence[int],
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """@brief KL(pi || p0) summed over the branches, (B,). Under a mask p0 is renormalized over the enabled actions,
    so a head masked down to one action contributes 0 (as it does to log-prob and entropy)."""
    parts = torch.split(prior_logp.to(dists[0].logits.device), list(branches))
    masks = torch.split(action_mask > 0.5, list(branches), dim=-1) if action_mask is not None else [None] * len(parts)
    kl = 0.0
    for d, lp, m in zip(dists, parts, masks):
        lp = lp.expand_as(d.logits)
        if m is not None:
            lp = lp.masked_fill(~m, -1e9)
        lp = F.log_softmax(lp, dim=-1)
        kl = kl + (d.probs * (d.logits - lp)).sum(-1)
    return kl


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
        ## @brief Fixed logit offset added to the network's output: log p0 of the action prior, or zeros (no prior).
        ##        A buffer, so it is saved in checkpoints and rollout, update and evaluate.py all apply it
        ##        (2026-10-07, audit M1).
        self.register_buffer("prior_offset", torch.zeros(num_actions))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints from before 2026-10-07 have no prior_offset: they ran without one (a scale-init prior lives in
        # the output bias instead), so load them with zeros.
        state_dict.setdefault(prefix + "prior_offset", torch.zeros_like(self.prior_offset))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """@brief Compute raw action logits.

        @param x  Fused feature vector of shape (B, @p input_dim).
        @return Action logits of shape (B, sum(@ref branches)).
        """
        return self.net(x) + self.prior_offset

    def get_distributions(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> List[Categorical]:
        """@brief One (masked) Categorical per branch.

        @param x            Fused feature vector of shape (B, @p input_dim).
        @param action_mask  (B, sum(branches)), 1 = enabled; None enables every action.
        """
        return branch_distributions(self.forward(x), self.branches, action_mask)

    @torch.no_grad()
    def set_prior_offset(self, prior_logp: torch.Tensor) -> None:
        """@brief Start the policy at the prior by a fixed logit offset log p0 on a default-initialized output layer
        (2026-10-07, the default). A default layer's logit spread across states is about 0.07, so the start is
        about p0 (its mean W.h still tilts each logit by a few tenths of a nat: p = 0.8 starts near 0.7), and the
        policy gradient reaches the shared trunk at full strength (audit M1: the scale init below cut it about
        190x)."""
        self.prior_offset.copy_(prior_logp.to(self.prior_offset.device))
        self.net[-1].bias.zero_()     # the default layer's random bias is a state-independent tilt away from p0

    @torch.no_grad()
    def init_prior(self, prior_logp: torch.Tensor, weight_scale: float = 0.01) -> None:
        """@brief Start the policy at the prior: output-layer weights scaled by @p weight_scale and its bias set to
        log p0, so the logits are about log p0 for any input (action-prior branch, 2026-10-06).

        @note Scaling the output weights scales the actor gradient into the trunk by the same factor (audit M1,
        docs/experiments/review_1007/TRAINING_STACK_AUDIT_1007.md); kept for reproducing the 10-06 prior runs and
        for the milder weight_scale 0.1-0.3. set_prior_offset is the default since 2026-10-07."""
        last = self.net[-1]
        last.weight.mul_(weight_scale)
        last.bias.copy_(prior_logp.to(last.bias.device))


class CriticHead(nn.Module):
    """@brief Critic network: estimates the state value V(s), with PopArt value normalization.

    @details
    Architecture mirrors ActorHead but outputs a single scalar per batch element. The last layer predicts the
    value normalized by running return statistics (mean @c ret_mean, std @c ret_std); forward() returns it in
    reward units, so rollouts, GAE and bootstraps see real values. update_stats() moves the statistics toward a
    rollout's returns and rescales the last layer so the unnormalized outputs do not change ("preserving outputs
    precisely", PopArt: van Hasselt et al. 2016). The value loss divides by ret_std**2 (value_loss_scale), which is
    regression on normalized targets. Added 2026-10-03: the tardiness training mix has episodes whose returns
    differ several-fold in scale, and the critic shares the trunk with the actor, so raw-scale value gradients
    swamped the policy's (audit section 9). With update_stats never called (ret_mean 0, ret_std 1) the head is the
    plain critic it was before.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        """@brief Construct the critic head.

        @param input_dim   Dimensionality of the fused feature vector.
        @param hidden_dim  Width of the hidden layer.
        """
        super().__init__()

        ## @brief Single-hidden-layer MLP producing a scalar (normalized) value estimate.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        ## @brief Running return statistics (float64; saved in checkpoints). ret_updates counts update_stats calls,
        ##        for the bias correction of the moving averages.
        self.register_buffer("ret_mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("ret_sq", torch.ones((), dtype=torch.float64))
        self.register_buffer("ret_updates", torch.zeros((), dtype=torch.float64))

    @property
    def ret_std(self) -> torch.Tensor:
        """@brief Return std from the second moment (floored, so a constant return cannot divide by zero)."""
        return (self.ret_sq - self.ret_mean ** 2).clamp_min(1e-8).sqrt().clamp_min(1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """@brief Compute the state-value estimate in reward units.

        @param x  Fused feature vector of shape (B, @p input_dim).
        @return Value estimate of shape (B, 1).
        """
        return self.net(x) * self.ret_std.to(x.dtype) + self.ret_mean.to(x.dtype)

    def value_loss_scale(self) -> float:
        """@brief 1 / ret_std**2: multiplies the MSE of unnormalized values so it equals the normalized one."""
        return float(1.0 / self.ret_std ** 2)

    @torch.no_grad()
    def update_stats(self, returns, beta: float) -> None:
        """@brief Move the return statistics toward @p returns (exponential average with rate @p beta, bias-corrected
        as in Adam, so the first call takes the rollout's own mean and second moment) and rescale the last layer so
        forward() gives the same outputs as before.

        @param returns  The rollout's GAE returns (any shape; numpy or tensor).
        @param beta     Rate per call, in (0, 1].
        """
        r = torch.as_tensor(returns, dtype=torch.float64).flatten()
        old_mean, old_std = self.ret_mean.clone(), self.ret_std.clone()
        n = self.ret_updates + 1
        # Bias-corrected EMA: keep the uncorrected averages implicitly by blending toward the batch with weight
        # beta / (1 - (1 - beta)^n), which is 1 on the first call.
        w = beta / (1.0 - (1.0 - beta) ** n)
        self.ret_mean.mul_(1 - w).add_(w * r.mean())
        self.ret_sq.mul_(1 - w).add_(w * (r ** 2).mean())
        self.ret_updates.copy_(n)
        new_mean, new_std = self.ret_mean, self.ret_std
        last = self.net[-1]
        last.weight.mul_((old_std / new_std).to(last.weight.dtype))
        last.bias.copy_(((old_std * last.bias.double() + old_mean - new_mean) / new_std).to(last.bias.dtype))


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
                 branches: Sequence[int] = (4, 3), critic_extra_dim: int = 0):
        """@brief Construct the combined actor-critic.

        @param input_dim    Dimensionality of the fused feature vector.
        @param hidden_dim   Width of hidden layers in both heads.
        @param branches     Size of each discrete action branch.
        @param critic_extra_dim  Critic-only extra inputs (asymmetric actor-critic, 2026-10-07): the critic reads
                            [features, critic_extra]; 0 = the critic reads the features only.
        """
        super().__init__()

        ## @brief Policy (actor) head producing concatenated branch logits.
        self.actor = ActorHead(input_dim, hidden_dim, branches)
        ## @brief Value (critic) head producing V(s).
        self.critic = CriticHead(input_dim + critic_extra_dim, hidden_dim)
        ## @brief Width of the critic-only extra input.
        self.critic_extra_dim = critic_extra_dim

    def value(self, features: torch.Tensor, critic_extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        """@brief V(s) in reward units, (B, 1); with critic_extra_dim > 0 it needs @p critic_extra (B, extra)."""
        if self.critic_extra_dim:
            if critic_extra is None:
                raise ValueError("this critic reads critic-only look-ahead features (obs 'critic_lookahead'), but the "
                                 "observation has none: train / evaluate with --critic-lookahead on the twin")
            features = torch.cat([features, critic_extra.to(features.dtype)], dim=-1)
        return self.critic(features)

    def forward(self, features: torch.Tensor, critic_extra: Optional[torch.Tensor] = None):
        """@brief Forward pass returning both action logits and value.

        @param features  Fused representation of shape (B, @p input_dim).
        @return Tuple of (action_logits, value) with shapes (B, sum(branches)) and (B, 1).
        """
        return self.actor(features), self.value(features, critic_extra)

    def act(self, features: torch.Tensor, action_mask: Optional[torch.Tensor] = None,
            deterministic: bool = False, critic_extra: Optional[torch.Tensor] = None):
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
        value = self.value(features, critic_extra).squeeze(-1)

        if deterministic:
            action = torch.stack([d.logits.argmax(dim=-1) for d in dists], dim=-1)
        else:
            action = torch.stack([d.sample() for d in dists], dim=-1)

        log_prob = sum(d.log_prob(action[:, i]) for i, d in enumerate(dists))
        return action, log_prob, value

    def evaluate(self, features: torch.Tensor, actions: torch.Tensor,
                 action_mask: Optional[torch.Tensor] = None, prior_logp: Optional[torch.Tensor] = None,
                 critic_extra: Optional[torch.Tensor] = None):
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
                - @c prior_kl  (B,) — KL(pi || p0), only when @p prior_logp is given (a 4th element).
        """
        dists = self.actor.get_distributions(features, action_mask)
        value = self.value(features, critic_extra).squeeze(-1)

        log_probs = sum(d.log_prob(actions[:, i]) for i, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
        if prior_logp is not None:
            return log_probs, value, entropy, kl_to_prior(dists, prior_logp, self.actor.branches, action_mask)
        return log_probs, value, entropy
