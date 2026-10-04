"""@file test_value_norm.py
@brief PopArt value normalization in CriticHead (2026-10-03): stats updates preserve outputs, the first update takes
the rollout's own statistics, and the scaled loss equals the MSE on normalized targets."""
import numpy as np
import torch

from models.actor_critic import CriticHead


def test_update_preserves_outputs_and_tracks_returns():
    torch.manual_seed(0)
    critic = CriticHead(16, 32)
    x = torch.randn(50, 16)
    before = critic(x).detach()
    returns = np.random.default_rng(0).normal(-40.0, 12.0, size=2048)
    critic.update_stats(returns, beta=0.05)
    assert torch.allclose(critic(x).detach(), before, atol=1e-4)
    assert abs(float(critic.ret_mean) - returns.mean()) < 1e-6          # first call: the rollout's own stats
    assert abs(float(critic.ret_std) - returns.std()) < 1e-4
    critic.update_stats(returns * 3.0, beta=0.05)
    assert torch.allclose(critic(x).detach(), before, atol=1e-3)


def test_scaled_loss_equals_normalized_target_loss():
    torch.manual_seed(1)
    critic = CriticHead(8, 16)
    critic.update_stats(np.random.default_rng(1).normal(-20.0, 5.0, size=512), beta=1.0)
    x, ret = torch.randn(64, 8), torch.randn(64) * 5.0 - 20.0
    pred = critic(x).squeeze(-1)
    mu, sd = critic.ret_mean.float(), critic.ret_std.float()
    normalized = torch.nn.functional.mse_loss((pred - mu) / sd, (ret - mu) / sd)
    scaled = critic.value_loss_scale() * torch.nn.functional.mse_loss(pred, ret)
    assert torch.allclose(normalized, scaled, rtol=1e-4)


def test_without_updates_it_is_the_plain_critic():
    critic = CriticHead(8, 16)
    x = torch.randn(4, 8)
    assert torch.equal(critic(x), critic.net(x))
    assert critic.value_loss_scale() == 1.0
