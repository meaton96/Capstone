"""
@file test_action_prior.py
@brief Action prior (action-prior branch, 2026-10-06): p0 construction, policy initialization at p0, KL(pi || p0)
       with and without masks, and the extended evaluate() return.
"""
import pytest
import torch

from config import JOB_HEAD_RULES, MACHINE_HEAD_RULES, ACTION_BRANCHES
from models.actor_critic import branch_distributions, kl_to_prior, prior_log_probs
from models.network import SchedulingNetwork
from tests.test_architecture import make_dummy_obs, random_actions

HEADS = (JOB_HEAD_RULES, MACHINE_HEAD_RULES)


def test_prior_log_probs_put_prob_on_the_pair():
    lp = prior_log_probs("MDD-TECT", 0.8, HEADS)
    job, machine = torch.split(lp.exp(), list(ACTION_BRANCHES))
    assert torch.isclose(job.sum(), torch.tensor(1.0)) and torch.isclose(machine.sum(), torch.tensor(1.0))
    assert job[JOB_HEAD_RULES.index("MDD")] == pytest.approx(0.8)
    assert machine[MACHINE_HEAD_RULES.index("TECT")] == pytest.approx(0.8)
    others = [p for i, p in enumerate(job.tolist()) if i != JOB_HEAD_RULES.index("MDD")]
    assert others == pytest.approx([0.2 / (len(JOB_HEAD_RULES) - 1)] * len(others))


@pytest.mark.parametrize("pair,prob", [("MDD", 0.8), ("XYZ-TECT", 0.8), ("MDD-TECT", 1.0), ("MDD-TECT", 0.0)])
def test_prior_log_probs_rejects_bad_input(pair, prob):
    with pytest.raises(ValueError):
        prior_log_probs(pair, prob, HEADS)


def test_init_prior_starts_the_policy_at_p0():
    torch.manual_seed(0)
    net = SchedulingNetwork()
    lp = prior_log_probs("MDD-TECT", 0.8, HEADS)
    net.actor_critic.actor.init_prior(lp)
    net.eval()
    with torch.no_grad():
        dists = net.distributions(make_dummy_obs(8))
    p0 = torch.split(lp.exp(), list(ACTION_BRANCHES))
    for d, p in zip(dists, p0):
        assert torch.allclose(d.probs, p.expand_as(d.probs), atol=0.03)


def test_kl_to_prior_zero_at_prior_positive_elsewhere():
    lp = prior_log_probs("MDD-TECT", 0.8, HEADS)
    at_prior = branch_distributions(lp.expand(3, -1).clone(), ACTION_BRANCHES)
    assert torch.allclose(kl_to_prior(at_prior, lp, ACTION_BRANCHES), torch.zeros(3), atol=1e-6)
    away = branch_distributions(torch.zeros(3, sum(ACTION_BRANCHES)), ACTION_BRANCHES)   # uniform
    assert (kl_to_prior(away, lp, ACTION_BRANCHES) > 0.1).all()


def test_kl_to_prior_ignores_a_head_masked_to_one_action():
    lp = prior_log_probs("MDD-TECT", 0.8, HEADS)
    mask = torch.ones(2, sum(ACTION_BRANCHES))
    mask[:, ACTION_BRANCHES[0] + 1:] = 0.0      # machine head masked down to its action 0 (ECT)
    logits = torch.randn(2, sum(ACTION_BRANCHES))
    dists = branch_distributions(logits, ACTION_BRANCHES, mask)
    kl_both = kl_to_prior(dists, lp, ACTION_BRANCHES, mask)
    kl_job = kl_to_prior(dists[:1], lp[:ACTION_BRANCHES[0]], ACTION_BRANCHES[:1])
    assert torch.isfinite(kl_both).all()
    assert torch.allclose(kl_both, kl_job, atol=1e-5)


def test_evaluate_returns_prior_kl_only_when_asked():
    net = SchedulingNetwork()
    obs, actions = make_dummy_obs(), random_actions()
    assert len(net.evaluate(obs, actions)) == 3
    out = net.evaluate(obs, actions, prior_log_probs("MDD-TECT", 0.8, HEADS))
    assert len(out) == 4 and out[3].shape == (actions.shape[0],) and (out[3] >= -1e-6).all()
