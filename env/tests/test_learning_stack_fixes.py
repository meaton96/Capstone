"""
@file test_learning_stack_fixes.py
@brief Fixes from the 10-07 training-stack audit and credit trace (docs/handoffs/HANDOFF_2026-10-07_learning_stack_fixes.md):
       the prior as a fixed logit offset (M1), λ over simulated time (M2), the shared-instance baseline, checkpoint run
       settings in evaluate.py (m2), the discounted-return column (M3) and the three-reference comparison table.
"""
import argparse
import json

import numpy as np
import pytest
import torch

from config import ACTION_BRANCHES, JOB_HEAD_RULES, MACHINE_HEAD_RULES
from models.actor_critic import prior_log_probs
from models.network import SchedulingNetwork
from rollout_buffer import RolloutBuffer
from tests.test_architecture import make_dummy_obs
from tests.test_evaluate import FakeEnv

HEADS = (JOB_HEAD_RULES, MACHINE_HEAD_RULES)


# ── M1: prior as a fixed logit offset ──────────────────────────────────────────────────────────────────

def test_prior_offset_starts_near_p0_and_is_saved():
    torch.manual_seed(0)
    net = SchedulingNetwork()
    lp = prior_log_probs("MDD-TECT", 0.8, HEADS)
    net.actor_critic.actor.set_prior_offset(lp)
    net.eval()
    with torch.no_grad():
        dists = net.distributions(make_dummy_obs(8))
    # About p0: the default layer's mean W.h tilts each logit by a few tenths of a nat (p(MDD) ~0.7 instead of 0.8),
    # while the prior's rule stays the clear argmax of every head.
    for d, p in zip(dists, torch.split(lp.exp(), list(ACTION_BRANCHES))):
        assert torch.allclose(d.probs, p.expand_as(d.probs), atol=0.15)
        assert (d.probs.argmax(-1) == p.argmax()).all()
    other = SchedulingNetwork()
    other.load_state_dict(net.state_dict())
    assert torch.equal(other.actor_critic.actor.prior_offset, lp)


def test_a_checkpoint_without_the_offset_loads_with_zeros():
    net = SchedulingNetwork()
    state = {k: v for k, v in net.state_dict().items() if not k.endswith("prior_offset")}
    other = SchedulingNetwork()
    other.actor_critic.actor.prior_offset.fill_(1.0)
    other.load_state_dict(state)          # strict: the missing key is filled, not an error
    assert torch.equal(other.actor_critic.actor.prior_offset, torch.zeros(sum(ACTION_BRANCHES)))


def _trunk_pg_norm(net, obs):
    net.eval()
    with torch.no_grad():
        actions, old_lp, _ = net.act(obs)
    net.train()
    lp, _, _ = net.evaluate(obs, actions)
    adv = torch.randn(len(old_lp), generator=torch.Generator().manual_seed(1))
    loss = -(torch.exp(lp - old_lp) * adv).mean()
    grads = torch.autograd.grad(loss, list(net.fusion.parameters()))
    return float(torch.sqrt(sum((g ** 2).sum() for g in grads)))


def test_offset_prior_keeps_the_policy_gradient_into_the_trunk():
    obs = make_dummy_obs(32)
    lp = prior_log_probs("MDD-TECT", 0.8, HEADS)
    norms = {}
    for mode in ("none", "offset", "scale"):
        torch.manual_seed(0)
        net = SchedulingNetwork()
        if mode == "offset":
            net.actor_critic.actor.set_prior_offset(lp)
        elif mode == "scale":
            net.actor_critic.actor.init_prior(lp, 0.01)
        norms[mode] = _trunk_pg_norm(net, obs)
    assert norms["offset"] > 0.3 * norms["none"]
    assert norms["scale"] < 0.05 * norms["none"]


# ── M2: λ over simulated time ──────────────────────────────────────────────────────────────────────────

def test_lambda_over_time_matches_per_step_lambda_at_the_reference_duration():
    def gae(dts, **kw):
        buf = RolloutBuffer(len(dts), 1, {"global_scalars": (1,)}, gamma_per_second=1 - 1 / 3000,
                            action_shape=(2,), **kw)
        for dt in dts:
            buf.add({"global_scalars": np.zeros((1, 1))}, np.zeros((1, 2)), np.zeros(1), np.ones(1),
                    np.zeros(1), np.zeros(1), dts=np.array([dt]))
        buf.compute_gae(np.zeros(1))
        return buf.advantages[:, 0].copy()

    per_step = gae([900.0] * 4, gae_lambda=0.95)
    per_time = gae([900.0] * 4, gae_lambda=0.95, gae_lambda_time_s=900.0)
    assert np.allclose(per_step, per_time)
    # Short steps: λ per step would decay 0.95 per 10 s step; per time it barely decays over 40 s.
    short_step = gae([10.0] * 4, gae_lambda=0.95)
    short_time = gae([10.0] * 4, gae_lambda=0.95, gae_lambda_time_s=900.0)
    assert short_time[0] > short_step[0]


def test_lambda_over_time_needs_time_discounting():
    with pytest.raises(ValueError):
        RolloutBuffer(2, 1, {"global_scalars": (1,)}, gae_lambda_time_s=900.0)


# ── Shared-instance baseline ───────────────────────────────────────────────────────────────────────────

def _buffer_with(advantages, returns):
    T, N = advantages.shape
    buf = RolloutBuffer(T, N, {"global_scalars": (1,)})
    buf.advantages = advantages.astype(np.float32)
    buf.returns = returns.astype(np.float32)
    return buf


def test_shared_baseline_subtracts_the_leave_one_out_mean_within_a_group():
    adv = np.array([[1.0, 3.0, 10.0, 20.0]])
    ret = np.array([[5.0, 7.0, 0.0, 0.0]])
    keys = np.array([[0, 0, 0, 0]])
    groups = np.array([0, 0, 1, 1])
    buf = _buffer_with(adv, ret)
    stats = buf.apply_shared_baseline(keys, groups, "residual")
    assert np.allclose(buf.advantages, [[-2.0, 2.0, -10.0, 10.0]])
    assert stats["shared_baseline_share"] == 1.0
    assert np.allclose(buf.returns, ret)          # the critic's target is unchanged
    buf = _buffer_with(adv, ret)
    buf.apply_shared_baseline(keys, groups, "return")
    assert np.allclose(buf.advantages[0, :2], [-2.0, 2.0])


def test_shared_baseline_matches_slots_by_key_and_leaves_singletons_alone():
    # env 1 is one slot behind env 0 (different rollout positions, same instance slot keys).
    adv = np.array([[1.0, 9.0], [2.0, 4.0], [3.0, 5.0]])
    keys = np.array([[0, -1], [1, 0], [2, 1]])
    buf = _buffer_with(adv, adv)
    buf.apply_shared_baseline(keys, np.array([0, 0]), "residual")
    # key 0: env0 t0 (1) vs env1 t1 (4); key 1: env0 t1 (2) vs env1 t2 (5); key 2 only env0; env1 t0 no key.
    assert np.allclose(buf.advantages, [[-3.0, 9.0], [-3.0, 3.0], [3.0, 3.0]])


def test_instance_groups_share_seed_streams(tmp_path):
    import os
    from tests.test_twin_env import FIX, with_frame
    from env_wrappers.twin_env import VectorizedTwinEnv
    floor = tmp_path / "des_floor.json"
    floor.write_text(json.dumps(with_frame(json.load(open(os.path.join(FIX, "convoy_agv3", "des_floor.json"))))))
    floor = str(floor)
    scenario = json.load(open(os.path.join(FIX, "convoy_waves.json")))
    vec = VectorizedTwinEnv(4, floor, train_seed=3, scenario=scenario, obs_caps=(15, 64), instance_group=2)
    draws = [[int(e.seed_rng.integers(0, 1 << 24)) for _ in range(3)] for e in vec.envs]
    assert draws[0] == draws[1] and draws[2] == draws[3] and draws[0] != draws[2]
    plain = VectorizedTwinEnv(2, floor, train_seed=3, scenario=scenario, obs_caps=(15, 64))
    assert [int(e.seed_rng.integers(0, 1 << 24)) for e in plain.envs][1] != draws[0][0]


# ── evaluate.py: settings, discounted return, comparisons ─────────────────────────────────────────────

class _Ckpt:
    kind = "checkpoint"

    def __init__(self, name, settings, ppo):
        self.name, self.settings, self.ppo = name, settings, ppo


def _eval_args(**kw):
    base = dict(slot_seconds=None, discount_horizon_s=None, twin_transport="kinematic", scenario_generator="randomized",
                params=None, episode_duration_seconds=21600.0, random_warmup=True, machine_flexibility=0.0)
    base.update(kw)
    return argparse.Namespace(**base)


def test_evaluate_takes_slot_and_discount_from_the_checkpoint(capsys):
    from evaluate import resolve_checkpoint_settings
    ck = _Ckpt("ckpt:a", {"slot_seconds": 900.0, "twin_transport": "kinematic", "scenario_generator": "randomized",
                          "episode_duration_seconds": 21600.0, "random_warmup": True}, {"discount_horizon_s": 10800.0})
    slot, gamma = resolve_checkpoint_settings(_eval_args(), [ck])
    assert slot == 900.0 and gamma == pytest.approx(1 - 1 / 10800)
    slot, gamma = resolve_checkpoint_settings(_eval_args(slot_seconds=0.0, discount_horizon_s=0.0,
                                                         twin_transport="instant"), [ck])
    out = capsys.readouterr().out
    assert slot == 0.0 and gamma is None
    assert "trained with slot_seconds=900.0" in out and "twin_transport" in out


def test_evaluate_refuses_checkpoints_with_different_slot_settings():
    from evaluate import resolve_checkpoint_settings
    a = _Ckpt("a", {"slot_seconds": 900.0}, {})
    b = _Ckpt("b", {"slot_seconds": 3600.0}, {})
    with pytest.raises(SystemExit):
        resolve_checkpoint_settings(_eval_args(), [a, b])


def test_old_checkpoints_evaluate_per_decision_with_a_note(capsys):
    from evaluate import resolve_checkpoint_settings
    slot, gamma = resolve_checkpoint_settings(_eval_args(), [_Ckpt("old", {}, {})])
    assert slot == 0.0 and gamma is None and "record no slot setting" in capsys.readouterr().out


class _RewardEnv(FakeEnv):
    """FakeEnv with a reward of -1 and dt = 100 s per step."""

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, -1.0, done, dict(info, dt=100.0)


def test_run_evaluation_reports_the_discounted_return():
    from evaluate import ConstantPolicy, run_evaluation
    g = 1 - 1 / 1000
    rows = run_evaluation(_RewardEnv(length=3), [ConstantPolicy((0, 0), "A")], [(0, 5), (0, 6)], log=lambda _: None,
                          gamma_per_second=g)
    assert [r["seed"] for r in rows] == [5, 6]
    for r in rows:
        assert r["discounted_return"] == pytest.approx(-(1 + g ** 100 + g ** 200))
    rows = run_evaluation(_RewardEnv(length=3), [ConstantPolicy((0, 0), "A")], [(0, 5)], log=lambda _: None)
    assert rows[0]["discounted_return"] is None


def _row(policy, kind, seed, ret):
    return {"policy": policy, "kind": kind, "seed": seed, "makespan": 1.0, "return": ret, "jobs_exited": 1,
            "total_flow_time": 1.0, "mean_flow_time": 1.0, "deadlock": False, "timed_out": False}


def test_summarize_reports_best_fixed_hindsight_and_oracle():
    from evaluate import summarize
    rows = [_row("A", "pdr", 1, -100), _row("B", "pdr", 1, -90), _row("C", "checkpoint", 1, -95),
            _row("A", "pdr", 2, -100), _row("B", "pdr", 2, -130), _row("C", "checkpoint", 2, -98)]
    oracle = {1: -80.0, 2: -90.0}
    s = {e["policy"]: e for e in summarize(rows, oracle=oracle)}
    # Best fixed overall: A (mean cost 100) beats B (110).
    assert s["C"]["best_fixed"] == "A"
    assert s["C"]["vs_best_fixed_pct"] == pytest.approx(100 * (96.5 / 100 - 1))
    assert (s["C"]["wins_vs_best_fixed"], s["C"]["losses_vs_best_fixed"]) == (2, 0)
    # Hindsight: per seed best is B (90) on seed 1 and A (100) on seed 2.
    assert s["C"]["vs_hindsight_fixed_pct"] == pytest.approx(100 * (96.5 / 95 - 1))
    assert s["C"]["vs_oracle_pct"] == pytest.approx(100 * (96.5 / 85 - 1))
    # Oracle gain over A: 15 per seed on average; C gains 3.5.
    assert s["C"]["oracle_gain_captured_pct"] == pytest.approx(100 * 3.5 / 15)
    named = {e["policy"]: e for e in summarize(rows, oracle=oracle, best_fixed="B")}
    assert named["C"]["best_fixed"] == "B"
    with pytest.raises(ValueError):
        summarize(rows, best_fixed="ZZZ")


def test_load_oracle_reads_both_result_formats_and_checks_the_fixed_pairs(tmp_path, capsys):
    from evaluate import check_oracle, load_oracle
    (tmp_path / "a.json").write_text(json.dumps({"seed": 3, "oracle_return": -12.5, "fixed_returns": {"A": -14.0}}))
    (tmp_path / "B2_s4.json").write_text(json.dumps({"seed": 4, "oracle": 20.0, "fixed": {"A": {"tard": 22.0}}}))
    oracle, fixed = load_oracle([str(tmp_path / "*.json")])
    assert oracle == {3: -12.5, 4: -20.0} and fixed == {3: {"A": -14.0}, 4: {"A": -22.0}}
    check_oracle([_row("A", "pdr", 3, -14.0), _row("A", "pdr", 4, -22.0)], fixed)
    assert "match" in capsys.readouterr().out
    check_oracle([_row("A", "pdr", 3, -14.0), _row("A", "pdr", 4, -25.0)], fixed)
    assert "not comparable" in capsys.readouterr().out


# ── Paired same-future advantage (fix 5) ───────────────────────────────────────────────────────────────

def _fixture_floor(tmp_path):
    import os
    from tests.test_twin_env import FIX, with_frame
    floor = tmp_path / "des_floor.json"
    floor.write_text(json.dumps(with_frame(json.load(open(os.path.join(FIX, "convoy_agv3", "des_floor.json"))))))
    return str(floor)


def _play(env, plan, n):
    """Play n slot steps with plan(k) -> action; return the slot start times, rewards and infos."""
    env.reset()
    times, rewards, infos = [], [], []
    for k in range(n):
        times.append(env._now())
        _, r, done, info = env.step(plan(k))
        rewards.append(r)
        infos.append(info)
        if done:
            break
    return times, rewards, infos


def test_paired_advantage_on_the_own_future_matches_the_real_runs(tmp_path):
    import os
    from env_wrappers.paired_slot_env import PairedSlotEnv
    from env_wrappers.twin_env import TwinSchedulingEnv
    from rewards import load_reward
    from tests.test_twin_env import FIX, REWARDS
    floor = _fixture_floor(tmp_path)
    scenario = dict(json.load(open(os.path.join(FIX, "convoy_waves.json"))), agvCount=3)
    reward = os.path.join(REWARDS, "flow_time.json")
    g, slot, tail = 1 - 1 / 3000, 300.0, 300.0
    default, alt = (0, 0), (2, 1)

    def make():
        inner = TwinSchedulingEnv(floor, reward_fn=load_reward(reward).build(), scenario=scenario, obs_caps=(15, 64))
        return PairedSlotEnv(inner, slot, g, default, futures=0, tail_seconds=tail)

    # Real runs: alt in slot 1 then default; and default throughout. Slot rewards are discounted to the slot start.
    t_alt, r_alt, info_alt = _play(make(), lambda k: alt if k == 1 else default, 4)
    t_def, r_def, _ = _play(make(), lambda k: default, 4)
    assert t_alt[:2] == t_def[:2]
    t0 = t_alt[1]
    # Slots 1-2 (slot + a 300 s tail = one more slot here), discounted to t0.
    real_alt = r_alt[1] + g ** (t_alt[2] - t0) * r_alt[2]
    real_def = r_def[1] + g ** (t_def[2] - t0) * r_def[2]
    adv = info_alt[1]["paired_advantage"]
    assert info_alt[0]["paired_advantage"] == 0.0           # the default pair: nothing to simulate
    assert adv == pytest.approx(real_alt - real_def, rel=1e-4, abs=1e-6)


def test_paired_advantage_on_redrawn_futures_reaches_the_same_state(tmp_path):
    import os
    from env_wrappers.twin_env import VectorizedTwinEnv
    from scenarios.randomized import RandomizedParams, randomized_generator
    from tests.test_twin_env import REWARDS
    floor = _fixture_floor(tmp_path)
    gen = randomized_generator(1800.0, random_warmup=True, params=RandomizedParams(agv_count=3, failure_probability=0.0))
    vec = VectorizedTwinEnv(1, floor, reward_spec=os.path.join(REWARDS, "flow_time.json"), train_seed=0,
                            scenario_generator=gen, obs_caps=(15, 256), slot_seconds=300.0,
                            gamma_per_second=1 - 1 / 3000,
                            paired={"default_action": (0, 0), "futures": 2, "tail_seconds": 300.0})
    vec.reset()
    advs = []
    for k in range(4):
        _, _, dones, _, infos = vec.step(np.array([[2, 1] if k % 2 else [0, 0]]))
        advs.append(infos[0]["paired_advantage"])
        if dones[0]:
            break
    env = vec.envs[0]
    assert env.mismatches == 0 and env.branch_runs > 0
    assert advs[0] == 0.0 and all(a is not None and np.isfinite(a) for a in advs)


# ── Critic-only look-ahead (fix 6) ─────────────────────────────────────────────────────────────────────

def test_fleet_fraction_follows_the_agv_schedule():
    from des_twin.lookahead import _fleet_fraction
    assert _fleet_fraction((), 7, 0.0, 100.0) == 1.0
    sched = ((0.0, 2), (5400.0, 7))
    assert _fleet_fraction(sched, 7, 3600.0, 7200.0) == pytest.approx((1800 * 2 + 1800 * 7) / (3600 * 7))
    assert _fleet_fraction(((1000.0, 3),), 6, 0.0, 2000.0) == pytest.approx((1000 * 6 + 1000 * 3) / (2000 * 6))


def test_critic_lookahead_reaches_the_critic_only(tmp_path):
    import os
    from des_twin.lookahead import LOOKAHEAD_DIM
    from env_wrappers.twin_env import TwinSchedulingEnv
    from config import ActorCriticConfig
    from models.network import ac_config_for
    from tests.test_twin_env import FIX
    floor = _fixture_floor(tmp_path)
    scenario = dict(json.load(open(os.path.join(FIX, "convoy_waves.json"))), agvCount=3)
    env = TwinSchedulingEnv(floor, scenario=scenario, obs_caps=(15, 64), critic_lookahead=True)
    obs = env.reset()
    assert obs["critic_lookahead"].shape == (LOOKAHEAD_DIM,) and np.isfinite(obs["critic_lookahead"]).all()
    net = SchedulingNetwork(ac_cfg=ActorCriticConfig(critic_extra_dim=LOOKAHEAD_DIM))
    batch = make_dummy_obs(4)
    batch["critic_lookahead"] = torch.rand(4, LOOKAHEAD_DIM)
    _, _, v1 = net.act(batch, deterministic=True)
    batch2 = dict(batch, critic_lookahead=torch.rand(4, LOOKAHEAD_DIM))
    _, _, v2 = net.act(batch2, deterministic=True)
    assert not torch.allclose(v1, v2)                                  # the critic reads it
    d1 = [d.probs for d in net.distributions(batch)]
    d2 = [d.probs for d in net.distributions(batch2)]
    assert all(torch.equal(a, b) for a, b in zip(d1, d2))               # the actor does not
    with pytest.raises(ValueError):
        net.act(make_dummy_obs(2))
    assert ac_config_for(net.state_dict()).critic_extra_dim == LOOKAHEAD_DIM
    assert ac_config_for(SchedulingNetwork().state_dict()).critic_extra_dim == 0
