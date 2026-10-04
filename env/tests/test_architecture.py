"""
@file test_architecture.py
@brief Test suite for the DRL Scheduling Architecture.

@details
Tests cover:
  1. Tensor shape correctness through every layer
  2. Forward/backward pass (gradient flow)
  3. Rollout buffer GAE computation
  4. PPO loss computation
  5. Checkpoint save/load round-trip
  6. Deterministic vs stochastic action selection
  7. Observation slicing (slice_obs) from unity_env
  8. UnitySchedulingEnv decision-loop logic (mocked)
  9. VectorizedUnityEnv batching and auto-reset (mocked)

@par Running
@code{.sh}
python -m pytest tests/test_architecture.py -v
@endcode
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from config import (
    EncoderConfig, FusionConfig, ActorCriticConfig,
    GRID_SIZE, GRID_CHANNELS, MAX_JOBS, JOB_FEATURES, MAX_MACHINES, MACHINE_FEATURES,
    GLOBAL_SCALARS, EVENT_FLAGS, TOTAL_OBS_SIZE, OBS_SHAPES, OBS_LAYOUT,
    SLICE_SPATIAL_END, SLICE_MACHINES_END, SLICE_JOBS_END, SLICE_SCALARS_END, SLICE_FLAGS_END,
    MACHINE_CANDIDATE_COL, ACTION_BRANCHES, ACTION_MASK_LEN, ACTION_LAYOUT,
)
from models.encoder import CNNSPPFEncoder, SPPF, MLPEncoder, MultiModalEncoder, SetEncoder
from models.actor_critic import FusionHead, ActorHead, CriticHead, ActorCritic
from models.network import SchedulingNetwork
from env_wrappers.unity_env import slice_obs
from config import obs_shapes as obs_shapes_for, PPOConfig
from rollout_buffer import RolloutBuffer


## @brief Default batch size used across all tests.
BATCH = 4

## @brief Device used for tensor allocation in tests.
DEVICE = "cpu"


## @brief The streams slice_obs cuts from the flat vector (every OBS_SHAPES key except action_mask).
SLICED_SHAPES = {k: v for k, v in OBS_SHAPES.items() if k != "action_mask"}


def random_actions(batch_size: int = BATCH) -> torch.Tensor:
    """@brief A batch of random (job head, machine head) actions."""
    return torch.stack([torch.randint(0, size, (batch_size,)) for size in ACTION_BRANCHES], dim=-1)


def make_dummy_obs(batch_size: int = BATCH) -> dict:
    """@brief Create a batch of dummy observations matching the state space.

    @param batch_size  Number of samples in the batch.
    @return Dict of random tensors keyed by observation-space names.
    """
    obs = {k: torch.rand(batch_size, *shape) for k, shape in OBS_SHAPES.items()}
    # A 15-machine floor with 10 active jobs: rows past those are padding (present = 0).
    obs["machine_table"][:, 15:] = 0.0
    obs["machine_table"][:, :15, 0] = 1.0
    obs["job_table"][:, 10:] = 0.0
    obs["job_table"][:, :10, 0] = 1.0
    obs["action_mask"] = torch.ones(batch_size, ACTION_MASK_LEN)
    return obs


# ============================================================
#  1. Shape tests for individual components
# ============================================================

class TestSPPF:
    """@brief Unit tests for the @ref SPPF module."""

    def test_output_shape(self):
        """@brief Verify SPPF produces the expected (B, C, H, W) output."""
        sppf = SPPF(128, 128)
        x = torch.randn(BATCH, 128, 16, 16)
        out = sppf(x)
        assert out.shape == (BATCH, 128, 16, 16), f"SPPF shape: {out.shape}"

    def test_preserves_spatial(self):
        """@brief Spatial dimensions must be preserved through SPPF."""
        sppf = SPPF(64, 64)
        x = torch.randn(2, 64, 8, 8)
        out = sppf(x)
        assert out.shape[2:] == x.shape[2:], "SPPF should preserve spatial dims"


class TestCNNSPPFEncoder:
    """@brief Shape tests for @ref CNNSPPFEncoder with factory and schedule inputs."""

    def test_factory_encoder_shape(self):
        """@brief Factory-floor grid (3, 64, 64) → 256-D embedding."""
        enc = CNNSPPFEncoder(in_channels=3, out_dim=256)
        x = torch.randn(BATCH, 3, 64, 64)
        out = enc(x)
        assert out.shape == (BATCH, 256), f"Factory encoder: {out.shape}"



class TestSetEncoder:
    """@brief @ref SetEncoder: shape, and invariance to row order and padding."""

    def _table(self, rows=15, total=MAX_MACHINES):
        t = torch.zeros(BATCH, total, MACHINE_FEATURES)
        t[:, :rows] = torch.rand(BATCH, rows, MACHINE_FEATURES)
        t[:, :rows, 0] = 1.0
        return t

    def _enc(self):
        torch.manual_seed(0)
        return SetEncoder(MACHINE_FEATURES, 128, 64, present_col=0,
                          candidate_col=MACHINE_CANDIDATE_COL).eval()

    def test_shape(self):
        assert self._enc()(self._table()).shape == (BATCH, 128)

    def test_row_permutation_invariant(self):
        """@brief Reordering the present rows must not change the embedding."""
        enc, t = self._enc(), self._table()
        perm = torch.randperm(15)
        t2 = t.clone()
        t2[:, :15] = t[:, perm]
        torch.testing.assert_close(enc(t), enc(t2))

    def test_padding_invariant(self):
        """@brief Garbage in padding rows (present = 0) must be ignored, so a 15-machine floor
        encodes the same in a 100-row or a 30-row table."""
        enc, t = self._enc(), self._table()
        noisy = t.clone()
        noisy[:, 15:, 1:] = torch.rand(BATCH, MAX_MACHINES - 15, MACHINE_FEATURES - 1)
        torch.testing.assert_close(enc(t), enc(noisy))
        torch.testing.assert_close(enc(t), enc(t[:, :30]))

    def test_candidate_flag_changes_embedding(self):
        """@brief Marking different candidate rows must change the output (the candidate pool is live)."""
        enc, t = self._enc(), self._table()
        a, b = t.clone(), t.clone()
        a[:, :15, MACHINE_CANDIDATE_COL] = 0.0
        b[:, :15, MACHINE_CANDIDATE_COL] = 0.0
        a[:, 0, MACHINE_CANDIDATE_COL] = 1.0
        b[:, 1, MACHINE_CANDIDATE_COL] = 1.0
        assert not torch.allclose(enc(a), enc(b))

    def test_empty_table_is_finite(self):
        """@brief No present rows (e.g. the zero-padded obs between episodes) must not produce NaN/inf."""
        out = self._enc()(torch.zeros(BATCH, MAX_MACHINES, MACHINE_FEATURES))
        assert torch.isfinite(out).all()


class TestMLPEncoder:
    """@brief Shape tests for @ref MLPEncoder across the three vector modalities."""

    def test_global_scalars(self):
        """@brief 16-D global scalars → 32-D embedding."""
        mlp = MLPEncoder(GLOBAL_SCALARS, 32)
        out = mlp(torch.randn(BATCH, GLOBAL_SCALARS))
        assert out.shape == (BATCH, 32)

    def test_event_flags(self):
        """@brief 6-D event flags → 16-D embedding."""
        mlp = MLPEncoder(6, 16)
        out = mlp(torch.randn(BATCH, 6))
        assert out.shape == (BATCH, 16)


class TestMultiModalEncoder:
    """@brief Integration tests for @ref MultiModalEncoder."""

    def test_output_dim(self):
        """@brief Concatenated output must be (B, 560)."""
        enc = MultiModalEncoder()
        obs = make_dummy_obs()
        out = enc(obs)
        assert out.shape == (BATCH, 560), f"Encoder concat: {out.shape}"

    def test_output_dim_matches_config(self):
        """@brief @ref MultiModalEncoder.output_dim must agree with EncoderConfig.concat_dim."""
        cfg = EncoderConfig()
        enc = MultiModalEncoder(cfg)
        assert enc.output_dim == cfg.concat_dim == 560


class TestFusionHead:
    """@brief Tests for @ref FusionHead."""

    def test_shape(self):
        """@brief Fusion output must be (B, 256)."""
        fusion = FusionHead(560, 512, 256)
        x = torch.randn(BATCH, 560)
        out = fusion(x)
        assert out.shape == (BATCH, 256)

    def test_deterministic_eval(self):
        """@brief Eval-mode forward passes on the same input must match."""
        fusion = FusionHead(560, 512, 256)
        fusion.eval()
        x = torch.randn(1, 560)
        with torch.no_grad():
            out1 = fusion(x)
            out2 = fusion(x)
        assert torch.allclose(out1, out2)


class TestActorCritic:
    """@brief Shape and contract tests for @ref ActorHead, @ref CriticHead,
    and @ref ActorCritic."""

    def test_actor_shape(self):
        """@brief Actor logits must be (B, 8)."""
        actor = ActorHead(256, 256, ACTION_BRANCHES)
        out = actor(torch.randn(BATCH, 256))
        assert out.shape == (BATCH, ACTION_MASK_LEN)

    def test_critic_shape(self):
        """@brief Critic value must be (B, 1)."""
        critic = CriticHead(256, 256)
        out = critic(torch.randn(BATCH, 256))
        assert out.shape == (BATCH, 1)

    def test_act_outputs(self):
        """@brief @ref ActorCritic.act must return (action, log_prob, value)
        with one in-range index per branch."""
        ac = ActorCritic(256, 256, ACTION_BRANCHES)
        features = torch.randn(BATCH, 256)
        action, log_prob, value = ac.act(features)
        assert action.shape == (BATCH, len(ACTION_BRANCHES))
        assert log_prob.shape == (BATCH,)
        assert value.shape == (BATCH,)
        for i, size in enumerate(ACTION_BRANCHES):
            assert (action[:, i] >= 0).all() and (action[:, i] < size).all()

    def test_evaluate_outputs(self):
        """@brief @ref ActorCritic.evaluate must return (log_probs, values, entropy)
        with non-negative entropy."""
        ac = ActorCritic(256, 256, ACTION_BRANCHES)
        features = torch.randn(BATCH, 256)
        actions = random_actions(BATCH)
        lp, val, ent = ac.evaluate(features, actions)
        assert lp.shape == (BATCH,)
        assert val.shape == (BATCH,)
        assert ent.shape == (BATCH,)
        assert (ent >= 0).all(), "Entropy should be non-negative"

    def test_masked_head_is_forced_and_excluded(self):
        """@brief A head masked down to action 0 always picks 0 and adds nothing to log-prob or entropy,
        so the decision's log-prob equals the unmasked head's alone."""
        ac = ActorCritic(256, 256, ACTION_BRANCHES)
        features = torch.randn(BATCH, 256)
        mask = torch.ones(BATCH, ACTION_MASK_LEN)
        mask[:, ACTION_BRANCHES[0] + 1:] = 0.0          # machine head: only action 0 enabled
        action, log_prob, _ = ac.act(features, mask)
        assert (action[:, 1] == 0).all()

        lp, _, ent = ac.evaluate(features, action, mask)
        job_dist = ac.actor.get_distributions(features)[0]
        torch.testing.assert_close(lp, job_dist.log_prob(action[:, 0]))
        torch.testing.assert_close(ent, job_dist.entropy())
        torch.testing.assert_close(lp, log_prob)

    def test_fully_masked_decision_has_zero_log_prob(self):
        """@brief A decision no head can change contributes no policy gradient (log-prob 0, entropy 0)."""
        ac = ActorCritic(256, 256, ACTION_BRANCHES)
        features = torch.randn(BATCH, 256, requires_grad=True)
        mask = torch.zeros(BATCH, ACTION_MASK_LEN)
        mask[:, 0] = mask[:, ACTION_BRANCHES[0]] = 1.0
        lp, _, ent = ac.evaluate(features, torch.zeros(BATCH, 2, dtype=torch.long), mask)
        torch.testing.assert_close(lp, torch.zeros(BATCH))
        torch.testing.assert_close(ent, torch.zeros(BATCH))


# ============================================================
#  2. Full network tests
# ============================================================

class TestSchedulingNetwork:
    """@brief End-to-end tests for @ref SchedulingNetwork."""

    def test_forward_shapes(self):
        """@brief Forward pass must produce concatenated branch logits and value (B, 1)."""
        net = SchedulingNetwork()
        obs = make_dummy_obs()
        logits, value = net(obs)
        assert logits.shape == (BATCH, ACTION_MASK_LEN), f"Logits: {logits.shape}"
        assert value.shape == (BATCH, 1), f"Value: {value.shape}"

    def test_act(self):
        """@brief @ref SchedulingNetwork.act must return one action per branch, respecting obs["action_mask"]."""
        net = SchedulingNetwork()
        obs = make_dummy_obs()
        obs["action_mask"][:, 1:ACTION_BRANCHES[0]] = 0.0      # job head masked down to action 0
        action, lp, val = net.act(obs)
        assert action.shape == (BATCH, len(ACTION_BRANCHES))
        assert (action[:, 0] == 0).all()

    def test_evaluate(self):
        """@brief @ref SchedulingNetwork.evaluate must return log_probs of shape (B,)."""
        net = SchedulingNetwork()
        obs = make_dummy_obs()
        actions = random_actions()
        lp, val, ent = net.evaluate(obs, actions)
        assert lp.shape == (BATCH,)

    def test_gradient_flow(self):
        """@brief Verify gradients flow through every learnable parameter.

        @details
        Constructs a composite loss from log-probs, values, and entropy,
        calls @c backward(), then checks that every parameter received a
        non-None, non-NaN gradient.
        """
        net = SchedulingNetwork()
        obs = make_dummy_obs()
        actions = random_actions()
        lp, val, ent = net.evaluate(obs, actions)
        loss = -lp.mean() + val.mean() - 0.01 * ent.mean()
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"

    def test_deterministic_action(self):
        """@brief Two deterministic calls on the same input must return
        the same action."""
        net = SchedulingNetwork()
        net.eval()
        obs = make_dummy_obs(1)
        a1, _, _ = net.act(obs, deterministic=True)
        a2, _, _ = net.act(obs, deterministic=True)
        assert torch.equal(a1, a2), "Deterministic actions should match"

    def test_param_summary(self):
        """@brief @ref SchedulingNetwork.get_param_summary totals must be
        consistent with submodule counts."""
        net = SchedulingNetwork()
        summary = net.get_param_summary()
        assert summary["total"] > 0
        part_sum = sum(v for k, v in summary.items() if k != "total")
        # Parts should roughly equal total (some params might be shared)
        assert abs(part_sum - summary["total"]) < 100

    def test_checkpoint_roundtrip(self):
        """@brief Save and reload model weights, verify outputs match.

        @details
        Serialises the state dict to a temporary file, loads it into a
        fresh @ref SchedulingNetwork, and asserts that forward-pass outputs
        are identical within floating-point tolerance.
        """
        net1 = SchedulingNetwork()
        net1.eval()
        obs = make_dummy_obs(1)
        with torch.no_grad():
            logits1, val1 = net1(obs)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(net1.state_dict(), f.name)
            net2 = SchedulingNetwork()
            net2.load_state_dict(torch.load(f.name, weights_only=True))
            net2.eval()

        with torch.no_grad():
            logits2, val2 = net2(obs)

        assert torch.allclose(logits1, logits2, atol=1e-6)
        assert torch.allclose(val1, val2, atol=1e-6)
        os.unlink(f.name)

    @pytest.mark.parametrize("grid_norm", ["group", "batch"])
    def test_rollout_and_update_modes_compute_the_same_function(self, grid_norm):
        """@brief PPO acts in eval mode and re-scores in train mode on minibatches; with no weight change the
        ratio must be 1. BatchNorm in train mode broke this (minibatch statistics); GroupNorm and pinned legacy
        BatchNorm must not."""
        from config import EncoderConfig
        torch.manual_seed(0)
        net = SchedulingNetwork(EncoderConfig(grid_norm=grid_norm))
        obs = make_dummy_obs(8)
        net.eval()
        with torch.no_grad():
            actions, log_probs, values = net.act(obs)
        net.train()
        with torch.no_grad():
            half = {k: v[:4] for k, v in obs.items()}
            new_lp, new_v, _ = net.evaluate(half, actions[:4])
        assert torch.allclose(new_lp, log_probs[:4], atol=1e-5)
        assert torch.allclose(new_v, values[:4], atol=1e-5)

    def test_default_grid_norm_is_group_with_no_batchnorm(self):
        net = SchedulingNetwork()
        assert net.encoder_cfg.grid_norm == "group"
        assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in net.modules())

    def test_encoder_config_for_detects_legacy_batchnorm_checkpoints(self):
        """@brief Checkpoints from before 2026-10-01 (BatchNorm) must still load, e.g. rnd02 in evaluate.py."""
        from config import EncoderConfig
        from models.network import encoder_config_for
        legacy = SchedulingNetwork(EncoderConfig(grid_norm="batch")).state_dict()
        assert encoder_config_for(legacy).grid_norm == "batch"
        assert encoder_config_for(SchedulingNetwork().state_dict()).grid_norm == "group"
        SchedulingNetwork(encoder_config_for(legacy)).load_state_dict(legacy)   # strict load succeeds


# ============================================================
#  3. Rollout buffer tests
# ============================================================

class TestRolloutBuffer:
    """@brief Tests for @ref RolloutBuffer GAE computation and batch generation."""

    def test_gae_computation(self):
        """@brief GAE advantages must be non-trivial and returns must
        equal advantages + values.

        @details
        Fills the buffer with 8 steps of random data for 2 parallel envs,
        computes GAE, then verifies the identity
        @c returns = @c advantages + @c values.
        """
        obs_shapes = dict(OBS_SHAPES)
        buf = RolloutBuffer(
            rollout_length=8, num_envs=2,
            obs_shapes=obs_shapes, gamma=0.99, gae_lambda=0.95,
        )
        for t in range(8):
            obs = {k: np.random.randn(2, *s).astype(np.float32)
                   for k, s in obs_shapes.items()}
            buf.add(
                obs,
                actions=np.array([0, 1]),
                log_probs=np.array([-1.0, -1.5], dtype=np.float32),
                rewards=np.array([1.0, 0.5], dtype=np.float32),
                values=np.array([0.5, 0.3], dtype=np.float32),
                dones=np.array([0.0, 0.0], dtype=np.float32),
            )
        buf.compute_gae(last_values=np.array([0.4, 0.2], dtype=np.float32))
        # Advantages should be non-trivial
        assert not np.allclose(buf.advantages, 0)
        # Returns = advantages + values
        np.testing.assert_allclose(
            buf.returns, buf.advantages + buf.values, atol=1e-6
        )

    def test_batch_generation(self):
        """@brief Mini-batch iterator must yield the correct number of batches.

        @details
        With 4 steps × 2 envs = 8 total transitions and a batch size of 4,
        exactly 2 batches should be produced.
        """
        obs_shapes = {"global_scalars": (10,)}
        buf = RolloutBuffer(4, 2, obs_shapes)
        for t in range(4):
            buf.add(
                {"global_scalars": np.random.randn(2, 10).astype(np.float32)},
                np.array([0, 0]),
                np.array([-1.0, -1.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
            )
        buf.compute_gae(np.zeros(2, dtype=np.float32))
        batches = list(buf.get_batches(batch_size=4))
        assert len(batches) == 2  # 8 total / 4 batch

    def test_gae_stops_at_episode_boundary(self):
        """@brief A step whose done flag is set must not bootstrap from the next observation.

        @details
        dones[t] is the done flag returned by the step taken from obs t, so step 0 ending
        its episode means obs 1 starts a new one: advantage[0] = r0 - V0, while the final
        step still bootstraps from last_values.
        """
        buf = RolloutBuffer(2, 1, {"global_scalars": (10,)}, gamma=0.99, gae_lambda=0.95)
        for done in (1.0, 0.0):
            buf.add(
                {"global_scalars": np.zeros((1, 10), dtype=np.float32)},
                np.array([0]),
                np.array([0.0], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
                np.array([0.5], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
        buf.compute_gae(last_values=np.array([10.0], dtype=np.float32))
        np.testing.assert_allclose(buf.advantages[1], [1.0 + 0.99 * 10.0 - 0.5], atol=1e-5)
        np.testing.assert_allclose(buf.advantages[0], [0.5], atol=1e-5)

    def test_truncation_bootstrap_substitutes_for_the_zeroed_next_value(self):
        """@brief A step whose episode ended by truncation must bootstrap from
        gamma * V(terminal_obs) (bootstrap_values) instead of the zeroed-out next_values term —
        dones[t]=1 there too (same as a real terminal), since the next stored observation
        belongs to a new episode either way; only the source of the bootstrap differs.
        """
        buf = RolloutBuffer(2, 1, {"global_scalars": (10,)}, gamma=0.99, gae_lambda=0.95)
        # Step 0: truncated (not a real terminal) with a known terminal-obs value estimate.
        buf.add(
            {"global_scalars": np.zeros((1, 10), dtype=np.float32)},
            np.array([0]), np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32),
            np.array([0.5], dtype=np.float32), np.array([1.0], dtype=np.float32),
            bootstrap_values=np.array([7.0], dtype=np.float32),
        )
        # Step 1: an ordinary non-terminal step (default bootstrap_values omitted -> 0).
        buf.add(
            {"global_scalars": np.zeros((1, 10), dtype=np.float32)},
            np.array([0]), np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32),
            np.array([0.5], dtype=np.float32), np.array([0.0], dtype=np.float32),
        )
        buf.compute_gae(last_values=np.array([10.0], dtype=np.float32))

        # delta[0] = r + gamma*bootstrap_values[0] - V[0]; next_values[0]=values[1] is correctly
        # ignored (next_nonterminal=0), and step 1's untouched delta chains in via gae.
        delta1 = 1.0 + 0.99 * 10.0 - 0.5
        expected0 = (1.0 + 0.99 * 7.0 - 0.5) + 0.99 * 0.95 * 0.0 * delta1
        np.testing.assert_allclose(buf.advantages[0], [expected0], atol=1e-5)
        np.testing.assert_allclose(buf.advantages[1], [delta1], atol=1e-5)

    def test_add_defaults_bootstrap_values_to_zero(self):
        """@brief Omitting bootstrap_values (the common case: no truncation this step) must not
        add any correction — same result as the pre-existing (pre-truncation-support) behaviour."""
        buf = RolloutBuffer(1, 1, {"global_scalars": (10,)}, gamma=0.99, gae_lambda=0.95)
        buf.add(
            {"global_scalars": np.zeros((1, 10), dtype=np.float32)},
            np.array([0]), np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32),
            np.array([0.5], dtype=np.float32), np.array([0.0], dtype=np.float32),
        )
        buf.compute_gae(last_values=np.array([10.0], dtype=np.float32))
        np.testing.assert_allclose(buf.advantages[0], [1.0 + 0.99 * 10.0 - 0.5], atol=1e-5)

    @staticmethod
    def _fill(buf, rewards, values, dones, dts=None, bootstrap_values=None):
        for t in range(len(rewards)):
            buf.add(
                {"global_scalars": np.zeros((1, 10), dtype=np.float32)},
                np.array([0]), np.array([0.0], dtype=np.float32),
                np.array([rewards[t]], dtype=np.float32), np.array([values[t]], dtype=np.float32),
                np.array([dones[t]], dtype=np.float32),
                bootstrap_values=None if bootstrap_values is None
                else np.array([bootstrap_values[t]], dtype=np.float32),
                dts=None if dts is None else np.array([dts[t]], dtype=np.float32),
            )

    def test_time_discount_with_unit_steps_matches_per_decision(self):
        """@brief gamma_per_second = gamma with every dt = 1 s must reproduce per-decision GAE exactly."""
        rewards, values, dones = [1.0, -2.0, 0.5, 3.0], [0.5, 0.1, -0.3, 2.0], [0.0, 1.0, 0.0, 0.0]
        old = RolloutBuffer(4, 1, {"global_scalars": (10,)}, gamma=0.99, gae_lambda=0.95)
        self._fill(old, rewards, values, dones)
        new = RolloutBuffer(4, 1, {"global_scalars": (10,)}, gamma=0.5, gae_lambda=0.95,
                            gamma_per_second=0.99)
        self._fill(new, rewards, values, dones, dts=[1.0] * 4)
        last = np.array([4.0], dtype=np.float32)
        old.compute_gae(last)
        new.compute_gae(last)
        np.testing.assert_allclose(new.advantages, old.advantages, atol=1e-6)
        np.testing.assert_allclose(new.returns, old.returns, atol=1e-6)

    def test_time_discount_uses_each_steps_elapsed_time(self):
        """@brief Uneven dt: step t is discounted by gamma_s ** dt[t] in the bootstrap and the GAE recursion."""
        gs, lam = 0.999, 0.95
        rewards, values, dts = [1.0, 2.0], [0.5, 0.25], [10.0, 300.0]
        buf = RolloutBuffer(2, 1, {"global_scalars": (10,)}, gamma=0.99, gae_lambda=lam, gamma_per_second=gs)
        self._fill(buf, rewards, values, [0.0, 0.0], dts=dts)
        buf.compute_gae(last_values=np.array([10.0], dtype=np.float32))
        g0, g1 = gs ** 10.0, gs ** 300.0
        delta1 = 2.0 + g1 * 10.0 - 0.25
        delta0 = 1.0 + g0 * 0.25 - 0.5
        np.testing.assert_allclose(buf.advantages[1], [delta1], rtol=1e-5)
        np.testing.assert_allclose(buf.advantages[0], [delta0 + g0 * lam * delta1], rtol=1e-5)

    def test_time_discount_applies_to_truncation_bootstrap(self):
        """@brief A truncated step bootstraps V(terminal_obs) with that step's own gamma_s ** dt."""
        gs = 0.999
        buf = RolloutBuffer(1, 1, {"global_scalars": (10,)}, gamma=0.99, gae_lambda=0.95, gamma_per_second=gs)
        self._fill(buf, [1.0], [0.5], [1.0], dts=[50.0], bootstrap_values=[7.0])
        buf.compute_gae(last_values=np.array([10.0], dtype=np.float32))
        np.testing.assert_allclose(buf.advantages[0], [1.0 + gs ** 50.0 * 7.0 - 0.5], rtol=1e-5)

    def test_time_discount_requires_dts(self):
        buf = RolloutBuffer(1, 1, {"global_scalars": (10,)}, gamma=0.99, gae_lambda=0.95, gamma_per_second=0.999)
        with pytest.raises(ValueError, match="no dts"):
            self._fill(buf, [1.0], [0.5], [0.0])


class TestDiscountConfig:
    """@brief PPOConfig.discount_horizon_s -> gamma_s, and the trainer's per-step durations."""

    def test_default_is_time_discount_with_3000_s_horizon(self):
        cfg = PPOConfig()
        assert cfg.discount_horizon_s == 3000.0
        assert cfg.gamma_per_second == pytest.approx(1 - 1 / 3000)

    def test_none_means_per_decision(self):
        assert PPOConfig(discount_horizon_s=None).gamma_per_second is None

    def test_step_durations_rejects_a_missing_dt(self):
        from train import step_durations
        np.testing.assert_allclose(step_durations([{"dt": 12.5}, {"dt": 0.0}]), [12.5, 0.0])
        with pytest.raises(RuntimeError, match="info\\['dt'\\]"):
            step_durations([{"dt": 12.5}, {"dt": None}])

    def test_elapsed_sim_time(self):
        from rewards import MetricsSnapshot, elapsed_sim_time
        a, b = MetricsSnapshot.from_dict({"sim_time": 100.0}), MetricsSnapshot.from_dict({"sim_time": 113.5})
        assert elapsed_sim_time(a, b) == pytest.approx(13.5)
        assert elapsed_sim_time(None, b) is None


# ============================================================
#  4. Observation slicing tests (unity_env.slice_obs)
# ============================================================

class TestSliceObs:
    """@brief Tests for @ref slice_obs: flat vector → named observation dict (schema v2)."""

    def _make_flat_obs(self) -> np.ndarray:
        """@brief Flat vector of length TOTAL_OBS_SIZE with a distinct constant per stream."""
        raw = np.zeros(TOTAL_OBS_SIZE, dtype=np.float32)
        raw[:SLICE_SPATIAL_END] = 0.1
        raw[SLICE_SPATIAL_END:SLICE_MACHINES_END] = 0.2
        raw[SLICE_MACHINES_END:SLICE_JOBS_END] = 0.3
        raw[SLICE_JOBS_END:SLICE_SCALARS_END] = 0.4
        raw[SLICE_SCALARS_END:SLICE_FLAGS_END] = 0.5
        return raw

    def test_output_keys_and_shapes(self):
        """@brief slice_obs yields every observation stream; action_mask comes from the step's masks, not the vector."""
        d = slice_obs(self._make_flat_obs())
        assert set(d) == set(SLICED_SHAPES)
        for k, shape in SLICED_SHAPES.items():
            assert d[k].shape == shape, k
            assert d[k].dtype == np.float32, k

    def test_stream_values_preserved(self):
        d = slice_obs(self._make_flat_obs())
        for k, v in (("factory_grid", 0.1), ("machine_table", 0.2), ("job_table", 0.3),
                     ("global_scalars", 0.4), ("event_flags", 0.5)):
            np.testing.assert_allclose(d[k], v, atol=1e-7, err_msg=k)

    def test_tables_are_row_major(self):
        """@brief C# writes table[row * features + col]; row r, col c must land at [r, c]."""
        raw = np.zeros(TOTAL_OBS_SIZE, dtype=np.float32)
        raw[SLICE_SPATIAL_END + 14 * MACHINE_FEATURES + 13] = 1.0   # machine 14, candidate column
        raw[SLICE_MACHINES_END + 3 * JOB_FEATURES + 7] = 1.0        # job row 3, age column
        d = slice_obs(raw)
        assert d["machine_table"][14, 13] == 1.0 and d["machine_table"].sum() == 1.0
        assert d["job_table"][3, 7] == 1.0 and d["job_table"].sum() == 1.0

    def test_wrong_length_raises(self):
        """@brief A v1 player (13,328 floats) must be rejected, not silently mis-sliced."""
        try:
            slice_obs(np.zeros(13_328, dtype=np.float32))
            assert False, "Expected AssertionError"
        except AssertionError as e:
            assert "Expected" in str(e)

    def test_batched_slice(self):
        B = 3
        d = slice_obs(np.zeros((B, TOTAL_OBS_SIZE), dtype=np.float32))
        for k, shape in SLICED_SHAPES.items():
            assert d[k].shape == (B, *shape), k

    def test_total_obs_size_matches_csharp(self):
        """@brief Mirrors ObservationBuilder.TotalObservationSize at the default caps (64*64*3 + 105*16 + 1792*17 + 16 + 6)."""
        assert TOTAL_OBS_SIZE == 51_624
        assert SLICE_FLAGS_END == TOTAL_OBS_SIZE
        assert (MAX_MACHINES, MACHINE_FEATURES, MAX_JOBS, JOB_FEATURES) == (105, 16, 1792, 21)


class TestPerLaunchRowCaps:
    """@brief Row caps chosen per player launch (config.obs_row_caps, scenarios.row_caps_for, slice_obs caps)."""

    def test_obs_row_caps_rule(self):
        """@brief Machine rows = the floor, job rows = 256 per started block of 15 machines."""
        from config import obs_row_caps
        assert obs_row_caps(15) == (15, 256)
        assert obs_row_caps(30) == (30, 512)
        assert obs_row_caps(16) == (16, 512)
        assert obs_row_caps(105) == (105, 1792) == (MAX_MACHINES, MAX_JOBS)

    def test_total_size_at_default_caps(self):
        from config import obs_total_size
        assert obs_total_size(MAX_MACHINES, MAX_JOBS) == TOTAL_OBS_SIZE
        assert obs_total_size(15, 256) == 17_928

    def test_row_caps_for_scenarios(self):
        """@brief Caps fit the largest floor; overrides replace one half; no scenarios -> defaults."""
        from scenarios import row_caps_for
        small = {"machineTypeLayout": ["Mill"] * 15}
        big = {"machineTypeLayout": ["Mill"] * 105}
        assert row_caps_for([small]) == (15, 256)
        assert row_caps_for([small, big]) == (105, 1792)
        assert row_caps_for([small], max_jobs=64) == (15, 64)
        assert row_caps_for([]) == (MAX_MACHINES, MAX_JOBS)

    def test_randomized_generator_floor(self):
        """@brief The training generator's floor (15 machines) gives the small caps."""
        from scenarios import randomized_generator, row_caps_for
        gen = randomized_generator(5400.0)
        assert row_caps_for([gen(10_000 + i) for i in range(3)]) == (15, 256)

    def test_slice_small_caps(self):
        from config import obs_total_size
        raw = np.zeros(obs_total_size(15, 256), dtype=np.float32)
        raw[SLICE_SPATIAL_END + 14 * MACHINE_FEATURES + 13] = 1.0
        raw[SLICE_SPATIAL_END + 15 * MACHINE_FEATURES + 255 * JOB_FEATURES + 7] = 1.0   # last job row
        raw[-1] = 0.5                                                                    # last event flag
        d = slice_obs(raw, 15, 256)
        assert d["machine_table"].shape == (15, MACHINE_FEATURES) and d["machine_table"][14, 13] == 1.0
        assert d["job_table"].shape == (256, JOB_FEATURES) and d["job_table"][255, 7] == 1.0
        assert d["event_flags"][-1] == 0.5
        try:
            slice_obs(raw, MAX_MACHINES, MAX_JOBS)
            assert False, "Expected AssertionError"
        except AssertionError as e:
            assert "row caps" in str(e)

    def test_network_output_independent_of_caps(self):
        """@brief One network, the same floor at small and default caps: identical outputs."""
        torch.manual_seed(0)
        net = SchedulingNetwork().eval()
        small = {k: torch.rand(BATCH, *shape) for k, shape in obs_shapes_for(15, 256).items()}
        small["machine_table"][:, :, 0] = 1.0
        small["job_table"][:, 10:] = 0.0
        small["job_table"][:, :10, 0] = 1.0
        small["action_mask"] = torch.ones(BATCH, ACTION_MASK_LEN)
        big = dict(small)
        big["machine_table"] = torch.zeros(BATCH, MAX_MACHINES, MACHINE_FEATURES)
        big["machine_table"][:, :15] = small["machine_table"]
        big["job_table"] = torch.zeros(BATCH, MAX_JOBS, JOB_FEATURES)
        big["job_table"][:, :256] = small["job_table"]
        with torch.no_grad():
            (la, va), (lb, vb) = net(small), net(big)
        torch.testing.assert_close(la, lb, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(va, vb, atol=1e-5, rtol=1e-5)


class TestCheckpointSchema:
    """@brief Checkpoint compatibility follows the per-row feature layout, not the row caps."""

    def _check(self, ckpt):
        from train import check_obs_schema
        check_obs_schema(ckpt, "ckpt.pt")

    def test_v1_checkpoint_rejected(self):
        try:
            self._check({"model_state_dict": {}})   # no obs_layout = v1
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "schema v1" in str(e)

    def test_current_layout_accepted(self):
        self._check({"obs_layout": dict(OBS_LAYOUT)})

    def test_different_row_caps_accepted(self):
        """@brief Caps are not part of the layout: a checkpoint from a 100-row player is fine in a 200-row one."""
        self._check({"obs_layout": dict(OBS_LAYOUT), "obs_row_caps": {"max_machines": 40, "max_jobs": 32}})

    def test_different_feature_width_rejected(self):
        layout = dict(OBS_LAYOUT, machine_features=OBS_LAYOUT["machine_features"] + 1)
        try:
            self._check({"obs_layout": layout})
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "machine_features" in str(e)

    def test_weights_run_on_a_larger_table(self):
        """@brief The same weights accept more machine/job rows (a player built with bigger caps)."""
        net = SchedulingNetwork().eval()
        obs = make_dummy_obs(2)
        big = dict(obs)
        big["machine_table"] = torch.cat([obs["machine_table"], torch.zeros(2, 100, MACHINE_FEATURES)], dim=1)
        big["job_table"] = torch.cat([obs["job_table"], torch.zeros(2, 64, JOB_FEATURES)], dim=1)
        with torch.no_grad():
            torch.testing.assert_close(net(obs)[0], net(big)[0])



class TestActionSpace:
    """@brief Two-branch action space: checkpoint layout, Unity masks, branched rollout storage."""

    def test_v1_action_checkpoint_rejected(self):
        from train import check_action_layout
        try:
            check_action_layout({"obs_layout": dict(OBS_LAYOUT)}, "ckpt.pt")   # no action_layout = v1
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "action schema v1" in str(e)

    def test_different_head_rejected(self):
        from train import check_action_layout
        layout = dict(ACTION_LAYOUT, machine_head=["ECT", "SRWT"])
        try:
            check_action_layout({"action_layout": layout}, "ckpt.pt")
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "action layout differs" in str(e)

    def test_current_action_layout_accepted(self):
        from train import check_action_layout
        check_action_layout({"action_layout": dict(ACTION_LAYOUT)}, "ckpt.pt")

    def test_mask_from_mlagents(self):
        """@brief ML-Agents masks are per branch, True = masked; obs["action_mask"] is 1 = enabled, concatenated."""
        from env_wrappers.unity_env import action_mask_from
        steps = MagicMock()
        steps.action_mask = [np.array([[False, True, True, True]]), np.array([[False, False, False]])]
        np.testing.assert_array_equal(action_mask_from(steps), [1, 0, 0, 0, 1, 1, 1])

    def test_no_mask_enables_everything(self):
        from env_wrappers.unity_env import action_mask_from
        steps = MagicMock()
        steps.action_mask = None
        np.testing.assert_array_equal(action_mask_from(steps), np.ones(ACTION_MASK_LEN))

    def test_step_sends_both_branches(self):
        """@brief The wrapper hands Unity one row (job head, machine head)."""
        env, mock_unity = TestUnitySchedulingEnv()._make_env(
            [(_make_mock_steps(np.random.rand(TOTAL_OBS_SIZE).astype(np.float32)), _empty_steps())])
        env.step((2, 1))
        sent = mock_unity.set_actions.call_args[0][1].discrete
        np.testing.assert_array_equal(sent, [[2, 1]])

    def test_legacy_player_refused(self):
        """@brief A player still built with one 8-way branch must fail at connect, not mid-training."""
        from env_wrappers.unity_env import UnitySchedulingEnv
        with patch("env_wrappers.unity_env.LoopbackUnityEnvironment") as MockUnity, \
             patch("env_wrappers.unity_env.EngineConfigurationChannel"):
            spec = MagicMock()
            obs_spec = MagicMock()
            obs_spec.shape, obs_spec.name = (TOTAL_OBS_SIZE,), "VectorSensor"
            spec.observation_specs = [obs_spec]
            spec.action_spec.discrete_branches = (8,)
            MockUnity.return_value.behavior_specs = {"SchedulingBehavior?team=0": spec}
            try:
                UnitySchedulingEnv(file_name=None)
                assert False, "Expected RuntimeError"
            except RuntimeError as e:
                assert "Rebuild the player" in str(e)

    def test_buffer_stores_branched_actions(self):
        buf = RolloutBuffer(2, 3, {"global_scalars": (4,)}, action_shape=(len(ACTION_BRANCHES),))
        for t in range(2):
            buf.add({"global_scalars": np.zeros((3, 4), dtype=np.float32)},
                    np.array([[0, 1], [2, 0], [3, 2]]), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        buf.compute_gae(np.zeros(3, dtype=np.float32))
        batch = next(buf.get_batches(6))
        assert batch["actions"].shape == (6, len(ACTION_BRANCHES))


# ============================================================
#  5. Unity environment wrapper tests (mocked)
# ============================================================

def _make_mock_steps(obs_array, reward=0.0, n_agents=1):
    """@brief Build a mock DecisionSteps/TerminalSteps object.

    @param obs_array  The flat observation vector to return.
    @param reward     Scalar reward for agent 0.
    @param n_agents   Number of agents (controls len()).
    @return A MagicMock that behaves like DecisionSteps or TerminalSteps.
    """
    steps = MagicMock()
    steps.obs = [obs_array.reshape(1, -1)]  # (1, TOTAL_OBS_SIZE)
    steps.reward = np.array([reward], dtype=np.float32)
    steps.interrupted = np.array([False] * n_agents)
    steps.action_mask = None
    steps.__len__ = lambda self: n_agents
    return steps


def _empty_steps():
    """@brief Build a mock steps object with len() == 0 (silent frame)."""
    steps = MagicMock()
    steps.__len__ = lambda self: 0
    return steps


class TestUnitySchedulingEnv:
    """@brief Tests for @ref UnitySchedulingEnv using a mocked UnityEnvironment.

    @details
    These tests verify the decision loop, looping through silent frames,
    episode termination and rollover, and Python-side reward computation
    — all without requiring a running Unity process.
    """

    def _make_env(self, get_steps_sequence, reward_fn=None, with_metrics=False, seed_rng=None,
                 scenario_generator=None):
        """@brief Construct a UnitySchedulingEnv with a fully mocked backend.

        @param get_steps_sequence  List of (decision_steps, terminal_steps)
                                   tuples returned by successive get_steps calls.
        @param reward_fn           Optional Python reward function.
        @param with_metrics        Also advertise the reward-metrics sensor spec.
        @return A patched UnitySchedulingEnv instance.
        """
        from env_wrappers.unity_env import UnitySchedulingEnv
        from rewards import METRIC_NAMES, SENSOR_NAME

        with patch("env_wrappers.unity_env.LoopbackUnityEnvironment") as MockUnity, \
             patch("env_wrappers.unity_env.EngineConfigurationChannel"):

            mock_env_instance = MagicMock()

            # Set up behavior_specs to return the correct obs shape
            mock_spec = MagicMock()
            mock_obs_spec = MagicMock()
            mock_obs_spec.shape = (TOTAL_OBS_SIZE,)
            mock_obs_spec.name = f"VectorSensor_size{TOTAL_OBS_SIZE}"
            mock_spec.observation_specs = [mock_obs_spec]
            mock_spec.action_spec.discrete_branches = ACTION_BRANCHES
            if with_metrics:
                metrics_spec = MagicMock()
                metrics_spec.shape = (len(METRIC_NAMES),)
                metrics_spec.name = SENSOR_NAME
                mock_spec.observation_specs.append(metrics_spec)

            mock_env_instance.behavior_specs = {"SchedulingBehavior?team=0": mock_spec}

            # __init__ calls self.env.reset() but never get_steps(),
            # so the side-effect list starts at the caller's sequence.
            mock_env_instance.get_steps.side_effect = list(get_steps_sequence)

            MockUnity.return_value = mock_env_instance

            env = UnitySchedulingEnv(file_name=None, time_scale=1.0, reward_fn=reward_fn,
                                     seed_rng=seed_rng, scenario_generator=scenario_generator)
            return env, mock_env_instance

    def test_step_returns_on_decision(self):
        """@brief When Unity immediately returns a decision step, the
        wrapper must return that observation and reward without looping."""
        obs_vec = np.random.rand(TOTAL_OBS_SIZE).astype(np.float32)
        decision = _make_mock_steps(obs_vec, reward=1.5, n_agents=1)
        terminal = _empty_steps()

        env, _ = self._make_env([(decision, terminal)])
        obs, reward, done, info = env.step((3, 0))

        assert not done
        assert abs(reward - 1.5) < 1e-6
        assert set(obs.keys()) == {
            "factory_grid", "machine_table", "job_table", "action_mask",
            "global_scalars", "event_flags",
        }

    def test_row_caps_passed_to_player_and_checked(self):
        """@brief obs_caps become -obsmaxmachines / -obsmaxjobs, and the spec size is checked against them."""
        from config import obs_total_size
        from env_wrappers.unity_env import UnitySchedulingEnv
        size = obs_total_size(15, 256)
        with patch("env_wrappers.unity_env.LoopbackUnityEnvironment") as MockUnity, \
             patch("env_wrappers.unity_env.EngineConfigurationChannel"):
            spec = MagicMock()
            obs_spec = MagicMock()
            obs_spec.shape, obs_spec.name = (size,), "VectorSensor"
            spec.observation_specs = [obs_spec]
            spec.action_spec.discrete_branches = ACTION_BRANCHES
            MockUnity.return_value.behavior_specs = {"SchedulingBehavior?team=0": spec}
            decision = _make_mock_steps(np.zeros(size, dtype=np.float32))
            MockUnity.return_value.get_steps.side_effect = [(decision, _empty_steps())]
            env = UnitySchedulingEnv(file_name=None, obs_caps=(15, 256))
            args = MockUnity.call_args.kwargs["additional_args"]
            assert args[args.index("-obsmaxmachines") + 1] == "15"
            assert args[args.index("-obsmaxjobs") + 1] == "256"
            obs, _, _, _ = env.step((0, 0))
            assert obs["machine_table"].shape == (15, MACHINE_FEATURES)
            assert obs["job_table"].shape == (256, JOB_FEATURES)
            try:
                UnitySchedulingEnv(file_name=None)   # default caps vs a 15-machine-sized player
                assert False, "Expected AssertionError"
            except AssertionError as e:
                assert "rebuild" in str(e)

    def test_step_returns_on_terminal(self):
        """@brief On a terminal step done must be True, and the returned obs must already
        be the next episode's first observation (Unity restarts episodes on its own)."""
        obs_vec = np.random.rand(TOTAL_OBS_SIZE).astype(np.float32)
        next_vec = np.full(TOTAL_OBS_SIZE, 0.25, dtype=np.float32)
        terminal = _make_mock_steps(obs_vec, reward=10.0, n_agents=1)
        next_decision = _make_mock_steps(next_vec, n_agents=1)

        env, mock_unity = self._make_env([
            (_empty_steps(), terminal),
            (next_decision, _empty_steps()),
        ])
        obs, reward, done, info = env.step((0, 0))

        assert done
        assert abs(reward - 10.0) < 1e-6
        assert info["episode"]["length"] == 1
        np.testing.assert_allclose(obs["global_scalars"], 0.25)
        # Only __init__'s reset: episodes must roll over without a Python-side reset.
        mock_unity.reset.assert_called_once()

    def test_accumulates_reward_across_silent_frames(self):
        """@brief Rewards from silent frames must be accumulated.

        @details Three silent frames (both steps empty) followed by a
        decision step.  Only the decision step carries reward, but the
        wrapper must have looped through all four frames.
        """
        obs_vec = np.random.rand(TOTAL_OBS_SIZE).astype(np.float32)
        silent = (_empty_steps(), _empty_steps())
        decision = _make_mock_steps(obs_vec, reward=2.0, n_agents=1)
        terminal = _empty_steps()

        # 3 silent frames, then a decision
        env, mock_unity = self._make_env([
            silent, silent, silent, (decision, terminal)
        ])
        obs, reward, done, info = env.step((1, 0))

        assert not done
        # Only the decision frame contributes reward (silent frames
        # have no steps to read reward from).
        assert abs(reward - 2.0) < 1e-6
        # Unity.step() should have been called 4 times for this step()
        # (plus 1 from __init__'s reset).
        assert mock_unity.step.call_count >= 4

    def test_reset_returns_valid_obs(self):
        """@brief reset() must loop until a decision and return a valid obs dict."""
        obs_vec = np.random.rand(TOTAL_OBS_SIZE).astype(np.float32)
        decision = _make_mock_steps(obs_vec, reward=0.0, n_agents=1)
        terminal = _empty_steps()

        env, _ = self._make_env([(decision, terminal)])
        obs = env.reset()

        assert set(obs.keys()) == {
            "factory_grid", "machine_table", "job_table", "action_mask",
            "global_scalars", "event_flags",
        }
        assert obs["factory_grid"].shape == (GRID_CHANNELS, GRID_SIZE, GRID_SIZE)

    def test_close_delegates_to_unity(self):
        """@brief close() must call the underlying Unity env's close()."""
        obs_vec = np.random.rand(TOTAL_OBS_SIZE).astype(np.float32)
        decision = _make_mock_steps(obs_vec, reward=0.0, n_agents=1)
        terminal = _empty_steps()

        env, mock_unity = self._make_env([(decision, terminal)])
        env.close()
        mock_unity.close.assert_called_once()

    def test_python_reward_from_metrics(self):
        """@brief With a reward function, the reward comes from consecutive metric
        snapshots and Unity's own reward is ignored — including when a terminal step and
        the next episode's first decision arrive in the same batch."""
        from rewards import MetricsSnapshot, load_reward

        reward_fn = load_reward({
            "entry": "rewards/functions/time_penalty.py:TimePenaltyReward",
            "params": {"time_scale": 1.0},
        }).build()

        def steps_at(sim_time):
            obs = np.zeros(TOTAL_OBS_SIZE, dtype=np.float32)
            metrics = MetricsSnapshot.from_dict({"sim_time": sim_time}).to_array().astype(np.float32)
            steps = _make_mock_steps(obs, reward=99.0)
            steps.obs = [obs.reshape(1, -1), metrics.reshape(1, -1)]
            return steps

        env, _ = self._make_env([
            (steps_at(0.0), _empty_steps()),   # reset -> first decision
            (steps_at(4.0), _empty_steps()),   # step 1
            (steps_at(0.0), steps_at(9.0)),    # step 2: episode ends at t=9; next episode starts
            (steps_at(2.5), _empty_steps()),   # step 3, in the new episode
        ], reward_fn=reward_fn, with_metrics=True)

        env.reset()
        assert abs(env.step((0, 0))[1] - (-4.0)) < 1e-6

        _, reward, done, info = env.step((0, 0))
        assert done
        assert abs(reward - (-5.0)) < 1e-6
        assert info["episode"]["makespan"] == 9.0
        assert abs(info["episode"]["return"] - (-9.0)) < 1e-6
        assert abs(info["episode"]["reward_terms"]["time"] - (-9.0)) < 1e-6

        # Measured from the new episode's first snapshot (t=0), not the old terminal (t=9).
        assert abs(env.step((0, 0))[1] - (-2.5)) < 1e-6

    def test_seed_rng_keeps_unity_seed_queue_filled(self):
        """@brief With a seed RNG, reset() replaces Unity's seed queue with a buffer of
        training seeds and every finished episode tops it up by one. The unseeded episode
        already running at reset (the player's default config) is played out inside reset()
        and never returned (audit P1.2), so the first observation is the first seeded episode's."""
        from rewards import MetricsSnapshot
        from env_wrappers.unity_env import SEED_BUFFER, TRAIN_SEED_LOW

        def steps_with_seed(seed, index):
            obs = np.zeros(TOTAL_OBS_SIZE, dtype=np.float32)
            metrics = MetricsSnapshot.from_dict(
                {"episode_seed": seed, "episode_seed_index": index}).to_array().astype(np.float32)
            steps = _make_mock_steps(obs)
            steps.obs = [obs.reshape(1, -1), metrics.reshape(1, -1)]
            return steps

        env, _ = self._make_env([
            (steps_with_seed(-1, -1), _empty_steps()),                # reset: unseeded episode already running
            (steps_with_seed(12345, 0), steps_with_seed(-1, -1)),     # it ends; first seeded episode begins
        ], with_metrics=True, seed_rng=np.random.default_rng(0))
        env.seed_channel = MagicMock()

        env.reset()
        first = env.seed_channel.queue_seeds.call_args_list[0]
        assert len(first.args[0]) == SEED_BUFFER
        assert first.kwargs["clear"] is True
        assert all(s >= TRAIN_SEED_LOW for s in first.args[0])

        # reset() already stepped through the unseeded episode's end and topped the queue up by one.
        top_up = env.seed_channel.queue_seeds.call_args_list[1]
        assert len(top_up.args[0]) == 1
        assert top_up.kwargs["clear"] is False
        assert env.current_metrics.episode_seed == 12345
        assert env.episodes_completed == 0

    def test_scenario_generator_requires_seed_rng(self):
        """@brief scenario_generator needs a seed to build each variant from; fail fast,
        before even connecting to Unity, if seed_rng wasn't also given."""
        from env_wrappers.unity_env import UnitySchedulingEnv

        try:
            UnitySchedulingEnv(file_name=None, scenario_generator=lambda seed: {})
            raise AssertionError("construction should have failed")
        except ValueError as exc:
            assert "seed_rng" in str(exc)

    def test_scenario_generator_queues_scenarios_in_lockstep_with_seeds(self):
        """@brief With a scenario_generator, reset() and every episode end must queue matching
        scenarios alongside the seed queue: same seeds, same batch sizes, same clear flag."""
        from rewards import MetricsSnapshot
        from env_wrappers.unity_env import SEED_BUFFER

        def steps_with_seed(seed, index):
            obs = np.zeros(TOTAL_OBS_SIZE, dtype=np.float32)
            metrics = MetricsSnapshot.from_dict(
                {"episode_seed": seed, "episode_seed_index": index}).to_array().astype(np.float32)
            steps = _make_mock_steps(obs)
            steps.obs = [obs.reshape(1, -1), metrics.reshape(1, -1)]
            return steps

        generator = MagicMock(side_effect=lambda seed: {"name": f"variant-{seed}", "seed": seed})
        env, _ = self._make_env([
            (steps_with_seed(-1, -1), _empty_steps()),
            (steps_with_seed(999, 0), steps_with_seed(-1, -1)),
        ], with_metrics=True, seed_rng=np.random.default_rng(0), scenario_generator=generator)
        env.seed_channel = MagicMock()
        env.config_channel = MagicMock()

        env.reset()
        seed_call = env.seed_channel.queue_seeds.call_args_list[0]
        scenario_call = env.config_channel.queue_scenarios.call_args_list[0]
        assert len(scenario_call.args[0]) == SEED_BUFFER == len(seed_call.args[0])
        assert scenario_call.kwargs["clear"] is True
        # Each queued scenario was built from the matching queued seed, in the same order.
        assert [s["seed"] for s in scenario_call.args[0]] == list(seed_call.args[0])
        assert generator.call_args_list[:SEED_BUFFER] == [((s,),) for s in seed_call.args[0]]

        # reset() played out the unseeded episode, which topped both queues up by exactly one.
        seed_top_up = env.seed_channel.queue_seeds.call_args_list[1]
        scenario_top_up = env.config_channel.queue_scenarios.call_args_list[1]
        assert len(seed_top_up.args[0]) == 1 and len(scenario_top_up.args[0]) == 1
        assert scenario_top_up.args[0][0]["seed"] == seed_top_up.args[0][0]
        assert scenario_top_up.kwargs["clear"] is False

    def test_truncated_episode_reports_flag_and_terminal_obs(self):
        """@brief A truncated (time-limit) episode end must set info["episode"]["truncated"] and
        carry the ended episode's last decision observation as info["terminal_obs"] — distinct from
        obs/next_obs, which by then is already the next episode's first frame (Unity auto-resets),
        and from the terminal step's own observation, which the player pads with zeros."""
        from rewards import MetricsSnapshot

        ended_obs = np.full(TOTAL_OBS_SIZE, 0.7, dtype=np.float32)
        padded_obs = np.zeros(TOTAL_OBS_SIZE, dtype=np.float32)
        next_obs = np.full(TOTAL_OBS_SIZE, 0.1, dtype=np.float32)

        def steps(obs_vec, truncated=0):
            metrics = MetricsSnapshot.from_dict({"truncated": truncated}).to_array().astype(np.float32)
            s = _make_mock_steps(obs_vec)
            s.obs = [obs_vec.reshape(1, -1), metrics.reshape(1, -1)]
            return s

        env, _ = self._make_env([
            (steps(ended_obs), _empty_steps()),
            (steps(next_obs), steps(padded_obs, truncated=1)),
        ], with_metrics=True)

        env.reset()
        _, _, done, info = env.step((0, 0))

        assert done
        assert info["episode"]["truncated"] is True
        # terminal_obs is the ENDED episode's last decision (uniformly 0.7 here), not the zero-padded
        # terminal step, nor the next episode's first frame that obs/next_obs already carries (0.1).
        np.testing.assert_allclose(info["terminal_obs"]["global_scalars"], 0.7)

    def test_log_file_passed_as_absolute_path(self):
        """@brief A relative log path must reach Unity as an absolute path: the player exits
        at startup if it cannot open the log file."""
        from env_wrappers.unity_env import UnitySchedulingEnv

        with patch("env_wrappers.unity_env.LoopbackUnityEnvironment") as MockUnity, \
             patch("env_wrappers.unity_env.EngineConfigurationChannel"):
            mock_env_instance = MagicMock()
            mock_spec = MagicMock()
            mock_obs_spec = MagicMock()
            mock_obs_spec.shape = (TOTAL_OBS_SIZE,)
            mock_obs_spec.name = f"VectorSensor_size{TOTAL_OBS_SIZE}"
            mock_spec.observation_specs = [mock_obs_spec]
            mock_spec.action_spec.discrete_branches = ACTION_BRANCHES
            mock_env_instance.behavior_specs = {"SchedulingBehavior?team=0": mock_spec}
            MockUnity.return_value = mock_env_instance

            UnitySchedulingEnv(file_name=None, log_file=os.path.join("relative", "dir", "Player.log"),
                               extra_args=["-decisionlogdir", "/tmp/decisions"])

        args = MockUnity.call_args.kwargs["additional_args"]
        log_path = args[args.index("-logFile") + 1]
        assert os.path.isabs(log_path)
        assert log_path.endswith(os.path.join("relative", "dir", "Player.log"))
        assert args[args.index("-decisionlogdir") + 1] == "/tmp/decisions"

    def test_load_scenario_queues_absolute_path_immediately(self):
        """@brief load_scenario() must queue an absolute path right away (like queue_seeds,
        ML-Agents buffers side-channel messages and delivers them on the next round-trip;
        nothing here needs to wait for reset())."""
        obs_vec = np.random.rand(TOTAL_OBS_SIZE).astype(np.float32)
        env, _ = self._make_env([(_make_mock_steps(obs_vec), _empty_steps())])
        env.config_channel = MagicMock()

        env.load_scenario(os.path.join("relative", "single_bottleneck.json"))

        items, clear = env.config_channel.queue_scenarios.call_args.args[0], \
            env.config_channel.queue_scenarios.call_args.kwargs["clear"]
        assert clear is True
        assert len(items) == 1 and os.path.isabs(items[0])
        assert items[0].endswith(os.path.join("relative", "single_bottleneck.json"))


class TestVectorizedUnityEnv:
    """@brief Tests for @ref VectorizedUnityEnv using mocked sub-environments.

    @details
    Patches UnitySchedulingEnv entirely to avoid any Unity dependency.
    Verifies stacking, episode rollover pass-through, and shape correctness.
    """

    def _make_dummy_obs_dict(self):
        """@brief Create a single-env observation dict with valid shapes."""
        return {
            k: np.random.rand(*shape).astype(np.float32) for k, shape in OBS_SHAPES.items()
        }

    def test_reset_stacks_obs(self):
        """@brief reset() must stack observations from N sub-envs along dim 0."""
        from env_wrappers.unity_env import VectorizedUnityEnv

        num_envs = 3
        obs_dicts = [self._make_dummy_obs_dict() for _ in range(num_envs)]

        with patch("env_wrappers.unity_env.UnitySchedulingEnv") as MockSingle:
            mock_instances = []
            for i in range(num_envs):
                m = MagicMock()
                m.reset.return_value = obs_dicts[i]
                mock_instances.append(m)

            MockSingle.side_effect = mock_instances
            vec = VectorizedUnityEnv(num_envs=num_envs, file_name=None)
            obs, infos = vec.reset()

        assert obs["factory_grid"].shape == (num_envs, GRID_CHANNELS, GRID_SIZE, GRID_SIZE)
        assert obs["global_scalars"].shape == (num_envs, GLOBAL_SCALARS)
        assert len(infos) == num_envs

    def test_step_returns_correct_shapes(self):
        """@brief step() must return stacked obs, rewards (N,), dones (N,)."""
        from env_wrappers.unity_env import VectorizedUnityEnv

        num_envs = 2
        obs_dicts = [self._make_dummy_obs_dict() for _ in range(num_envs)]
        reset_obs = [self._make_dummy_obs_dict() for _ in range(num_envs)]

        with patch("env_wrappers.unity_env.UnitySchedulingEnv") as MockSingle:
            mock_instances = []
            for i in range(num_envs):
                m = MagicMock()
                m.reset.return_value = reset_obs[i]
                m.step.return_value = (obs_dicts[i], 0.5, False, {})
                mock_instances.append(m)

            MockSingle.side_effect = mock_instances
            vec = VectorizedUnityEnv(num_envs=num_envs, file_name=None)
            vec.reset()
            obs, rewards, dones, truncs, infos = vec.step(np.array([0, 1]))

        assert rewards.shape == (num_envs,)
        assert dones.shape == (num_envs,)
        assert truncs.shape == (num_envs,)
        assert obs["factory_grid"].shape == (num_envs, GRID_CHANNELS, GRID_SIZE, GRID_SIZE)

    def test_done_env_not_reset_from_python(self):
        """@brief Episodes roll over inside each sub-env, so VectorizedUnityEnv must pass
        the sub-env's returned obs straight through and never reset() on done."""
        from env_wrappers.unity_env import VectorizedUnityEnv

        num_envs = 2
        step_obs = [self._make_dummy_obs_dict() for _ in range(num_envs)]
        # Tag env 0's returned obs (its next episode's first obs) so we can identify it
        step_obs[0]["global_scalars"][:] = 99.0

        with patch("env_wrappers.unity_env.UnitySchedulingEnv") as MockSingle:
            mock_instances = []
            for i in range(num_envs):
                m = MagicMock()
                m.reset.return_value = self._make_dummy_obs_dict()
                # Env 0 is done (terminated, not truncated), env 1 is not
                m.step.return_value = (
                    step_obs[i],
                    1.0 if i == 0 else 0.5,
                    i == 0,  # done for env 0
                    {"episode": {"truncated": False}} if i == 0 else {},
                )
                mock_instances.append(m)

            MockSingle.side_effect = mock_instances
            vec = VectorizedUnityEnv(num_envs=num_envs, file_name=None)
            vec.reset()
            obs, rewards, dones, truncs, infos = vec.step(np.array([0, 0]))

        assert dones[0] == True
        assert dones[1] == False
        assert truncs[0] == False and truncs[1] == False
        np.testing.assert_allclose(obs["global_scalars"][0], 99.0)
        for m in mock_instances:
            m.reset.assert_called_once()
            assert m.global_step == num_envs

    def test_parallel_step_overlaps_envs_and_keeps_order(self):
        """@brief Parallel stepping must run sub-env steps concurrently yet return results in env order."""
        import time
        from env_wrappers.unity_env import VectorizedUnityEnv

        num_envs, delay = 4, 0.2
        obs_dicts = [self._make_dummy_obs_dict() for _ in range(num_envs)]
        for i, obs in enumerate(obs_dicts):
            obs["global_scalars"][:] = i

        def slow_step(i):
            def step(action):
                time.sleep(delay)
                return obs_dicts[i], float(action), False, {}
            return step

        with patch("env_wrappers.unity_env.UnitySchedulingEnv") as MockSingle:
            mock_instances = []
            for i in range(num_envs):
                m = MagicMock()
                m.reset.return_value = obs_dicts[i]
                m.step.side_effect = slow_step(i)
                mock_instances.append(m)
            MockSingle.side_effect = mock_instances

            vec = VectorizedUnityEnv(num_envs=num_envs, file_name=None)
            start = time.perf_counter()
            obs, rewards, dones, truncs, infos = vec.step(np.array([10, 11, 12, 13]))
            elapsed = time.perf_counter() - start
            vec.close()

        assert elapsed < delay * num_envs * 0.75, f"sub-env steps did not overlap ({elapsed:.2f}s)"
        np.testing.assert_allclose(rewards, [10, 11, 12, 13])
        np.testing.assert_allclose(obs["global_scalars"][:, 0], [0, 1, 2, 3])

    def test_sequential_mode_uses_no_pool(self):
        """@brief parallel=False must keep the plain one-by-one stepping path."""
        from env_wrappers.unity_env import VectorizedUnityEnv

        with patch("env_wrappers.unity_env.UnitySchedulingEnv") as MockSingle:
            MockSingle.side_effect = [MagicMock() for _ in range(2)]
            vec = VectorizedUnityEnv(num_envs=2, file_name=None, parallel=False)
        assert vec._pool is None

    def test_failed_construction_closes_started_envs(self):
        """@brief If a later Unity instance fails to start, already-launched players must be closed."""
        from env_wrappers.unity_env import VectorizedUnityEnv

        started = MagicMock()
        with patch("env_wrappers.unity_env.UnitySchedulingEnv") as MockSingle:
            MockSingle.side_effect = [started, RuntimeError("port in use")]
            try:
                VectorizedUnityEnv(num_envs=2, file_name=None)
                raise AssertionError("construction should have failed")
            except RuntimeError as exc:
                assert "port in use" in str(exc)
        started.close.assert_called_once()

    def test_close_all_sub_envs(self):
        """@brief close() must call close() on every sub-environment."""
        from env_wrappers.unity_env import VectorizedUnityEnv

        num_envs = 3
        with patch("env_wrappers.unity_env.UnitySchedulingEnv") as MockSingle:
            mock_instances = [MagicMock() for _ in range(num_envs)]
            for m in mock_instances:
                m.reset.return_value = self._make_dummy_obs_dict()
            MockSingle.side_effect = mock_instances

            vec = VectorizedUnityEnv(num_envs=num_envs, file_name=None)
            vec.close()

        for i, m in enumerate(mock_instances):
            m.close.assert_called_once(), f"Sub-env {i} was not closed"

    def test_scenario_generator_requires_train_seed(self):
        """@brief Each env's variants are keyed by its own drawn seed, so a scenario_generator
        needs --train-seed; fail fast rather than silently falling back to no variants."""
        from env_wrappers.unity_env import VectorizedUnityEnv

        try:
            VectorizedUnityEnv(num_envs=2, file_name=None, scenario_generator=lambda seed: {})
            raise AssertionError("construction should have failed")
        except ValueError as exc:
            assert "train-seed" in str(exc)

    def test_truncated_flag_propagates_per_env(self):
        """@brief truncateds must reflect each env's own episode["truncated"], not just done —
        a truncated and a normally-terminated env in the same step() call must be told apart."""
        from env_wrappers.unity_env import VectorizedUnityEnv

        num_envs = 2
        obs_dicts = [self._make_dummy_obs_dict() for _ in range(num_envs)]
        with patch("env_wrappers.unity_env.UnitySchedulingEnv") as MockSingle:
            mock_instances = []
            for i in range(num_envs):
                m = MagicMock()
                m.reset.return_value = obs_dicts[i]
                m.step.return_value = (obs_dicts[i], 1.0, True, {"episode": {"truncated": i == 0}})
                mock_instances.append(m)
            MockSingle.side_effect = mock_instances

            vec = VectorizedUnityEnv(num_envs=num_envs, file_name=None)
            vec.reset()
            _, _, dones, truncs, _ = vec.step(np.array([0, 0]))

        assert dones[0] and dones[1]
        assert truncs[0] == True and truncs[1] == False

class TestEntropySchedule:
    """@brief --ent-coef / --ent-coef-final: constant by default, linear decay by absolute step otherwise."""

    def test_constant_by_default(self):
        from train import entropy_coef_at
        from config import PPOConfig
        cfg = PPOConfig(total_timesteps=1000, entropy_coef=0.02)
        assert entropy_coef_at(cfg, 0) == entropy_coef_at(cfg, 999) == 0.02

    def test_linear_decay_and_clamp(self):
        from train import entropy_coef_at
        from config import PPOConfig
        cfg = PPOConfig(total_timesteps=1000, entropy_coef=0.01, entropy_coef_final=0.001)
        assert abs(entropy_coef_at(cfg, 0) - 0.01) < 1e-12
        assert abs(entropy_coef_at(cfg, 500) - 0.0055) < 1e-12
        assert abs(entropy_coef_at(cfg, 5000) - 0.001) < 1e-12      # past the end stays at the final value


def test_player_closed_when_startup_check_fails():
    """@brief A failed startup check (here an 8-branch player) closes the player, so the process can exit
    instead of hanging on its gRPC thread (rnd02, 2026-09-26)."""
    from env_wrappers.unity_env import UnitySchedulingEnv
    with patch("env_wrappers.unity_env.LoopbackUnityEnvironment") as MockUnity, \
         patch("env_wrappers.unity_env.EngineConfigurationChannel"):
        spec = MagicMock()
        obs_spec = MagicMock()
        obs_spec.shape, obs_spec.name = (TOTAL_OBS_SIZE,), "VectorSensor"
        spec.observation_specs = [obs_spec]
        spec.action_spec.discrete_branches = (8,)
        MockUnity.return_value.behavior_specs = {"SchedulingBehavior?team=0": spec}
        try:
            UnitySchedulingEnv(file_name=None)
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            assert "action branches" in str(e)
        MockUnity.return_value.close.assert_called_once()
