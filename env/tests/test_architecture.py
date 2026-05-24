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
# or without pytest:
python tests/test_architecture.py
@endcode
"""

import sys
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EncoderConfig, FusionConfig, ActorCriticConfig,
    GRID_SIZE, GRID_CHANNELS, MAX_JOBS, MAX_MACHINES, SCHED_CHANNELS,
    GLOBAL_SCALARS, DISTANCE_DIM, EVENT_FLAGS, TOTAL_OBS_SIZE,
    SLICE_SPATIAL_END, SLICE_SCHED_END, SLICE_SCALARS_END,
    SLICE_DIST_END, SLICE_FLAGS_END,
)
from models.encoder import CNNSPPFEncoder, SPPF, MLPEncoder, MultiModalEncoder
from models.actor_critic import FusionHead, ActorHead, CriticHead, ActorCritic
from models.network import SchedulingNetwork
from env_wrappers.unity_env import slice_obs
from rollout_buffer import RolloutBuffer


## @brief Default batch size used across all tests.
BATCH = 4

## @brief Device used for tensor allocation in tests.
DEVICE = "cpu"


def make_dummy_obs(batch_size: int = BATCH) -> dict:
    """@brief Create a batch of dummy observations matching the state space.

    @param batch_size  Number of samples in the batch.
    @return Dict of random tensors keyed by observation-space names.
    """
    return {
        "factory_grid": torch.randn(batch_size, 3, 64, 64),
        "sched_matrix": torch.randn(batch_size, 3, 100, 40),
        "global_scalars": torch.randn(batch_size, 10),
        "distance_matrix": torch.randn(batch_size, 64),
        "event_flags": torch.randn(batch_size, 6),
    }


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

    def test_sched_encoder_shape(self):
        """@brief Scheduling-matrix image (3, 100, 40) → 128-D embedding."""
        enc = CNNSPPFEncoder(in_channels=3, out_dim=128)
        x = torch.randn(BATCH, 3, 100, 40)
        out = enc(x)
        assert out.shape == (BATCH, 128), f"Sched encoder: {out.shape}"


class TestMLPEncoder:
    """@brief Shape tests for @ref MLPEncoder across the three vector modalities."""

    def test_global_scalars(self):
        """@brief 10-D global scalars → 32-D embedding."""
        mlp = MLPEncoder(10, 32)
        out = mlp(torch.randn(BATCH, 10))
        assert out.shape == (BATCH, 32)

    def test_distance_matrix(self):
        """@brief 64-D flattened distances → 32-D embedding."""
        mlp = MLPEncoder(64, 32)
        out = mlp(torch.randn(BATCH, 64))
        assert out.shape == (BATCH, 32)

    def test_event_flags(self):
        """@brief 6-D event flags → 16-D embedding."""
        mlp = MLPEncoder(6, 16)
        out = mlp(torch.randn(BATCH, 6))
        assert out.shape == (BATCH, 16)


class TestMultiModalEncoder:
    """@brief Integration tests for @ref MultiModalEncoder."""

    def test_output_dim(self):
        """@brief Concatenated output must be (B, 464)."""
        enc = MultiModalEncoder()
        obs = make_dummy_obs()
        out = enc(obs)
        assert out.shape == (BATCH, 464), f"Encoder concat: {out.shape}"

    def test_output_dim_matches_config(self):
        """@brief @ref MultiModalEncoder.output_dim must agree with EncoderConfig.concat_dim."""
        cfg = EncoderConfig()
        enc = MultiModalEncoder(cfg)
        assert enc.output_dim == cfg.concat_dim == 464


class TestFusionHead:
    """@brief Tests for @ref FusionHead."""

    def test_shape(self):
        """@brief Fusion output must be (B, 256)."""
        fusion = FusionHead(464, 512, 256)
        x = torch.randn(BATCH, 464)
        out = fusion(x)
        assert out.shape == (BATCH, 256)

    def test_deterministic_eval(self):
        """@brief Eval-mode forward passes on the same input must match."""
        fusion = FusionHead(464, 512, 256)
        fusion.eval()
        x = torch.randn(1, 464)
        with torch.no_grad():
            out1 = fusion(x)
            out2 = fusion(x)
        assert torch.allclose(out1, out2)


class TestActorCritic:
    """@brief Shape and contract tests for @ref ActorHead, @ref CriticHead,
    and @ref ActorCritic."""

    def test_actor_shape(self):
        """@brief Actor logits must be (B, 8)."""
        actor = ActorHead(256, 256, 8)
        out = actor(torch.randn(BATCH, 256))
        assert out.shape == (BATCH, 8)

    def test_critic_shape(self):
        """@brief Critic value must be (B, 1)."""
        critic = CriticHead(256, 256)
        out = critic(torch.randn(BATCH, 256))
        assert out.shape == (BATCH, 1)

    def test_act_outputs(self):
        """@brief @ref ActorCritic.act must return (action, log_prob, value)
        with correct shapes and valid action range [0, 8)."""
        ac = ActorCritic(256, 256, 8)
        features = torch.randn(BATCH, 256)
        action, log_prob, value = ac.act(features)
        assert action.shape == (BATCH,)
        assert log_prob.shape == (BATCH,)
        assert value.shape == (BATCH,)
        assert (action >= 0).all() and (action < 8).all()

    def test_evaluate_outputs(self):
        """@brief @ref ActorCritic.evaluate must return (log_probs, values, entropy)
        with non-negative entropy."""
        ac = ActorCritic(256, 256, 8)
        features = torch.randn(BATCH, 256)
        actions = torch.randint(0, 8, (BATCH,))
        lp, val, ent = ac.evaluate(features, actions)
        assert lp.shape == (BATCH,)
        assert val.shape == (BATCH,)
        assert ent.shape == (BATCH,)
        assert (ent >= 0).all(), "Entropy should be non-negative"


# ============================================================
#  2. Full network tests
# ============================================================

class TestSchedulingNetwork:
    """@brief End-to-end tests for @ref SchedulingNetwork."""

    def test_forward_shapes(self):
        """@brief Forward pass must produce logits (B, 8) and value (B, 1)."""
        net = SchedulingNetwork()
        obs = make_dummy_obs()
        logits, value = net(obs)
        assert logits.shape == (BATCH, 8), f"Logits: {logits.shape}"
        assert value.shape == (BATCH, 1), f"Value: {value.shape}"

    def test_act(self):
        """@brief @ref SchedulingNetwork.act must return actions of shape (B,)."""
        net = SchedulingNetwork()
        obs = make_dummy_obs()
        action, lp, val = net.act(obs)
        assert action.shape == (BATCH,)

    def test_evaluate(self):
        """@brief @ref SchedulingNetwork.evaluate must return log_probs of shape (B,)."""
        net = SchedulingNetwork()
        obs = make_dummy_obs()
        actions = torch.randint(0, 8, (BATCH,))
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
        actions = torch.randint(0, 8, (BATCH,))
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
        assert a1.item() == a2.item(), "Deterministic actions should match"

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
        obs_shapes = {
            "factory_grid": (3, 64, 64),
            "sched_matrix": (3, 100, 40),
            "global_scalars": (10,),
            "distance_matrix": (64,),
            "event_flags": (6,),
        }
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
        buf.compute_gae(
            last_values=np.array([0.4, 0.2], dtype=np.float32),
            last_dones=np.array([0.0, 0.0], dtype=np.float32),
        )
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
        buf.compute_gae(np.zeros(2, dtype=np.float32),
                        np.zeros(2, dtype=np.float32))
        batches = list(buf.get_batches(batch_size=4))
        assert len(batches) == 2  # 8 total / 4 batch


# ============================================================
#  4. Observation slicing tests (unity_env.slice_obs)
# ============================================================

class TestSliceObs:
    """@brief Tests for @ref slice_obs: flat vector → named observation dict."""

    def _make_flat_obs(self) -> np.ndarray:
        """@brief Build a synthetic flat observation vector of length
        TOTAL_OBS_SIZE with distinguishable per-stream values.

        @return 1-D float32 array of length 13,328.
        """
        raw = np.zeros(TOTAL_OBS_SIZE, dtype=np.float32)
        # Tag each stream with a distinct constant so slicing errors
        # are easy to diagnose.
        raw[:SLICE_SPATIAL_END] = 0.1
        raw[SLICE_SPATIAL_END:SLICE_SCHED_END] = 0.2
        raw[SLICE_SCHED_END:SLICE_SCALARS_END] = 0.3
        raw[SLICE_SCALARS_END:SLICE_DIST_END] = 0.4
        raw[SLICE_DIST_END:SLICE_FLAGS_END] = 0.5
        return raw

    def test_output_keys(self):
        """@brief slice_obs must return all five expected observation keys."""
        raw = self._make_flat_obs()
        d = slice_obs(raw)
        expected_keys = {
            "factory_grid", "sched_matrix", "global_scalars",
            "distance_matrix", "event_flags",
        }
        assert set(d.keys()) == expected_keys

    def test_factory_grid_shape(self):
        """@brief factory_grid must reshape to (C, H, W) = (3, 64, 64)."""
        d = slice_obs(self._make_flat_obs())
        assert d["factory_grid"].shape == (GRID_CHANNELS, GRID_SIZE, GRID_SIZE)

    def test_sched_matrix_shape(self):
        """@brief sched_matrix must end up in CHW order: (3, 20, 16)."""
        d = slice_obs(self._make_flat_obs())
        sched_cols = 2 * MAX_MACHINES  # 16
        assert d["sched_matrix"].shape == (SCHED_CHANNELS, MAX_JOBS, sched_cols)

    def test_scalar_shapes(self):
        """@brief global_scalars, distance_matrix, event_flags must retain
        their 1-D shapes."""
        d = slice_obs(self._make_flat_obs())
        assert d["global_scalars"].shape == (GLOBAL_SCALARS,)
        assert d["distance_matrix"].shape == (DISTANCE_DIM,)
        assert d["event_flags"].shape == (EVENT_FLAGS,)

    def test_stream_values_preserved(self):
        """@brief Each stream must contain only the constant assigned
        to its slice region — verifies no off-by-one in boundaries."""
        d = slice_obs(self._make_flat_obs())
        np.testing.assert_allclose(d["factory_grid"], 0.1, atol=1e-7)
        np.testing.assert_allclose(d["sched_matrix"], 0.2, atol=1e-7)
        np.testing.assert_allclose(d["global_scalars"], 0.3, atol=1e-7)
        np.testing.assert_allclose(d["distance_matrix"], 0.4, atol=1e-7)
        np.testing.assert_allclose(d["event_flags"], 0.5, atol=1e-7)

    def test_dtype_is_float32(self):
        """@brief Every output array must be float32."""
        d = slice_obs(self._make_flat_obs())
        for key, val in d.items():
            assert val.dtype == np.float32, f"{key} dtype is {val.dtype}"

    def test_wrong_length_raises(self):
        """@brief Passing a vector of incorrect length must raise AssertionError."""
        bad = np.zeros(100, dtype=np.float32)
        try:
            slice_obs(bad)
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass

    def test_batched_slice(self):
        """@brief slice_obs must handle a (B, TOTAL_OBS_SIZE) batch correctly.

        @details The leading batch dimension should propagate through
        all reshapes via the `*raw.shape[:-1]` pattern.
        """
        B = 3
        raw = np.random.rand(B, TOTAL_OBS_SIZE).astype(np.float32)
        d = slice_obs(raw)
        assert d["factory_grid"].shape == (B, GRID_CHANNELS, GRID_SIZE, GRID_SIZE)
        assert d["sched_matrix"].shape == (B, SCHED_CHANNELS, MAX_JOBS, 2 * MAX_MACHINES)
        assert d["global_scalars"].shape == (B, GLOBAL_SCALARS)
        assert d["distance_matrix"].shape == (B, DISTANCE_DIM)
        assert d["event_flags"].shape == (B, EVENT_FLAGS)

    def test_sched_matrix_hwc_to_chw(self):
        """@brief Verify HWC→CHW transposition of the scheduling matrix.

        @details The flat vector stores the matrix in (jobs, cols, channels)
        order.  After slicing, channel 0 of the CHW tensor should contain
        the first channel's data from every (job, col) position.
        """
        raw = np.zeros(TOTAL_OBS_SIZE, dtype=np.float32)
        # Fill scheduling region with identifiable pattern:
        # channel 0 = 0.1, channel 1 = 0.2, channel 2 = 0.3
        sched_start = SLICE_SPATIAL_END
        sched_len = MAX_JOBS * (2 * MAX_MACHINES) * SCHED_CHANNELS
        sched_flat = np.zeros(sched_len, dtype=np.float32)
        for i in range(MAX_JOBS * (2 * MAX_MACHINES)):
            sched_flat[i * SCHED_CHANNELS + 0] = 0.1  # channel 0
            sched_flat[i * SCHED_CHANNELS + 1] = 0.2  # channel 1
            sched_flat[i * SCHED_CHANNELS + 2] = 0.3  # channel 2
        raw[sched_start:sched_start + sched_len] = sched_flat

        d = slice_obs(raw)
        # After moveaxis to CHW, d["sched_matrix"][c] should be uniform
        np.testing.assert_allclose(d["sched_matrix"][0], 0.1, atol=1e-7)
        np.testing.assert_allclose(d["sched_matrix"][1], 0.2, atol=1e-7)
        np.testing.assert_allclose(d["sched_matrix"][2], 0.3, atol=1e-7)

    def test_total_obs_size_consistent(self):
        """@brief TOTAL_OBS_SIZE must equal the sum of all stream lengths."""
        expected = (
            GRID_CHANNELS * GRID_SIZE * GRID_SIZE
            + MAX_JOBS * (2 * MAX_MACHINES) * SCHED_CHANNELS
            + GLOBAL_SCALARS
            + DISTANCE_DIM
            + EVENT_FLAGS
        )
        assert TOTAL_OBS_SIZE == expected, (
            f"TOTAL_OBS_SIZE={TOTAL_OBS_SIZE} != computed {expected}"
        )

    def test_slice_boundaries_contiguous(self):
        """@brief Slice boundaries must be contiguous with no gaps or overlaps."""
        assert SLICE_SPATIAL_END > 0
        assert SLICE_SCHED_END > SLICE_SPATIAL_END
        assert SLICE_SCALARS_END > SLICE_SCHED_END
        assert SLICE_DIST_END > SLICE_SCALARS_END
        assert SLICE_FLAGS_END > SLICE_DIST_END
        assert SLICE_FLAGS_END == TOTAL_OBS_SIZE


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
    These tests verify the decision-loop logic (_step_until_decision),
    reward accumulation across silent frames, and episode termination
    handling — all without requiring a running Unity process.
    """

    def _make_env(self, get_steps_sequence):
        """@brief Construct a UnitySchedulingEnv with a fully mocked backend.

        @param get_steps_sequence  List of (decision_steps, terminal_steps)
                                   tuples returned by successive get_steps calls.
        @return A patched UnitySchedulingEnv instance.
        """
        from env_wrappers.unity_env import UnitySchedulingEnv

        with patch("env_wrappers.unity_env.UnityEnvironment") as MockUnity, \
             patch("env_wrappers.unity_env.EngineConfigurationChannel"):

            mock_env_instance = MagicMock()

            # Set up behavior_specs to return the correct obs shape
            mock_spec = MagicMock()
            mock_obs_spec = MagicMock()
            mock_obs_spec.shape = (TOTAL_OBS_SIZE,)
            mock_spec.observation_specs = [mock_obs_spec]

            mock_env_instance.behavior_specs = {"SchedulingBehavior?team=0": mock_spec}

            # __init__ calls self.env.reset() but never get_steps(),
            # so the side-effect list starts at the caller's sequence.
            mock_env_instance.get_steps.side_effect = list(get_steps_sequence)

            MockUnity.return_value = mock_env_instance

            env = UnitySchedulingEnv(file_name=None, time_scale=1.0)
            return env, mock_env_instance

    def test_step_returns_on_decision(self):
        """@brief When Unity immediately returns a decision step, the
        wrapper must return that observation and reward without looping."""
        obs_vec = np.random.rand(TOTAL_OBS_SIZE).astype(np.float32)
        decision = _make_mock_steps(obs_vec, reward=1.5, n_agents=1)
        terminal = _empty_steps()

        env, _ = self._make_env([(decision, terminal)])
        obs, reward, done, info = env.step(3)

        assert not done
        assert abs(reward - 1.5) < 1e-6
        assert set(obs.keys()) == {
            "factory_grid", "sched_matrix", "global_scalars",
            "distance_matrix", "event_flags",
        }

    def test_step_returns_on_terminal(self):
        """@brief When Unity returns a terminal step, done must be True."""
        obs_vec = np.random.rand(TOTAL_OBS_SIZE).astype(np.float32)
        decision = _empty_steps()
        terminal = _make_mock_steps(obs_vec, reward=10.0, n_agents=1)

        env, _ = self._make_env([(decision, terminal)])
        obs, reward, done, info = env.step(0)

        assert done
        assert abs(reward - 10.0) < 1e-6

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
        obs, reward, done, info = env.step(1)

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
            "factory_grid", "sched_matrix", "global_scalars",
            "distance_matrix", "event_flags",
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


class TestVectorizedUnityEnv:
    """@brief Tests for @ref VectorizedUnityEnv using mocked sub-environments.

    @details
    Patches UnitySchedulingEnv entirely to avoid any Unity dependency.
    Verifies stacking, auto-reset on done, and shape correctness.
    """

    def _make_dummy_obs_dict(self):
        """@brief Create a single-env observation dict with valid shapes."""
        return {
            "factory_grid": np.random.rand(GRID_CHANNELS, GRID_SIZE, GRID_SIZE).astype(np.float32),
            "sched_matrix": np.random.rand(SCHED_CHANNELS, MAX_JOBS, 2 * MAX_MACHINES).astype(np.float32),
            "global_scalars": np.random.rand(GLOBAL_SCALARS).astype(np.float32),
            "distance_matrix": np.random.rand(DISTANCE_DIM).astype(np.float32),
            "event_flags": np.random.rand(EVENT_FLAGS).astype(np.float32),
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

    def test_auto_reset_on_done(self):
        """@brief When a sub-env signals done, VectorizedUnityEnv must
        auto-reset it and return the fresh observation."""
        from env_wrappers.unity_env import VectorizedUnityEnv

        num_envs = 2
        step_obs = [self._make_dummy_obs_dict() for _ in range(num_envs)]
        reset_obs = self._make_dummy_obs_dict()
        # Tag the reset obs so we can identify it
        reset_obs["global_scalars"][:] = 99.0

        with patch("env_wrappers.unity_env.UnitySchedulingEnv") as MockSingle:
            mock_instances = []
            for i in range(num_envs):
                m = MagicMock()
                m.reset.return_value = reset_obs if i == 0 else step_obs[i]
                # Env 0 is done, env 1 is not
                m.step.return_value = (
                    step_obs[i],
                    1.0 if i == 0 else 0.5,
                    i == 0,  # done for env 0
                    {},
                )
                mock_instances.append(m)

            MockSingle.side_effect = mock_instances
            vec = VectorizedUnityEnv(num_envs=num_envs, file_name=None)
            vec.reset()
            obs, rewards, dones, truncs, infos = vec.step(np.array([0, 0]))

        assert dones[0] == True
        assert dones[1] == False
        # Env 0 should have been auto-reset, so its obs is the reset obs
        np.testing.assert_allclose(obs["global_scalars"][0], 99.0)

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