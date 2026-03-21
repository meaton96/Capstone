"""
Test suite for the DRL Scheduling Architecture.

Tests cover:
  1. Tensor shape correctness through every layer
  2. Forward/backward pass (gradient flow)
  3. Environment interface compliance
  4. Rollout buffer GAE computation
  5. PPO loss computation
  6. Checkpoint save/load round-trip
  7. Deterministic vs stochastic action selection

Run:  python -m pytest tests/test_architecture.py -v
  or: python tests/test_architecture.py
"""

import sys
import os
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EncoderConfig, FusionConfig, ActorCriticConfig, PPOConfig
from models.encoder import CNNSPPFEncoder, SPPF, MLPEncoder, MultiModalEncoder
from models.actor_critic import FusionHead, ActorHead, CriticHead, ActorCritic, DomainRandomization
from models.network import SchedulingNetwork
from env.placeholder_env import PlaceholderSchedulingEnv, VectorizedPlaceholderEnv
from rollout_buffer import RolloutBuffer


BATCH = 4
DEVICE = "cpu"


def make_dummy_obs(batch_size: int = BATCH) -> dict:
    """Create a batch of dummy observations matching the state space."""
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
    def test_output_shape(self):
        sppf = SPPF(128, 128)
        x = torch.randn(BATCH, 128, 16, 16)
        out = sppf(x)
        assert out.shape == (BATCH, 128, 16, 16), f"SPPF shape: {out.shape}"

    def test_preserves_spatial(self):
        sppf = SPPF(64, 64)
        x = torch.randn(2, 64, 8, 8)
        out = sppf(x)
        assert out.shape[2:] == x.shape[2:], "SPPF should preserve spatial dims"


class TestCNNSPPFEncoder:
    def test_factory_encoder_shape(self):
        enc = CNNSPPFEncoder(in_channels=3, out_dim=256)
        x = torch.randn(BATCH, 3, 64, 64)
        out = enc(x)
        assert out.shape == (BATCH, 256), f"Factory encoder: {out.shape}"

    def test_sched_encoder_shape(self):
        enc = CNNSPPFEncoder(in_channels=3, out_dim=128)
        x = torch.randn(BATCH, 3, 100, 40)
        out = enc(x)
        assert out.shape == (BATCH, 128), f"Sched encoder: {out.shape}"


class TestMLPEncoder:
    def test_global_scalars(self):
        mlp = MLPEncoder(10, 32)
        out = mlp(torch.randn(BATCH, 10))
        assert out.shape == (BATCH, 32)

    def test_distance_matrix(self):
        mlp = MLPEncoder(64, 32)
        out = mlp(torch.randn(BATCH, 64))
        assert out.shape == (BATCH, 32)

    def test_event_flags(self):
        mlp = MLPEncoder(6, 16)
        out = mlp(torch.randn(BATCH, 6))
        assert out.shape == (BATCH, 16)


class TestMultiModalEncoder:
    def test_output_dim(self):
        enc = MultiModalEncoder()
        obs = make_dummy_obs()
        out = enc(obs)
        assert out.shape == (BATCH, 464), f"Encoder concat: {out.shape}"

    def test_output_dim_matches_config(self):
        cfg = EncoderConfig()
        enc = MultiModalEncoder(cfg)
        assert enc.output_dim == cfg.concat_dim == 464


class TestFusionHead:
    def test_shape(self):
        fusion = FusionHead(464, 512, 256)
        x = torch.randn(BATCH, 464)
        out = fusion(x)
        assert out.shape == (BATCH, 256)

    def test_domain_rand_training_vs_eval(self):
        dr = DomainRandomization(dropout_rate=0.5, noise_std=0.1)
        x = torch.ones(100, 64)
        dr.train()
        out_train = dr(x)
        # In training, some values should be zeroed (dropout)
        assert (out_train == 0).any(), "Dropout should zero some values"

        dr.eval()
        out_eval = dr(x)
        # In eval, no dropout or noise
        assert torch.allclose(out_eval, x), "Eval should be identity"


class TestActorCritic:
    def test_actor_shape(self):
        actor = ActorHead(256, 256, 8)
        out = actor(torch.randn(BATCH, 256))
        assert out.shape == (BATCH, 8)

    def test_critic_shape(self):
        critic = CriticHead(256, 256)
        out = critic(torch.randn(BATCH, 256))
        assert out.shape == (BATCH, 1)

    def test_act_outputs(self):
        ac = ActorCritic(256, 256, 8)
        features = torch.randn(BATCH, 256)
        action, log_prob, value = ac.act(features)
        assert action.shape == (BATCH,)
        assert log_prob.shape == (BATCH,)
        assert value.shape == (BATCH,)
        assert (action >= 0).all() and (action < 8).all()

    def test_evaluate_outputs(self):
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
    def test_forward_shapes(self):
        net = SchedulingNetwork()
        obs = make_dummy_obs()
        logits, value = net(obs)
        assert logits.shape == (BATCH, 8), f"Logits: {logits.shape}"
        assert value.shape == (BATCH, 1), f"Value: {value.shape}"

    def test_act(self):
        net = SchedulingNetwork()
        obs = make_dummy_obs()
        action, lp, val = net.act(obs)
        assert action.shape == (BATCH,)

    def test_evaluate(self):
        net = SchedulingNetwork()
        obs = make_dummy_obs()
        actions = torch.randint(0, 8, (BATCH,))
        lp, val, ent = net.evaluate(obs, actions)
        assert lp.shape == (BATCH,)

    def test_gradient_flow(self):
        """Verify gradients flow through every parameter."""
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
        net = SchedulingNetwork()
        net.eval()
        obs = make_dummy_obs(1)
        a1, _, _ = net.act(obs, deterministic=True)
        a2, _, _ = net.act(obs, deterministic=True)
        assert a1.item() == a2.item(), "Deterministic actions should match"

    def test_param_summary(self):
        net = SchedulingNetwork()
        summary = net.get_param_summary()
        assert summary["total"] > 0
        part_sum = sum(v for k, v in summary.items() if k != "total")
        # Parts should roughly equal total (some params might be shared)
        assert abs(part_sum - summary["total"]) < 100

    def test_checkpoint_roundtrip(self):
        """Save and reload model, verify outputs match."""
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
#  3. Environment tests
# ============================================================

class TestPlaceholderEnv:
    def test_reset(self):
        env = PlaceholderSchedulingEnv(seed=42)
        obs, info = env.reset()
        assert set(obs.keys()) == {
            "factory_grid", "sched_matrix", "global_scalars",
            "distance_matrix", "event_flags",
        }
        assert obs["factory_grid"].shape == (3, 64, 64)
        assert obs["sched_matrix"].shape == (3, 100, 40)
        assert obs["global_scalars"].shape == (10,)
        assert obs["distance_matrix"].shape == (64,)
        assert obs["event_flags"].shape == (6,)

    def test_step(self):
        env = PlaceholderSchedulingEnv(seed=42)
        env.reset()
        obs, reward, term, trunc, info = env.step(0)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert "pdr_rule" in info

    def test_all_actions_valid(self):
        env = PlaceholderSchedulingEnv(seed=42)
        env.reset()
        for a in range(8):
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                env.reset()

    def test_episode_terminates(self):
        env = PlaceholderSchedulingEnv(max_steps=10, seed=42)
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, term, trunc, _ = env.step(0)
            done = term or trunc
            steps += 1
        assert steps <= 10

    def test_obs_ranges(self):
        """Observations should be in [0, 1] range."""
        env = PlaceholderSchedulingEnv(seed=42)
        obs, _ = env.reset()
        for key, val in obs.items():
            assert val.min() >= 0.0, f"{key} has negative values"
            assert val.max() <= 1.0, f"{key} exceeds 1.0"


class TestVectorizedEnv:
    def test_batched_obs(self):
        vec = VectorizedPlaceholderEnv(num_envs=3)
        obs, infos = vec.reset()
        assert obs["factory_grid"].shape == (3, 3, 64, 64)  # (num_envs, C, H, W)

    def test_step_shapes(self):
        vec = VectorizedPlaceholderEnv(num_envs=3)
        vec.reset()
        obs, rewards, terms, truncs, infos = vec.step([0, 1, 2])
        assert rewards.shape == (3,)
        assert terms.shape == (3,)


# ============================================================
#  4. Rollout buffer tests
# ============================================================

class TestRolloutBuffer:
    def test_gae_computation(self):
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
#  Run all tests
# ============================================================

def run_tests():
    """Simple test runner (works without pytest)."""
    import traceback

    test_classes = [
        TestSPPF, TestCNNSPPFEncoder, TestMLPEncoder,
        TestMultiModalEncoder, TestFusionHead, TestActorCritic,
        TestSchedulingNetwork, TestPlaceholderEnv, TestVectorizedEnv,
        TestRolloutBuffer,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in methods:
            total += 1
            test_id = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS  {test_id}")
            except Exception as e:
                failed += 1
                errors.append((test_id, traceback.format_exc()))
                print(f"  FAIL  {test_id}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for test_id, tb in errors:
            print(f"\n--- {test_id} ---")
            print(tb)
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
