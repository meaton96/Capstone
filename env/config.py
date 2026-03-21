"""
Configuration for DRL Job-Shop Scheduling Architecture.
All dimensions match the thesis architecture diagram.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class EnvConfig:
    """Factory environment parameters."""
    grid_size: int = 64                          # NxN spatial grid
    grid_channels: int = 3                       # Machine, Job, AGV layers
    num_machines_range: Tuple[int, int] = (15, 20)
    num_jobs_range: Tuple[int, int] = (50, 100)
    ops_per_machine: int = 20                    # O_per_mach = M (max)
    processing_time_range: Tuple[int, int] = (1, 99)
    num_global_scalars: int = 10
    distance_matrix_dim: int = 64                # 8x8 flattened
    num_event_flags: int = 6
    max_machines: int = 20                       # for fixed-size tensors
    max_jobs: int = 100


@dataclass
class SchedulingMatrixConfig:
    """Scheduling matrix image dimensions: n × 2m × 3."""
    max_jobs: int = 100       # n (padded to max)
    max_cols: int = 40        # 2m where m=max_machines=20
    channels: int = 3         # machines, processing time, zeros


@dataclass
class EncoderConfig:
    """Encoder output dimensions from the diagram."""
    # CNN-SPPF (Factory Floor) -> 256-D
    factory_cnn_out: int = 256
    # CNN-SPPF (Scheduling Matrix) -> 128-D
    sched_cnn_out: int = 128
    # Global Context MLP -> 32-D
    global_mlp_out: int = 32
    # Distance Embed MLP -> 32-D
    distance_mlp_out: int = 32
    # Event Flag Embed -> 16-D
    event_embed_out: int = 16

    # SPPF pyramid levels (Spatial Pyramid Pooling - Fast)
    sppf_pool_sizes: List[int] = field(default_factory=lambda: [5, 9, 13])

    @property
    def concat_dim(self) -> int:
        """Total concatenated dimension: 256+128+32+32+16 = 464."""
        return (
            self.factory_cnn_out
            + self.sched_cnn_out
            + self.global_mlp_out
            + self.distance_mlp_out
            + self.event_embed_out
        )


@dataclass
class FusionConfig:
    """Fusion head + domain randomization."""
    input_dim: int = 464     # from encoder concat
    hidden_dim: int = 512
    output_dim: int = 256    # fusion output
    dropout_rate: float = 0.1
    noise_std: float = 0.01  # Gaussian noise for domain randomization


@dataclass
class ActorCriticConfig:
    """Actor-Critic head dimensions."""
    input_dim: int = 256     # from fusion head
    hidden_dim: int = 256
    num_actions: int = 8     # 8 composite PDR rules


@dataclass
class PPOConfig:
    """PPO training hyperparameters."""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_epochs: int = 4
    batch_size: int = 64
    rollout_length: int = 128
    num_envs: int = 8        # parallel environments
    total_timesteps: int = 1_000_000


# Composite PDR actions (Job rule - Machine rule)
PDR_ACTIONS = [
    "SPT-SMPT",
    "SPT-SRWT",
    "LPT-MMUR",
    "SRT-SRWT",
    "LRT-SMPT",
    "LRT-MMUR",
    "SRT-SMPT",
    "SDT-SRWT",
]
