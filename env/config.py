"""
@file config.py
@brief Configuration for DRL Job-Shop Scheduling Architecture.

@details
All dimensions are synced to the C# ObservationBuilder constants:
  SpatialGridSize = 64,  SpatialChannels = 3
  MaxJobs         = 20,  MaxMachines     = 8,  SchedChannels = 3
  GlobalScalarLength = 10
  DistanceLength     = 64  (8 x 8)
  EventFlagLength    = 6

Total flat observation from ML-Agents: 13,328 floats.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────
#  Observation-stream dimensions  (must mirror ObservationBuilder.cs)
# ─────────────────────────────────────────────────────────────────────

## @brief C#-side constants reproduced here so every Python file can
##        import a single source of truth.
GRID_SIZE       = 64
GRID_CHANNELS   = 3
MAX_JOBS        = 20
MAX_MACHINES    = 8      # m — half-columns in the scheduling matrix
SCHED_CHANNELS  = 3
GLOBAL_SCALARS  = 10
DISTANCE_DIM    = MAX_MACHINES * MAX_MACHINES   # 64
EVENT_FLAGS     = 6

## @brief Byte-lengths of each stream inside the flat vector.
SPATIAL_LEN     = GRID_CHANNELS * GRID_SIZE * GRID_SIZE   # 12 288
SCHED_LEN       = MAX_JOBS * (2 * MAX_MACHINES) * SCHED_CHANNELS  # 960
TOTAL_OBS_SIZE  = SPATIAL_LEN + SCHED_LEN + GLOBAL_SCALARS + DISTANCE_DIM + EVENT_FLAGS  # 13 328

## @brief Slice boundaries inside the flat observation vector.
SLICE_SPATIAL_END   = SPATIAL_LEN
SLICE_SCHED_END     = SLICE_SPATIAL_END + SCHED_LEN
SLICE_SCALARS_END   = SLICE_SCHED_END  + GLOBAL_SCALARS
SLICE_DIST_END      = SLICE_SCALARS_END + DISTANCE_DIM
SLICE_FLAGS_END     = SLICE_DIST_END    + EVENT_FLAGS  # == TOTAL_OBS_SIZE


@dataclass
class EnvConfig:
    """@brief Factory environment parameters (synced to C# ObservationBuilder)."""

    grid_size: int = GRID_SIZE
    grid_channels: int = GRID_CHANNELS
    num_machines_range: Tuple[int, int] = (8, MAX_MACHINES)
    num_jobs_range: Tuple[int, int] = (10, MAX_JOBS)
    ops_per_machine: int = MAX_MACHINES
    processing_time_range: Tuple[int, int] = (1, 99)
    num_global_scalars: int = GLOBAL_SCALARS
    distance_matrix_dim: int = DISTANCE_DIM
    num_event_flags: int = EVENT_FLAGS
    max_machines: int = MAX_MACHINES
    max_jobs: int = MAX_JOBS


@dataclass
class SchedulingMatrixConfig:
    """@brief Scheduling-matrix image dimensions: n × 2m × 3.

    @details
    The C# side lays out the matrix in (jobs, cols, channels) order
    — i.e. HWC.  Python reshapes to (channels, jobs, cols) = CHW
    for the CNN encoder.
    """

    max_jobs: int = MAX_JOBS           # 20  (rows)
    max_cols: int = 2 * MAX_MACHINES   # 16  (columns)
    channels: int = SCHED_CHANNELS     # 3


@dataclass
class EncoderConfig:
    """@brief Encoder output dimensions from the architecture diagram."""

    factory_cnn_out: int = 256
    sched_cnn_out: int = 128
    global_mlp_out: int = 32
    distance_mlp_out: int = 32
    event_embed_out: int = 16
    sppf_pool_sizes: List[int] = field(default_factory=lambda: [5, 9, 13])

    @property
    def concat_dim(self) -> int:
        return (
            self.factory_cnn_out
            + self.sched_cnn_out
            + self.global_mlp_out
            + self.distance_mlp_out
            + self.event_embed_out
        )


@dataclass
class FusionConfig:
    """@brief Fusion head parameters."""

    input_dim: int = 464
    hidden_dim: int = 512
    output_dim: int = 256


@dataclass
class ActorCriticConfig:
    """@brief Actor-Critic head dimensions."""

    input_dim: int = 256
    hidden_dim: int = 256
    num_actions: int = 8


@dataclass
class PPOConfig:
    """@brief PPO training hyperparameters."""

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
    num_envs: int = 8
    total_timesteps: int = 1_000_000


PDR_ACTIONS = [
    "SPT-SMPT",
    "SPT-SRWT",
    "LPT-MMUR",
    "LPT-SMPT",
    "SRT-SRWT",
    "SRT-SMPT",
    "LRT-MMUR",
    "SDT-SRWT",
]