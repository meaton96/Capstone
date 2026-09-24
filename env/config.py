"""
@file config.py
@brief Configuration for DRL Job-Shop Scheduling Architecture.

@details
Observation schema v2 (2026-09-24). All dimensions are synced to the C# ObservationBuilder constants:
  SpatialGridSize = 64,  SpatialChannels = 3
  MaxMachines = 100, MachineFeatures = 16     (machine table, one row per machine, zero-padded)
  MaxJobs     = 64,  JobFeatures     = 17     (job table over ACTIVE jobs, decision-relevant first)
  GlobalScalarLength = 16
  EventFlagLength    = 6

Total flat observation from ML-Agents: 14,998 floats.

v1 (13,328 floats) had an 8-machine x first-20-jobs scheduling matrix and an 8x8 distance matrix: on the
15-machine floor it dropped machines 8-14 and went blank once the first 20 jobs had exited. v1 checkpoints
do not load into v2 networks.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────
#  Observation-stream dimensions  (must mirror ObservationBuilder.cs)
# ─────────────────────────────────────────────────────────────────────

## @brief Version of the observation's column meanings; bump together with ObservationBuilder.cs whenever a
##        feature is added, removed or redefined.
OBS_SCHEMA_VERSION = 2

GRID_SIZE        = 64
GRID_CHANNELS    = 3
MAX_MACHINES     = 100
MACHINE_FEATURES = 16
MAX_JOBS         = 64
JOB_FEATURES     = 17
GLOBAL_SCALARS   = 16
EVENT_FLAGS      = 6

## @brief Column meanings (see ObservationBuilder.BuildMachineTable / BuildJobTable). Column 0 of both
##        tables is the "present" mask; the candidate column marks rows that are options in the
##        current decision.
MACHINE_PRESENT_COL   = 0
MACHINE_CANDIDATE_COL = 13
JOB_PRESENT_COL       = 0
JOB_CANDIDATE_COL     = 15

## @brief Lengths of each stream inside the flat vector.
SPATIAL_LEN       = GRID_CHANNELS * GRID_SIZE * GRID_SIZE    # 12 288
MACHINE_TABLE_LEN = MAX_MACHINES * MACHINE_FEATURES          #  1 600
JOB_TABLE_LEN     = MAX_JOBS * JOB_FEATURES                  #  1 088
TOTAL_OBS_SIZE    = SPATIAL_LEN + MACHINE_TABLE_LEN + JOB_TABLE_LEN + GLOBAL_SCALARS + EVENT_FLAGS  # 14 998

## @brief Slice boundaries inside the flat observation vector (C# FlattenStreams order).
SLICE_SPATIAL_END  = SPATIAL_LEN
SLICE_MACHINES_END = SLICE_SPATIAL_END + MACHINE_TABLE_LEN
SLICE_JOBS_END     = SLICE_MACHINES_END + JOB_TABLE_LEN
SLICE_SCALARS_END  = SLICE_JOBS_END + GLOBAL_SCALARS
SLICE_FLAGS_END    = SLICE_SCALARS_END + EVENT_FLAGS          # == TOTAL_OBS_SIZE

## @brief What a trained network depends on: column meanings and per-row widths. The row caps
##        (MAX_MACHINES, MAX_JOBS) are deliberately NOT part of it -- the set encoders' weights do not
##        depend on the row count, so raising a cap (and rebuilding the player) keeps checkpoints loadable.
OBS_LAYOUT = {
    "schema_version": OBS_SCHEMA_VERSION,
    "grid": [GRID_CHANNELS, GRID_SIZE, GRID_SIZE],
    "machine_features": MACHINE_FEATURES,
    "job_features": JOB_FEATURES,
    "global_scalars": GLOBAL_SCALARS,
    "event_flags": EVENT_FLAGS,
}

## @brief Per-key observation shapes (no batch dim) — the single source for train.py, the
##        placeholder env and tests.
OBS_SHAPES = {
    "factory_grid":   (GRID_CHANNELS, GRID_SIZE, GRID_SIZE),
    "machine_table":  (MAX_MACHINES, MACHINE_FEATURES),
    "job_table":      (MAX_JOBS, JOB_FEATURES),
    "global_scalars": (GLOBAL_SCALARS,),
    "event_flags":    (EVENT_FLAGS,),
}


@dataclass
class EnvConfig:
    """@brief Factory environment parameters (synced to C# ObservationBuilder)."""

    grid_size: int = GRID_SIZE
    grid_channels: int = GRID_CHANNELS
    num_machines_range: Tuple[int, int] = (4, 20)
    num_jobs_range: Tuple[int, int] = (5, MAX_JOBS)
    num_global_scalars: int = GLOBAL_SCALARS
    num_event_flags: int = EVENT_FLAGS
    max_machines: int = MAX_MACHINES
    machine_features: int = MACHINE_FEATURES
    max_jobs: int = MAX_JOBS
    job_features: int = JOB_FEATURES


@dataclass
class EncoderConfig:
    """@brief Encoder output dimensions.

    @details The machine and job tables go through set encoders (shared per-row MLP, masked mean + max
    pooling over all present rows and over decision-candidate rows), so the embedding does not depend
    on the floor size or on row order.
    """

    factory_cnn_out: int = 256
    machine_set_out: int = 128
    job_set_out: int = 128
    set_row_hidden: int = 64
    global_mlp_out: int = 32
    event_embed_out: int = 16
    sppf_pool_sizes: List[int] = field(default_factory=lambda: [5, 9, 13])

    @property
    def concat_dim(self) -> int:
        return (
            self.factory_cnn_out
            + self.machine_set_out
            + self.job_set_out
            + self.global_mlp_out
            + self.event_embed_out
        )


@dataclass
class FusionConfig:
    """@brief Fusion head parameters (input is EncoderConfig.concat_dim, 560 by default)."""

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
    "FIFO-SRWT",
]