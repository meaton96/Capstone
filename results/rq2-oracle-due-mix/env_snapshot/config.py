"""
@file config.py
@brief Configuration for DRL Job-Shop Scheduling Architecture.

@details
Observation schema v3 (2026-10-03; v2 2026-09-24). All dimensions are synced to the C# ObservationBuilder constants:
  SpatialGridSize = 64,  SpatialChannels = 3
  MaxMachines = 105,  MachineFeatures = 16    (machine table, one row per machine, zero-padded)
  MaxJobs     = 1792, JobFeatures     = 21    (job table over ACTIVE jobs, decision-relevant first)
  GlobalScalarLength = 18
  EventFlagLength    = 6

Row caps are chosen per player launch (2026-09-27, see obs_row_caps / env_wrappers/unity_env.py): machine rows =
the largest floor the run will see, job rows = JOB_ROWS_PER_15_MACHINES per 15 machines. A 15-machine run is
12,312 + 15*16 + 256*21 = 17,928 floats; the defaults below (7 tiles, 105 machines / 1,792 jobs) give 51,624.
Row-cap history: MaxJobs 64 -> 256 on 2026-09-25; defaults 105 / 1792 and per-launch caps on 2026-09-27. The
caps never affect checkpoints (no weight depends on the row count).

v1 (13,328 floats) had an 8-machine x first-20-jobs scheduling matrix and an 8x8 distance matrix: on the
15-machine floor it dropped machines 8-14 and went blank once the first 20 jobs had exited. v1 checkpoints
do not load into v2 networks. v3 adds due dates for the tardiness objective: job columns 17-20 (has due date,
slack = due - now - remaining work, late, due - now; signed squashes) and global scalars 16-17 (share of due-dated
active jobs late / with negative slack). v2 checkpoints do not load into v3 networks.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
#  Observation-stream dimensions  (must mirror ObservationBuilder.cs)
# ─────────────────────────────────────────────────────────────────────

## @brief Version of the observation's column meanings; bump together with ObservationBuilder.cs whenever a
##        feature is added, removed or redefined.
OBS_SCHEMA_VERSION = 3

GRID_SIZE        = 64
GRID_CHANNELS    = 3
## @brief Default row caps (ObservationBuilder.DefaultMaxMachines / DefaultMaxJobs): what a player launched
##        without -obsmaxmachines / -obsmaxjobs uses. Sized for 7 tiles of 15 machines.
MAX_MACHINES     = 105
MACHINE_FEATURES = 16
MAX_JOBS         = 1792
## @brief Job rows per 15 machines when caps are chosen per launch (randomized-family WIP peaked at 165 on
##        15 machines; the 7-tile compound pilot at 476 = 68 per tile).
JOB_ROWS_PER_15_MACHINES = 256
JOB_FEATURES     = 21
GLOBAL_SCALARS   = 18
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
MACHINE_TABLE_LEN = MAX_MACHINES * MACHINE_FEATURES          #  1 680
JOB_TABLE_LEN     = MAX_JOBS * JOB_FEATURES                  # 30 464
TOTAL_OBS_SIZE    = SPATIAL_LEN + MACHINE_TABLE_LEN + JOB_TABLE_LEN + GLOBAL_SCALARS + EVENT_FLAGS  # 44 454

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

# ─────────────────────────────────────────────────────────────────────
#  Action space  (must mirror DispatchingEngine.JobHead / MachineHead)
# ─────────────────────────────────────────────────────────────────────

## @brief Version of the action space; bump whenever a head's rules or order change. v1 was one 8-way branch
##        of composite rules (SPT-SMPT ... FIFO-SRWT); v2 (2026-09-26) is two branches, job head x machine
##        head, chosen from the gen_rules0926 sweep (docs/features/DECISION_POINTS.md section 7). v3 (2026-10-03)
##        appends the due-date job rules EDD, MDD, ATC (rq2-twin-due; they need jobs with due dates, else they
##        rank every job as due at +infinity and keep the first candidate). v4 (2026-10-03) is the H15 head of the
##        tardiness objective (rq2-twin-due, rq2-oracle-due): PTWINQ and FIFO dropped, SRT first.
ACTION_SCHEMA_VERSION = 4

## @brief Branch 0: which job (dispatch: from the machine's queue; routing: from the pool).
JOB_HEAD_RULES = ["SRT", "SPT", "MDD", "EDD", "ATC"]
## @brief Branch 1: which machine a routed job goes to.
MACHINE_HEAD_RULES = ["ECT", "TECT", "SRWT"]
ACTION_BRANCHES = (len(JOB_HEAD_RULES), len(MACHINE_HEAD_RULES))

## @brief Unity masks a head that cannot change a decision down to its action 0 (DispatchingEngine.HeadsThatMatter).
##        The wrapper passes the masks as obs["action_mask"]: 1 = enabled, branches concatenated.
ACTION_MASK_LEN = sum(ACTION_BRANCHES)

## @brief What a trained actor depends on; checkpoints record it and train/evaluate refuse a mismatch.
ACTION_LAYOUT = {
    "schema_version": ACTION_SCHEMA_VERSION,
    "job_head": list(JOB_HEAD_RULES),
    "machine_head": list(MACHINE_HEAD_RULES),
}


## @brief Per-key observation shapes (no batch dim) — the single source for train.py, the
##        placeholder env and tests.
def obs_shapes(max_machines: int = MAX_MACHINES, max_jobs: int = MAX_JOBS) -> dict:
    """@brief Per-key observation shapes (no batch dim) for the given row caps."""
    return {
        "factory_grid":   (GRID_CHANNELS, GRID_SIZE, GRID_SIZE),
        "machine_table":  (max_machines, MACHINE_FEATURES),
        "job_table":      (max_jobs, JOB_FEATURES),
        "global_scalars": (GLOBAL_SCALARS,),
        "event_flags":    (EVENT_FLAGS,),
        "action_mask":    (ACTION_MASK_LEN,),   # not encoded: masks the actor's heads (see ACTION_MASK_LEN)
    }


## @brief Shapes at the default caps (the placeholder env and tests use these).
OBS_SHAPES = obs_shapes()


def obs_total_size(max_machines: int, max_jobs: int) -> int:
    """@brief Flat observation length the player sends for the given row caps."""
    return SPATIAL_LEN + max_machines * MACHINE_FEATURES + max_jobs * JOB_FEATURES + GLOBAL_SCALARS + EVENT_FLAGS


def obs_row_caps(n_machines: int) -> Tuple[int, int]:
    """@brief Smallest row caps for floors of up to @p n_machines machines: (machine rows, job rows).

    @details Job rows scale with the floor, JOB_ROWS_PER_15_MACHINES per started block of 15 machines,
    so 15 machines -> (15, 256) and 7 tiles of 15 -> (105, 1792), the defaults.
    """
    if n_machines <= 0:
        raise ValueError(f"n_machines must be positive, got {n_machines}")
    return n_machines, JOB_ROWS_PER_15_MACHINES * -(-n_machines // 15)


@dataclass
class EnvConfig:
    """@brief Factory environment parameters (synced to C# ObservationBuilder)."""

    grid_size: int = GRID_SIZE
    grid_channels: int = GRID_CHANNELS
    num_machines_range: Tuple[int, int] = (4, 20)
    num_jobs_range: Tuple[int, int] = (5, 256)   # placeholder env only; was (5, MAX_JOBS) when MAX_JOBS was 256
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
    ## @brief Normalization in the grid CNN: "group" (GroupNorm, 8 groups; the default since 2026-10-01) or
    ##        "batch" (BatchNorm, every checkpoint before that). BatchNorm normalized rollouts (eval mode, running
    ##        statistics) and PPO updates (train mode, minibatch statistics) differently, so the PPO ratio was not
    ##        1 before any weight changed; GroupNorm behaves the same in both. "batch" is kept only to load old
    ##        checkpoints, and SchedulingNetwork pins it to running statistics in both modes.
    grid_norm: str = "group"

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
    action_branches: Tuple[int, ...] = ACTION_BRANCHES


@dataclass
class PPOConfig:
    """@brief PPO training hyperparameters."""

    lr: float = 3e-4
    ## @brief Per-decision discount; used only when discount_horizon_s is None.
    gamma: float = 0.99
    ## @brief Discount over simulated time (SMDP form): a transition spanning Δτ seconds is discounted by
    ##        gamma_s ** Δτ, with gamma_s = 1 - 1/discount_horizon_s. 3000 s covers a job's whole stay
    ##        (mean flow ~1,500 s on rnd_load); the old per-decision 0.99 was ~1,370 s and varied 2x
    ##        between instances. None = per-decision gamma (all runs before 2026-10-01).
    discount_horizon_s: Optional[float] = 3000.0
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    ## @brief If set, the entropy coefficient decays linearly from entropy_coef to this value over
    ##        total_timesteps (by absolute global step, so a resumed run continues the schedule).
    entropy_coef_final: Optional[float] = None
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_epochs: int = 4
    batch_size: int = 64
    rollout_length: int = 128
    num_envs: int = 8
    total_timesteps: int = 1_000_000

    @property
    def gamma_per_second(self) -> Optional[float]:
        """@brief gamma_s for discounting over simulated time, or None for per-decision gamma."""
        if self.discount_horizon_s is None:
            return None
        if self.discount_horizon_s <= 1.0:
            raise ValueError(f"discount_horizon_s must be > 1 s, got {self.discount_horizon_s}")
        return 1.0 - 1.0 / self.discount_horizon_s


## @brief Fixed-rule baselines reachable through the RL action space: every (job head, machine head) pair,
##        named JOB-MACHINE, job-major. Rules outside the heads (e.g. SPT-SMPT) run through the batch runner.
PDR_ACTIONS = [f"{job}-{machine}" for job in JOB_HEAD_RULES for machine in MACHINE_HEAD_RULES]


def pdr_action(name: str) -> Tuple[int, int]:
    """@brief (job head, machine head) of a PDR_ACTIONS name."""
    job, machine = name.split("-")
    return JOB_HEAD_RULES.index(job), MACHINE_HEAD_RULES.index(machine)
