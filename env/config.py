"""
@file config.py
@brief Configuration for DRL Job-Shop Scheduling Architecture.

@details
All dimensions match the thesis architecture diagram.  Each dataclass
groups a related set of hyperparameters and exposes sensible defaults
so the system can be instantiated with zero arguments.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class EnvConfig:
    """@brief Factory environment parameters.

    @details
    Defines the spatial grid geometry, job/machine ranges, and
    observation-vector dimensions used by both the placeholder and
    Unity-backed environments.
    """

    ## @brief Side length of the square spatial grid (N×N).
    grid_size: int = 64
    ## @brief Number of grid channels (Machine, Job, AGV layers).
    grid_channels: int = 3
    ## @brief Inclusive (min, max) range for sampling the number of machines per episode.
    num_machines_range: Tuple[int, int] = (15, 20)
    ## @brief Inclusive (min, max) range for sampling the number of jobs per episode.
    num_jobs_range: Tuple[int, int] = (50, 100)
    ## @brief Maximum operations per machine (O_per_mach = M).
    ops_per_machine: int = 20
    ## @brief Inclusive (min, max) range for random processing times.
    processing_time_range: Tuple[int, int] = (1, 99)
    ## @brief Number of normalized global scalar features.
    num_global_scalars: int = 10
    ## @brief Flattened pairwise distance-matrix length (8×8 = 64).
    distance_matrix_dim: int = 64
    ## @brief Number of binary event-flag indicators.
    num_event_flags: int = 6
    ## @brief Upper bound on machines for fixed-size tensor allocation.
    max_machines: int = 20
    ## @brief Upper bound on jobs for fixed-size tensor allocation.
    max_jobs: int = 100


@dataclass
class SchedulingMatrixConfig:
    """@brief Scheduling-matrix image dimensions: n × 2m × 3.

    @details
    The scheduling matrix is encoded as a 3-channel image where
    channel 0 holds machine assignments, channel 1 holds processing
    times, and channel 2 is reserved (zeros).
    """

    ## @brief Number of job rows, padded to the maximum (n).
    max_jobs: int = 100
    ## @brief Number of columns: 2m where m = @ref EnvConfig.max_machines.
    max_cols: int = 40
    ## @brief Image channels (machines, processing time, zeros).
    channels: int = 3


@dataclass
class EncoderConfig:
    """@brief Encoder output dimensions from the architecture diagram.

    @details
    Each field specifies the embedding dimensionality produced by the
    corresponding sub-encoder.  The read-only property @ref concat_dim
    returns the sum of all outputs (464-D by default).
    """

    ## @brief CNN-SPPF output dim for the Factory Floor grid.
    factory_cnn_out: int = 256
    ## @brief CNN-SPPF output dim for the Scheduling Matrix image.
    sched_cnn_out: int = 128
    ## @brief MLP output dim for global context scalars.
    global_mlp_out: int = 32
    ## @brief MLP output dim for the flattened distance matrix.
    distance_mlp_out: int = 32
    ## @brief MLP output dim for binary event flags.
    event_embed_out: int = 16
    ## @brief Max-pool kernel sizes for the SPPF pyramid.
    sppf_pool_sizes: List[int] = field(default_factory=lambda: [5, 9, 13])

    @property
    def concat_dim(self) -> int:
        """@brief Total concatenated dimension: 256 + 128 + 32 + 32 + 16 = 464.

        @return Sum of all sub-encoder output dimensions.
        """
        return (
            self.factory_cnn_out
            + self.sched_cnn_out
            + self.global_mlp_out
            + self.distance_mlp_out
            + self.event_embed_out
        )


@dataclass
class FusionConfig:
    """@brief Fusion head and domain-randomization parameters.

    @details
    Controls the MLP widths, dropout probability, and Gaussian noise
    standard deviation applied during training for sim-to-real transfer.
    """

    ## @brief Input dimensionality (must match @ref EncoderConfig.concat_dim).
    input_dim: int = 464
    ## @brief Width of the intermediate fully-connected layer.
    hidden_dim: int = 512
    ## @brief Dimensionality of the fused representation.
    output_dim: int = 256
    ## @brief Dropout probability for domain randomization.
    dropout_rate: float = 0.1
    ## @brief Std-dev of additive Gaussian noise for domain randomization.
    noise_std: float = 0.01


@dataclass
class ActorCriticConfig:
    """@brief Actor-Critic head dimensions.

    @details
    Both the actor and critic share the same hidden-layer width.
    The number of actions corresponds to the composite PDR rule set.
    """

    ## @brief Input dimensionality (must match @ref FusionConfig.output_dim).
    input_dim: int = 256
    ## @brief Width of the hidden layer in both actor and critic.
    hidden_dim: int = 256
    ## @brief Number of discrete actions (composite PDR rules).
    num_actions: int = 8


@dataclass
class PPOConfig:
    """@brief PPO training hyperparameters.

    @details
    Default values follow standard PPO recommendations
    (Schulman et al., 2017) with minor adjustments for the
    scheduling domain.
    """

    ## @brief Adam learning rate.
    lr: float = 3e-4
    ## @brief Discount factor for future rewards.
    gamma: float = 0.99
    ## @brief GAE lambda for advantage estimation.
    gae_lambda: float = 0.95
    ## @brief PPO clipping range for the surrogate objective.
    clip_epsilon: float = 0.2
    ## @brief Coefficient for the entropy bonus in the total loss.
    entropy_coef: float = 0.01
    ## @brief Coefficient for the value-function loss term.
    value_coef: float = 0.5
    ## @brief Maximum gradient norm for clipping.
    max_grad_norm: float = 0.5
    ## @brief Number of PPO optimisation epochs per rollout.
    num_epochs: int = 4
    ## @brief Mini-batch size for each gradient step.
    batch_size: int = 64
    ## @brief Number of environment steps collected per rollout.
    rollout_length: int = 128
    ## @brief Number of parallel environments.
    num_envs: int = 8
    ## @brief Total environment steps for the training run.
    total_timesteps: int = 1_000_000


## @brief Composite PDR action labels (Job rule – Machine rule).
## @details
## Each entry pairs a job-dispatching rule with a machine-assignment
## rule.  The index into this list corresponds to the discrete action
## selected by the actor network.
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