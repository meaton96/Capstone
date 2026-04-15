# """
# @file sensor_corruption.py
# @brief Gymnasium observation wrapper simulating real-world sensor degradation.

# @details
# Applies per-modality corruption to raw observations *before* they reach
# the encoder network, simulating physical phenomena that would occur on a
# real factory floor:

#   - **Sensor dropout**: random grid cells or vector elements are zeroed,
#     modelling IoT sensor failures or communication dropouts.
#   - **Measurement noise**: additive Gaussian noise simulating imprecise
#     readings (e.g. AGV position off by a fraction of a metre, processing-
#     time estimates with jitter).

# Each observation modality can be configured independently because the
# corruption characteristics differ across sensor types.

# @par Usage
# @code{.py}
# from env.sensor_corruption import SensorCorruptionWrapper
# from config import SensorCorruptionConfig

# env = PlaceholderSchedulingEnv()
# env = SensorCorruptionWrapper(env, SensorCorruptionConfig())
# obs, info = env.reset()  # obs is already corrupted
# @endcode
# """

# import gymnasium as gym
# import numpy as np
# from dataclasses import dataclass, field
# from typing import Optional, Tuple


# @dataclass
# class ModalityCorruptionConfig:
#     """@brief Corruption parameters for a single observation modality.

#     @details
#     Each modality can have its own dropout probability and noise
#     standard deviation, reflecting the physical characteristics of
#     that sensor type.
#     """

#     ## @brief Probability of zeroing each element (simulates sensor failure).
#     dropout_rate: float = 0.0
#     ## @brief Std-dev of additive Gaussian noise (simulates measurement error).
#     noise_std: float = 0.0


# @dataclass
# class SensorCorruptionConfig:
#     """@brief Per-modality corruption configuration for the observation wrapper.

#     @details
#     Default values are conservative starting points.  Tune per modality
#     based on the noise characteristics of the real sensor hardware.

#     @par Design rationale
#     - **factory_grid**: moderate dropout (IoT sensor failures) and spatial
#       noise (AGV localisation error, ~0.1 m positional jitter).
#     - **sched_matrix**: low dropout (MES data is relatively reliable),
#       small noise (processing-time estimation error).
#     - **global_scalars**: minimal dropout, light noise (aggregated KPIs
#       have some smoothing already).
#     - **distance_matrix**: no dropout (derived from fixed layout), small
#       noise (measurement jitter on pairwise distances).
#     - **event_flags**: no corruption by default — these are discrete
#       binary states, not continuous sensor readings.
#     """

#     ## @brief Corruption config for the factory-floor occupancy grid.
#     factory_grid: ModalityCorruptionConfig = field(
#         default_factory=lambda: ModalityCorruptionConfig(
#             dropout_rate=0.05, noise_std=0.02,
#         )
#     )
#     ## @brief Corruption config for the scheduling-matrix image.
#     sched_matrix: ModalityCorruptionConfig = field(
#         default_factory=lambda: ModalityCorruptionConfig(
#             dropout_rate=0.02, noise_std=0.01,
#         )
#     )
#     ## @brief Corruption config for the normalized global scalars.
#     global_scalars: ModalityCorruptionConfig = field(
#         default_factory=lambda: ModalityCorruptionConfig(
#             dropout_rate=0.01, noise_std=0.01,
#         )
#     )
#     ## @brief Corruption config for the flattened distance matrix.
#     distance_matrix: ModalityCorruptionConfig = field(
#         default_factory=lambda: ModalityCorruptionConfig(
#             dropout_rate=0.0, noise_std=0.01,
#         )
#     )
#     ## @brief Corruption config for the binary event flags.
#     event_flags: ModalityCorruptionConfig = field(
#         default_factory=lambda: ModalityCorruptionConfig(
#             dropout_rate=0.0, noise_std=0.0,
#         )
#     )

#     ## @brief Master switch: when False, all corruption is disabled
#     ##        (equivalent to eval / deployment mode).
#     enabled: bool = True


# class SensorCorruptionWrapper(gym.ObservationWrapper):
#     """@brief Gymnasium wrapper that corrupts observations to simulate
#     real-world sensor degradation.

#     @details
#     Wraps any environment whose observations are dicts of numpy arrays.
#     Each modality is corrupted independently according to its
#     @ref ModalityCorruptionConfig.  Corruption is only applied when
#     @ref enabled is True, making it easy to toggle off for evaluation
#     or deployment without removing the wrapper from the stack.

#     Observations are clamped to [0, 1] after corruption to preserve
#     the original observation-space contract.
#     """

#     def __init__(self, env: gym.Env,
#                  corruption_cfg: SensorCorruptionConfig = None,
#                  seed: Optional[int] = None):
#         """@brief Construct the sensor corruption wrapper.

#         @param env             The environment to wrap.
#         @param corruption_cfg  Per-modality corruption settings.
#                                Defaults to SensorCorruptionConfig() if None.
#         @param seed            Optional RNG seed for reproducibility.
#         """
#         super().__init__(env)

#         ## @brief Per-modality corruption configuration.
#         self.corruption_cfg = corruption_cfg or SensorCorruptionConfig()
#         ## @brief Whether corruption is currently active.
#         self.enabled = self.corruption_cfg.enabled
#         ## @brief NumPy random generator instance.
#         self._rng = np.random.default_rng(seed)

#         ## @brief Mapping from observation key to its ModalityCorruptionConfig.
#         self._modality_configs = {
#             "factory_grid": self.corruption_cfg.factory_grid,
#             "sched_matrix": self.corruption_cfg.sched_matrix,
#             "global_scalars": self.corruption_cfg.global_scalars,
#             "distance_matrix": self.corruption_cfg.distance_matrix,
#             "event_flags": self.corruption_cfg.event_flags,
#         }

#     def observation(self, obs: dict) -> dict:
#         """@brief Apply per-modality corruption to a raw observation dict.

#         @details
#         Called automatically by Gymnasium on every @c reset() and @c step().
#         When @ref enabled is False the observation is returned unchanged.

#         For each modality with non-zero corruption parameters:
#           1. A binary dropout mask is sampled and applied (element-wise
#              multiply by 0 or 1), simulating sensor failures.
#           2. Additive Gaussian noise is applied to the *surviving*
#              elements, simulating measurement imprecision.
#           3. Values are clamped to [0, 1].

#         @param obs  Raw observation dict from the wrapped environment.
#         @return Corrupted observation dict (same keys and shapes).
#         """
#         if not self.enabled:
#             return obs

#         corrupted = {}
#         for key, val in obs.items():
#             cfg = self._modality_configs.get(key)
#             if cfg is None or (cfg.dropout_rate == 0.0 and cfg.noise_std == 0.0):
#                 corrupted[key] = val
#                 continue

#             out = val.copy()

#             # Sensor dropout: zero out random elements
#             if cfg.dropout_rate > 0.0:
#                 mask = self._rng.random(out.shape) > cfg.dropout_rate
#                 out = out * mask.astype(out.dtype)

#             # Measurement noise on surviving elements
#             if cfg.noise_std > 0.0:
#                 noise = self._rng.normal(0.0, cfg.noise_std, size=out.shape)
#                 out = out + noise.astype(out.dtype)

#             # Clamp to observation-space bounds
#             out = np.clip(out, 0.0, 1.0)
#             corrupted[key] = out

#         return corrupted

#     def set_enabled(self, enabled: bool):
#         """@brief Toggle corruption on or off.

#         @details
#         Call with False during evaluation or deployment to get clean
#         observations without removing the wrapper.

#         @param enabled  True to apply corruption, False to pass through.
#         """
#         self.enabled = enabled