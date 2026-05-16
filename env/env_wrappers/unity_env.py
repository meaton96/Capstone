"""
@file unity_env.py
@brief Gymnasium-compatible wrapper around the ML-Agents Unity environment.

Updated to include:
  - EpisodeConfigChannel: push full FJSSPConfig JSON to Unity before each reset
  - EpisodeTelemetryChannel: receive per-episode events/results from Unity

Side channel usage:
  env.send_config(config_dict)   # call before reset() to set next episode config
  obs = env.reset()
  ...episode loop...
  payload = env.telemetry.pop_payload()  # call after done=True

For curriculum training, the outer training loop manages which config to send:
  for episode in curriculum:
      env.send_config(curriculum.current_config())
      obs = env.reset()
      ...
"""

import numpy as np
from typing import Optional, Tuple, Dict

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.base_env import ActionTuple

from config import (
    GRID_SIZE, GRID_CHANNELS, MAX_JOBS, MAX_MACHINES, SCHED_CHANNELS,
    GLOBAL_SCALARS, DISTANCE_DIM, EVENT_FLAGS, TOTAL_OBS_SIZE,
    SLICE_SPATIAL_END, SLICE_SCHED_END, SLICE_SCALARS_END,
    SLICE_DIST_END, SLICE_FLAGS_END,
)
from env.channels import EpisodeConfigChannel, EpisodeTelemetryChannel


def slice_obs(raw: np.ndarray) -> Dict[str, np.ndarray]:
    """@brief Slice a flat observation vector into the five named streams."""
    assert raw.shape[-1] == TOTAL_OBS_SIZE, (
        f"Expected {TOTAL_OBS_SIZE} floats, got {raw.shape[-1]}"
    )

    factory_grid = raw[..., :SLICE_SPATIAL_END].reshape(
        *raw.shape[:-1], GRID_CHANNELS, GRID_SIZE, GRID_SIZE
    )

    sched_cols = 2 * MAX_MACHINES
    sched_hwc = raw[..., SLICE_SPATIAL_END:SLICE_SCHED_END].reshape(
        *raw.shape[:-1], MAX_JOBS, sched_cols, SCHED_CHANNELS
    )
    sched_matrix = np.moveaxis(sched_hwc, -1, -3)

    global_scalars  = raw[..., SLICE_SCHED_END:SLICE_SCALARS_END]
    distance_matrix = raw[..., SLICE_SCALARS_END:SLICE_DIST_END]
    event_flags     = raw[..., SLICE_DIST_END:SLICE_FLAGS_END]

    return {
        "factory_grid":    factory_grid.astype(np.float32),
        "sched_matrix":    sched_matrix.astype(np.float32),
        "global_scalars":  global_scalars.astype(np.float32),
        "distance_matrix": distance_matrix.astype(np.float32),
        "event_flags":     event_flags.astype(np.float32),
    }


class UnitySchedulingEnv:
    """@brief Single-agent wrapper around the ML-Agents Unity environment."""

    def __init__(self, file_name: Optional[str] = None,
                 time_scale: float = 20.0, worker_id: int = 0,
                 timeout_wait: int = 300, no_graphics: bool = False):

        self.engine_channel = EngineConfigurationChannel()
        self.config_channel = EpisodeConfigChannel()
        self.telemetry = EpisodeTelemetryChannel()

        self.env = UnityEnvironment(
            file_name=file_name,
            side_channels=[
                self.engine_channel,
                self.config_channel,
                self.telemetry,
            ],
            worker_id=worker_id,
            timeout_wait=timeout_wait,
            no_graphics=no_graphics,
        )
        self.engine_channel.set_configuration_parameters(time_scale=time_scale)

        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]

        obs_shape = self.spec.observation_specs[0].shape
        assert obs_shape == (TOTAL_OBS_SIZE,), (
            f"Unity VectorSensor size {obs_shape} does not match "
            f"expected ({TOTAL_OBS_SIZE},). Update BehaviorParameters "
            f"Space Size in the Inspector to {TOTAL_OBS_SIZE}."
        )

        self._last_obs = None
        self._pending_config = None

    def send_config(self, config: dict):
        """
        Queue a config to be sent on the next reset().
        Call this before reset() to control the next episode's parameters.

        In deterministic mode (no stochastic block), Unity uses its default
        config (whatever is set in the Inspector / HeadlessBatchRunner).
        In curriculum training, call this every episode with the current
        curriculum stage's config dict.
        """
        self._pending_config = config

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the Unity episode and return the first observation."""
        # Send config before reset so Unity receives it in OnEpisodeBegin
        if self._pending_config is not None:
            self.config_channel.send_config(self._pending_config)
            self._pending_config = None

        self.env.reset()
        obs, _, _, _ = self._step_until_decision(action=0)
        return obs

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        """Send a scheduling action and wait for the next decision point."""
        obs, reward, done, info = self._step_until_decision(action)

        # Attach telemetry payload to info when episode ends
        if done:
            info["telemetry"] = self.telemetry.pop_payload()

        return obs, reward, done, info

    def _step_until_decision(self, action: int):
        accumulated_reward = 0.0
        info = {}

        while True:
            action_tuple = ActionTuple(
                discrete=np.array([[action]], dtype=np.int32)
            )
            self.env.set_actions(self.behavior_name, action_tuple)
            self.env.step()

            decision_steps, terminal_steps = self.env.get_steps(
                self.behavior_name
            )

            if len(terminal_steps) > 0:
                obs = self._extract_obs(terminal_steps)
                accumulated_reward += float(terminal_steps.reward[0])
                self._last_obs = obs
                return obs, accumulated_reward, True, info

            if len(decision_steps) > 0:
                obs = self._extract_obs(decision_steps)
                accumulated_reward += float(decision_steps.reward[0])
                self._last_obs = obs
                return obs, accumulated_reward, False, info

    def _extract_obs(self, steps) -> Dict[str, np.ndarray]:
        raw = steps.obs[0][0]
        return slice_obs(raw)

    def close(self):
        self.env.close()


class VectorizedUnityEnv:
    """Manages multiple Unity instances for parallel data collection."""

    def __init__(self, num_envs: int, file_name: Optional[str] = None,
                 time_scale: float = 20.0, base_worker_id: int = 0,
                 timeout_wait: int = 300, no_graphics: bool = False):
        self.envs = [
            UnitySchedulingEnv(
                file_name=file_name,
                time_scale=time_scale,
                worker_id=base_worker_id + i,
                timeout_wait=timeout_wait,
                no_graphics=no_graphics,
            )
            for i in range(num_envs)
        ]
        self.num_envs = num_envs

    def send_configs(self, configs: list):
        """Push one config per env. configs[i] applies to envs[i]."""
        for env, cfg in zip(self.envs, configs):
            if cfg is not None:
                env.send_config(cfg)

    def send_config_all(self, config: dict):
        """Push the same config to all envs."""
        for env in self.envs:
            env.send_config(config)

    def reset(self):
        results = [env.reset() for env in self.envs]
        return self._stack_obs(results), [{}] * self.num_envs

    def step(self, actions):
        results = [
            env.step(int(a)) for env, a in zip(self.envs, actions)
        ]
        obs_list, rewards, dones, infos = zip(*results)

        new_obs = list(obs_list)
        for i, done in enumerate(dones):
            if done:
                new_obs[i] = self.envs[i].reset()

        return (
            self._stack_obs(new_obs),
            np.array(rewards, dtype=np.float32),
            np.array(dones),
            np.zeros(self.num_envs, dtype=bool),
            list(infos),
        )

    def close(self):
        for env in self.envs:
            env.close()

    @staticmethod
    def _stack_obs(obs_list):
        keys = obs_list[0].keys()
        return {k: np.stack([o[k] for o in obs_list]) for k in keys}