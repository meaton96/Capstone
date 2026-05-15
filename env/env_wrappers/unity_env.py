"""
@file unity_env.py
@brief Gymnasium-compatible wrapper around the ML-Agents Unity environment.

@details
Handles the fundamental mismatch between ML-Agents' frame-by-frame
communication and the event-driven scheduling simulation.  The key
pattern: env.step() loops internally through Unity frames until the
agent actually requests a scheduling decision (or the episode ends),
accumulating rewards across the gap.

@par Observation Slicing
The C# ObservationBuilder serializes five streams via FlattenStreams:

| Stream            | C# layout           | Floats | PyTorch shape      |
|-------------------|----------------------|--------|--------------------|
| Spatial grid      | (C, H, W)           | 12,288 | (3, 64, 64)        |
| Scheduling matrix | (H, W, C) = (n,2m,3)|    960 | (3, 20, 16) → CHW  |
| Global scalars    | flat                 |     10 | (10,)              |
| Distance matrix   | flat                 |     64 | (64,)              |
| Event flags       | flat                 |      6 | (6,)               |
|                   |                      | 13,328 |                    |
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


def slice_obs(raw: np.ndarray) -> Dict[str, np.ndarray]:
    """@brief Slice a flat observation vector into the five named streams.

    @param raw  1-D float32 array of length TOTAL_OBS_SIZE (13,328).
    @return Dict with keys matching the SchedulingNetwork input.
    """
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
        "factory_grid":   factory_grid.astype(np.float32),
        "sched_matrix":   sched_matrix.astype(np.float32),
        "global_scalars": global_scalars.astype(np.float32),
        "distance_matrix": distance_matrix.astype(np.float32),
        "event_flags":    event_flags.astype(np.float32),
    }


class UnitySchedulingEnv:
    """@brief Single-agent wrapper around the ML-Agents Unity environment.

    @details
    The simulation only requests decisions at discrete scheduling events
    (machine finishes, new job arrives).  Between events, Unity runs
    many physics frames with no agent interaction.  This wrapper loops
    through those "silent" frames internally so the caller only sees
    steps where a real scheduling decision was made.

    Rewards from intermediate frames are accumulated and returned as a
    single summed reward for the decision step.
    """

    def __init__(self, file_name: Optional[str] = None,
                 time_scale: float = 20.0, worker_id: int = 0,
                 timeout_wait: int = 300, no_graphics: bool = False):
        """@brief Connect to Unity and configure the simulation speed.

        @param file_name     Path to a built Unity executable, or None
                             to attach to the Editor.
        @param time_scale    Simulation speed multiplier.
        @param worker_id     Unique ID for parallel training processes.
        @param timeout_wait  Seconds before gRPC timeout (default 300
                             to handle long processing phases).
        @param no_graphics   Pass True for headless server training.
        """
        self.engine_channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(
            file_name=file_name,
            side_channels=[self.engine_channel],
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
            f"expected ({TOTAL_OBS_SIZE},).  Update the BehaviorParameters "
            f"Space Size in the Inspector to {TOTAL_OBS_SIZE}."
        )

        self._last_obs = None
        self._last_action = 0

    def reset(self) -> Dict[str, np.ndarray]:
        """@brief Reset the Unity episode and return the first observation.

        @details Loops through Unity frames until the agent requests its
                 first scheduling decision.
        """
        self.env.reset()
        self._last_action = 0

        # Spin until the first decision is requested.
        obs, _, _, _ = self._step_until_decision(action=0)
        return obs

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        """@brief Send a scheduling action and wait for the next decision point.

        @param action  Integer action index (0-7, selecting a PDR rule).
        @return Tuple of (obs_dict, accumulated_reward, done, info).
        """
        return self._step_until_decision(action)

    def _step_until_decision(self, action: int):
        """@brief Core loop: advances Unity until the agent needs a decision.

        @details
        Sends the action, then keeps stepping Unity frame-by-frame.
        Each frame that returns empty decision_steps is a "silent"
        physics frame — the reward is accumulated but no observation
        is returned to the caller.  The loop exits when:
          - decision_steps is non-empty (real scheduling decision needed)
          - terminal_steps is non-empty (episode ended)

        @param action  The discrete action to send to Unity.
        @return Tuple of (obs_dict, accumulated_reward, done, info).
        """
        accumulated_reward = 0.0
        info = {}
        silent_frames = 0
        while True:
            # Send the action (first iteration: the real action;
            # subsequent iterations: repeat the last action as a no-op
            # since the bridge ignores it when IsWaitingForAction=false).
            action_tuple = ActionTuple(
                discrete=np.array([[action]], dtype=np.int32)
            )
            self.env.set_actions(self.behavior_name, action_tuple)
            self.env.step()

            decision_steps, terminal_steps = self.env.get_steps(
                self.behavior_name
            )

            silent_frames += 1
            if silent_frames % 500 == 0:
                print(f"[DEBUG] {silent_frames} silent frames, "
                    f"decision={len(decision_steps)}, terminal={len(terminal_steps)}")

            # ── Episode ended ──
            if len(terminal_steps) > 0:
                obs = self._extract_obs(terminal_steps)
                accumulated_reward += float(terminal_steps.reward[0])
                self._last_obs = obs
                return obs, accumulated_reward, True, info

            # ── Agent requested a new decision ──
            if len(decision_steps) > 0:
                obs = self._extract_obs(decision_steps)
                accumulated_reward += float(decision_steps.reward[0])
                self._last_obs = obs
                return obs, accumulated_reward, False, info

            # ── Silent frame: no decision needed, keep spinning ──
            # (No reward to accumulate from empty steps.)

    def _extract_obs(self, steps) -> Dict[str, np.ndarray]:
        """@brief Slice the flat ML-Agents observation into named streams."""
        raw = steps.obs[0][0]
        return slice_obs(raw)

    def close(self):
        """@brief Shut down the Unity environment gracefully."""
        self.env.close()


class VectorizedUnityEnv:
    """@brief Manages multiple Unity instances for parallel data collection.

    @details
    For Unity Editor training, use num_envs=1 (only one Editor
    connection is possible).  For parallel training, build the Unity
    project and pass the executable path.
    """

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