"""
@file placeholder_env.py
@brief Placeholder Gym environment for the DRL scheduling network.

Generates synthetic observations matching the thesis state space so the
full training pipeline can be tested end-to-end before the Unity sim
is ready.

@par State Space
All observations are dicts of tensors:
| Key              | Shape          | Description                       |
|------------------|----------------|-----------------------------------|
| factory_grid     | (3, 64, 64)    | Spatial occupancy grid            |
| machine_table    | (100, 16)      | Per-machine features (padded)     |
| job_table        | (64, 17)       | Per-active-job features (padded)  |
| global_scalars   | (16,)          | Normalized scalar features        |
| event_flags      | (6,)           | Binary event indicators           |

@par Action Space
Discrete(8) — one of 8 composite PDR rules.

@par Reward
Synthetic shaped reward simulating makespan optimization.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (EnvConfig, PDR_ACTIONS, MACHINE_PRESENT_COL, MACHINE_CANDIDATE_COL,
                    JOB_PRESENT_COL, JOB_CANDIDATE_COL)


class PlaceholderSchedulingEnv(gym.Env):
    """@brief Synthetic environment that mimics the Unity factory sim's interface.

    When you connect the real Unity sim, replace this class with a wrapper
    that translates Unity observations into the same dict format.

    @details
    This environment generates random observations that conform to the
    expected observation-space schema, and returns a synthetic shaped
    reward so the full RL training loop can be validated before the
    high-fidelity Unity simulation is available.
    """

    ## @brief Gymnasium metadata declaring supported render modes.
    metadata = {"render_modes": ["human"]}

    def __init__(self, env_cfg: EnvConfig = None,
                 max_steps: int = 200, seed: Optional[int] = None):
        """@brief Construct the placeholder scheduling environment.

        @param env_cfg      Environment geometry/dimensions config.
                            Defaults to EnvConfig() if None.
        @param max_steps    Maximum steps per episode before truncation.
        @param seed         Optional RNG seed for reproducibility.
        """
        super().__init__()

        ## @brief Environment geometry configuration.
        self.cfg = env_cfg or EnvConfig()
        ## @brief Maximum steps per episode before truncation.
        self.max_steps = max_steps

        ## @brief Action space: 8 composite PDR rules (Discrete).
        self.action_space = spaces.Discrete(len(PDR_ACTIONS))

        ## @brief Dict observation space matching the thesis state-space spec.
        self.observation_space = spaces.Dict({
            "factory_grid": spaces.Box(
                0, 1, shape=(self.cfg.grid_channels, self.cfg.grid_size,
                             self.cfg.grid_size), dtype=np.float32
            ),
            "machine_table": spaces.Box(
                0, 1, shape=(self.cfg.max_machines, self.cfg.machine_features), dtype=np.float32
            ),
            "job_table": spaces.Box(
                0, 1, shape=(self.cfg.max_jobs, self.cfg.job_features), dtype=np.float32
            ),
            "global_scalars": spaces.Box(
                0, 1, shape=(self.cfg.num_global_scalars,), dtype=np.float32
            ),
            "event_flags": spaces.Box(
                0, 1, shape=(self.cfg.num_event_flags,), dtype=np.float32
            ),
        })

        ## @brief Current step count within the episode.
        self._step_count = 0
        ## @brief Number of machines sampled for this episode.
        self._num_machines = 0
        ## @brief Number of jobs sampled for this episode.
        self._num_jobs = 0
        ## @brief Running makespan estimate driving the reward signal.
        self._makespan_estimate = 0.0
        ## @brief NumPy random generator instance.
        self._rng = np.random.default_rng(seed)

    def _generate_obs(self) -> dict:
        """@brief Generate a synthetic observation matching the state space.

        @details
        Builds each observation component with plausible random values:
          - **factory_grid**: sparse occupancy across machine, job, and AGV layers.
          - **machine_table** / **job_table**: random features on this episode's present rows.
          - **global_scalars**: uniform-random, element 0 = episode progress.
          - **event_flags**: sparse binary vector with 0–2 active events.

        @return Dict mapping observation keys to numpy arrays.
        """
        rng = self._rng

        # --- Factory grid: sparse occupancy (mostly zeros with some active cells) ---
        factory_grid = np.zeros(
            (self.cfg.grid_channels, self.cfg.grid_size, self.cfg.grid_size),
            dtype=np.float32,
        )
        # Layer 0: scatter machines
        for _ in range(self._num_machines):
            r, c = rng.integers(0, self.cfg.grid_size, size=2)
            factory_grid[0, r, c] = 1.0
        # Layer 1: scatter active jobs with variable intensity
        n_active = rng.integers(0, min(self._num_jobs, 30))
        for _ in range(n_active):
            r, c = rng.integers(0, self.cfg.grid_size, size=2)
            factory_grid[1, r, c] = rng.uniform(0.1, 1.0)
        # Layer 2: AGV positions
        n_agv = rng.integers(1, 5)
        for _ in range(n_agv):
            r, c = rng.integers(0, self.cfg.grid_size, size=2)
            factory_grid[2, r, c] = 1.0

        # --- Machine table: present rows for this episode's floor, random features ---
        machine_table = np.zeros((self.cfg.max_machines, self.cfg.machine_features), dtype=np.float32)
        n_m = min(self._num_machines, self.cfg.max_machines)
        machine_table[:n_m] = rng.uniform(0, 1, size=(n_m, self.cfg.machine_features))
        machine_table[:n_m, MACHINE_PRESENT_COL] = 1.0
        machine_table[:n_m, MACHINE_CANDIDATE_COL] = (rng.uniform(size=n_m) < 0.3).astype(np.float32)

        # --- Job table: active jobs, random features ---
        job_table = np.zeros((self.cfg.max_jobs, self.cfg.job_features), dtype=np.float32)
        n_j = min(int(rng.integers(1, self._num_jobs + 1)), self.cfg.max_jobs)
        job_table[:n_j] = rng.uniform(0, 1, size=(n_j, self.cfg.job_features))
        job_table[:n_j, JOB_PRESENT_COL] = 1.0
        job_table[:n_j, JOB_CANDIDATE_COL] = (rng.uniform(size=n_j) < 0.3).astype(np.float32)

        # --- Global scalars (all normalized 0-1) ---
        global_scalars = rng.uniform(0, 1, size=(self.cfg.num_global_scalars,)).astype(np.float32)
        global_scalars[0] = self._step_count / self.max_steps

        # --- Event flags (sparse binary) ---
        event_flags = np.zeros(6, dtype=np.float32)
        n_events = rng.integers(0, 3)
        if n_events > 0:
            idxs = rng.choice(6, size=n_events, replace=False)
            event_flags[idxs] = 1.0

        return {
            "factory_grid": factory_grid,
            "machine_table": machine_table,
            "job_table": job_table,
            "global_scalars": global_scalars,
            "event_flags": event_flags,
        }

    def _compute_reward(self, action: int) -> float:
        """@brief Compute a synthetic reward signal for the given action.

        @details
        In the real environment the reward derives from DES queue metrics
        (makespan, tardiness).  Here we simulate a shaped signal where
        makespan decreases over time with action-dependent quality and
        additive noise, giving the agent a learnable gradient.

        A small step penalty (−0.01) encourages the agent to finish
        episodes efficiently.

        @param action  Index into PDR_ACTIONS (0–7).
        @return Scalar reward value (float).
        """
        rng = self._rng

        # Base reward: negative makespan delta (want to minimize)
        prev_makespan = self._makespan_estimate

        ## @brief Per-action quality multipliers (higher = stronger reduction).
        action_quality = [0.8, 0.7, 0.5, 0.75, 0.4, 0.45, 0.6, 0.65]
        reduction = action_quality[action] * rng.uniform(0.5, 1.5)
        self._makespan_estimate = max(0, prev_makespan - reduction + rng.normal(0, 0.3))

        reward = -(self._makespan_estimate - prev_makespan)  # positive when makespan decreases
        reward += -0.01  # small step penalty to encourage efficiency
        return float(reward)

    def reset(self, *, seed=None, options=None) -> Tuple[dict, dict]:
        """@brief Reset the environment to a new random episode.

        @param seed     Optional RNG seed override for this episode.
        @param options  Unused; present for Gymnasium API compatibility.
        @return Tuple of (observation dict, info dict).
                Info contains @c num_machines, @c num_jobs, and
                @c initial_makespan for the new episode.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._num_machines = int(
            self._rng.integers(*self.cfg.num_machines_range)
        )
        self._num_jobs = int(
            self._rng.integers(*self.cfg.num_jobs_range)
        )
        self._makespan_estimate = float(
            self._rng.uniform(50, 200)
        )

        obs = self._generate_obs()
        info = {
            "num_machines": self._num_machines,
            "num_jobs": self._num_jobs,
            "initial_makespan": self._makespan_estimate,
        }
        return obs, info

    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        """@brief Execute one scheduling decision and advance the environment.

        @param action  Integer action index in [0, 7] selecting a PDR rule.
        @return Tuple of (obs, reward, terminated, truncated, info).
                - @c terminated is True when makespan reaches zero.
                - @c truncated  is True when @ref _step_count reaches
                  @ref max_steps.
                - @c info contains @c step, @c makespan_estimate, and
                  @c pdr_rule name.
        @throws AssertionError if @p action is outside the action space.
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        self._step_count += 1

        reward = self._compute_reward(action)
        obs = self._generate_obs()

        # Episode ends when all jobs "complete" or max steps reached
        terminated = self._makespan_estimate <= 0
        truncated = self._step_count >= self.max_steps

        info = {
            "step": self._step_count,
            "makespan_estimate": self._makespan_estimate,
            "pdr_rule": PDR_ACTIONS[action],
        }
        return obs, reward, terminated, truncated, info


class VectorizedPlaceholderEnv:
    """@brief Simple synchronous vectorized wrapper for multiple environments.

    @details
    Manages a list of PlaceholderSchedulingEnv instances and exposes
    batched @c reset / @c step methods with automatic episode resets.

    @note Replace with gymnasium.vector.SyncVectorEnv if preferred.
    """

    def __init__(self, num_envs: int, env_wrapper=None, **env_kwargs):
        """@brief Create a bank of parallel placeholder environments.

        @param num_envs     Number of independent environment instances.
        @param env_wrapper  Optional callable that wraps each individual
                            environment (e.g. @ref SensorCorruptionWrapper).
                            Signature: @c wrapper(env) -> wrapped_env.
                            Applied after construction, before first reset.
        @param env_kwargs   Keyword arguments forwarded to each
                            PlaceholderSchedulingEnv constructor.
                            Each env receives its index as the seed.
        """
        ## @brief List of independent environment instances (possibly wrapped).
        self.envs = []
        for i in range(num_envs):
            env = PlaceholderSchedulingEnv(seed=i, **env_kwargs)
            if env_wrapper is not None:
                env = env_wrapper(env)
            self.envs.append(env)
        ## @brief Number of parallel environments.
        self.num_envs = num_envs

    def reset(self):
        """@brief Reset all environments simultaneously.

        @return Tuple of (batched_obs, info_list) where batched_obs is a
                dict of stacked numpy arrays with leading dimension
                @ref num_envs.
        """
        results = [env.reset() for env in self.envs]
        obs_list, info_list = zip(*results)
        return self._stack_obs(obs_list), list(info_list)

    def step(self, actions):
        """@brief Step all environments with the given action vector.

        Automatically resets any environment whose episode has ended
        (terminated or truncated).

        @param actions  Array-like of length @ref num_envs with integer
                        actions for each environment.
        @return Tuple of (batched_obs, rewards, terminateds, truncateds, infos).
                Each array has leading dimension @ref num_envs.
        """
        results = [
            env.step(int(a)) for env, a in zip(self.envs, actions)
        ]
        obs_list, rewards, terminateds, truncateds, infos = zip(*results)

        # Auto-reset terminated/truncated envs
        new_obs_list = list(obs_list)
        for i, (term, trunc) in enumerate(zip(terminateds, truncateds)):
            if term or trunc:
                new_obs, _ = self.envs[i].reset()
                new_obs_list[i] = new_obs

        return (
            self._stack_obs(new_obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds),
            np.array(truncateds),
            list(infos),
        )

    @staticmethod
    def _stack_obs(obs_list):
        """@brief Stack a list of observation dicts into a batched dict.

        @param obs_list  List of observation dicts from individual envs.
        @return Dict mapping each key to a numpy array with a new leading
                batch dimension.
        """
        keys = obs_list[0].keys()
        return {k: np.stack([o[k] for o in obs_list]) for k in keys}