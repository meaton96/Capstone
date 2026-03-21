"""
Placeholder Gym environment for the DRL scheduling network.

Generates synthetic observations matching the thesis state space so the
full training pipeline can be tested end-to-end before the Unity sim
is ready.

State space (all obs are dicts of tensors):
    factory_grid:    (3, 64, 64)   - spatial occupancy grid
    sched_matrix:    (3, 100, 40)  - scheduling matrix image
    global_scalars:  (10,)         - normalized scalar features
    distance_matrix: (64,)         - flattened pairwise distances
    event_flags:     (6,)          - binary event indicators

Action space: Discrete(8) - one of 8 composite PDR rules
Reward: synthetic shaped reward simulating makespan optimization
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EnvConfig, SchedulingMatrixConfig, PDR_ACTIONS


class PlaceholderSchedulingEnv(gym.Env):
    """
    Synthetic environment that mimics the Unity factory sim's interface.

    When you connect the real Unity sim, replace this class with a wrapper
    that translates Unity observations into the same dict format.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_cfg: EnvConfig = None,
                 sched_cfg: SchedulingMatrixConfig = None,
                 max_steps: int = 200, seed: Optional[int] = None):
        super().__init__()
        self.cfg = env_cfg or EnvConfig()
        self.sched_cfg = sched_cfg or SchedulingMatrixConfig()
        self.max_steps = max_steps

        # ---- Action space: 8 composite PDR rules ----
        self.action_space = spaces.Discrete(len(PDR_ACTIONS))

        # ---- Observation space ----
        self.observation_space = spaces.Dict({
            "factory_grid": spaces.Box(
                0, 1, shape=(self.cfg.grid_channels, self.cfg.grid_size,
                             self.cfg.grid_size), dtype=np.float32
            ),
            "sched_matrix": spaces.Box(
                0, 1, shape=(self.sched_cfg.channels, self.sched_cfg.max_jobs,
                             self.sched_cfg.max_cols), dtype=np.float32
            ),
            "global_scalars": spaces.Box(
                0, 1, shape=(self.cfg.num_global_scalars,), dtype=np.float32
            ),
            "distance_matrix": spaces.Box(
                0, 1, shape=(self.cfg.distance_matrix_dim,), dtype=np.float32
            ),
            "event_flags": spaces.Box(
                0, 1, shape=(self.cfg.num_event_flags,), dtype=np.float32
            ),
        })

        # ---- Internal state ----
        self._step_count = 0
        self._num_machines = 0
        self._num_jobs = 0
        self._makespan_estimate = 0.0
        self._rng = np.random.default_rng(seed)

    def _generate_obs(self) -> dict:
        """Generate a synthetic observation matching the state space."""
        rng = self._rng

        # Spatial grid: sparse occupancy (mostly zeros with some active cells)
        factory_grid = np.zeros(
            (self.cfg.grid_channels, self.cfg.grid_size, self.cfg.grid_size),
            dtype=np.float32,
        )
        # Scatter machines on layer 0
        for _ in range(self._num_machines):
            r, c = rng.integers(0, self.cfg.grid_size, size=2)
            factory_grid[0, r, c] = 1.0
        # Scatter some jobs on layer 1
        n_active = rng.integers(0, min(self._num_jobs, 30))
        for _ in range(n_active):
            r, c = rng.integers(0, self.cfg.grid_size, size=2)
            factory_grid[1, r, c] = rng.uniform(0.1, 1.0)
        # AGV layer 2: a few active positions
        n_agv = rng.integers(1, 5)
        for _ in range(n_agv):
            r, c = rng.integers(0, self.cfg.grid_size, size=2)
            factory_grid[2, r, c] = 1.0

        # Scheduling matrix image
        sched = np.zeros(
            (self.sched_cfg.channels, self.sched_cfg.max_jobs,
             self.sched_cfg.max_cols),
            dtype=np.float32,
        )
        # Fill active job rows
        for j in range(self._num_jobs):
            for m in range(self._num_machines):
                sched[0, j, m] = rng.uniform(0, 1)          # machine assignment
                sched[1, j, m + self._num_machines] = (
                    rng.uniform(0, 1)                         # processing time
                )
            # channel 2 stays zeros as per diagram

        # Global scalars (all normalized 0-1)
        progress = self._step_count / self.max_steps
        global_scalars = np.array([
            progress,                                 # sim_t_norm
            rng.uniform(0.3, 1.0),                    # C_max_LB_norm
            rng.uniform(0, 1),                        # #jobs_waiting_norm
            rng.uniform(0, 1),                        # #jobs_active_norm
            rng.uniform(0, 0.3),                      # #failures_norm
            rng.uniform(0, 1),                        # throughput_rolling
            rng.uniform(0, 1),                        # avg_queue_len_norm
            min(1.0, self._step_count / 50),          # job_completion_rate
            rng.uniform(0, 1),                        # time_since_last_event_norm
            float(progress > 0.9),                    # overtime_flag
        ], dtype=np.float32)

        # Pairwise distance matrix (8x8 flattened -> 64)
        dist_mat = rng.uniform(0, 1, size=(64,)).astype(np.float32)

        # Event flags (sparse binary)
        event_flags = np.zeros(6, dtype=np.float32)
        # Randomly trigger 0-2 events
        n_events = rng.integers(0, 3)
        if n_events > 0:
            idxs = rng.choice(6, size=n_events, replace=False)
            event_flags[idxs] = 1.0

        return {
            "factory_grid": factory_grid,
            "sched_matrix": sched,
            "global_scalars": global_scalars,
            "distance_matrix": dist_mat,
            "event_flags": event_flags,
        }

    def _compute_reward(self, action: int) -> float:
        """
        Synthetic reward signal.

        In the real env this comes from DES queue metrics (makespan, tardiness).
        Here we simulate a shaped reward that decreases makespan over time
        with some noise, so the agent has a learnable signal.
        """
        rng = self._rng

        # Base reward: negative makespan delta (want to minimize)
        prev_makespan = self._makespan_estimate
        # Simulate that some actions are better than others
        action_quality = [0.8, 0.7, 0.5, 0.75, 0.4, 0.45, 0.6, 0.65]
        reduction = action_quality[action] * rng.uniform(0.5, 1.5)
        self._makespan_estimate = max(0, prev_makespan - reduction + rng.normal(0, 0.3))

        reward = -(self._makespan_estimate - prev_makespan)  # positive when makespan decreases
        reward += -0.01  # small step penalty to encourage efficiency
        return float(reward)

    def reset(self, *, seed=None, options=None) -> Tuple[dict, dict]:
        """Reset and return (obs, info)."""
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
        """
        Execute one scheduling decision.

        Returns: (obs, reward, terminated, truncated, info)
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
    """
    Simple synchronous vectorized wrapper for multiple envs.
    (Replace with gymnasium.vector.SyncVectorEnv if preferred.)
    """

    def __init__(self, num_envs: int, **env_kwargs):
        self.envs = [
            PlaceholderSchedulingEnv(seed=i, **env_kwargs)
            for i in range(num_envs)
        ]
        self.num_envs = num_envs

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs_list, info_list = zip(*results)
        return self._stack_obs(obs_list), list(info_list)

    def step(self, actions):
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
        """Stack list of obs dicts into a batched dict of numpy arrays."""
        keys = obs_list[0].keys()
        return {k: np.stack([o[k] for o in obs_list]) for k in keys}
