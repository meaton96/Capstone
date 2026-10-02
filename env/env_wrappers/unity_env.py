"""
@file unity_env.py
@brief Wrapper around the ML-Agents Unity environment for the custom PPO loop.

Each Python step sends one scheduling action and advances Unity to the next decision
(or episode end). The reward is computed here in Python from the reward-metrics
snapshots Unity attaches to every agent step (see env/rewards), so reward functions
can change without a rebuild. Without a reward function, Unity's own reward is passed
through (always 0 in current builds).

Episodes restart automatically inside Unity (AutoStartOnPlay under ML-Agents): when a
step ends an episode, the returned observation is already the first observation of
the next episode. Callers must not call reset() between episodes.

Instances are reproducible through per-episode seeds queued in Unity (EpisodeSeedChannel).
Each episode Unity starts consumes one queued seed and reports it back in its metrics
(episode_seed / episode_seed_index, -1 when the queue was empty). Seeds below
@ref TRAIN_SEED_LOW are reserved for evaluation.

Scripted scenarios (ScenarioLoader JSON) can replay a fixed instance (load_scenario) or, with
a scenario_generator + seed_rng, a fresh seeded variant every episode — queued in lockstep with
the seed above, one item per episode, buffered the same way. Long scenarios can opt into a
steady-state time cap (a scenario's "stochastic": {"episodeDurationSeconds": N}); an episode cut
short that way is truncated, not terminated: info["episode"]["truncated"] is set, and the info
dict carries the truncated episode's last observation as "terminal_obs" (Unity has already
auto-reset by the time step() returns, so obs/next_obs is the new episode's first frame — the
value network still needs terminal_obs to bootstrap the truncated one correctly; see
rollout_buffer.RolloutBuffer.add). That is the observation of the episode's last decision, kept
here in Python: the player's own terminal observation is all zeros (see step).

Side channel usage:
  env.send_config(config_dict)      # applied on the next reset()
  env.queue_seeds([3, 4, 5])        # instance seeds for upcoming episodes
  env.queue_scenarios([path_or_dict, ...])  # scripted scenarios for upcoming episodes
  info["telemetry"]                 # per-episode events, attached when done=True
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

from config import (
    GRID_SIZE, GRID_CHANNELS, MAX_JOBS, JOB_FEATURES, MAX_MACHINES, MACHINE_FEATURES,
    SPATIAL_LEN, GLOBAL_SCALARS, EVENT_FLAGS, ACTION_BRANCHES, ACTION_MASK_LEN, obs_total_size,
)
from channels.channels import EpisodeConfigChannel, EpisodeSeedChannel, EpisodeTelemetryChannel
from env_wrappers.loopback import LoopbackUnityEnvironment
from rewards import (
    SENSOR_NAME, LoadedReward, MetricsSnapshot, RewardContext, RewardFunction, elapsed_sim_time, load_reward,
)

## @brief Training seeds are drawn from [TRAIN_SEED_LOW, EpisodeSeedChannel.MAX_SEED);
##        seeds below it are reserved for evaluation so the two never overlap.
TRAIN_SEED_LOW = 10_000

## @brief Seeds kept queued in Unity ahead of the running episode. Unity starts the next
##        episode before Python sees the previous one end, so the queue must never run dry.
SEED_BUFFER = 4


def action_mask_from(steps) -> np.ndarray:
    """@brief obs["action_mask"] for the first agent in @p steps: 1 = enabled, branches concatenated.

    @details ML-Agents reports masks as one (n_agents, branch_size) array per branch, True = masked. Terminal
    steps carry no mask, and neither does a decision whose agent wrote none: every action is enabled then.
    """
    masks = getattr(steps, "action_mask", None)
    if not masks:
        return np.ones(ACTION_MASK_LEN, dtype=np.float32)
    return np.concatenate([~np.asarray(m[0], dtype=bool) for m in masks]).astype(np.float32)


def slice_obs(raw: np.ndarray, max_machines: int = MAX_MACHINES,
              max_jobs: int = MAX_JOBS) -> Dict[str, np.ndarray]:
    """@brief Slice a flat observation vector into the five named streams (schema v2, see config.py).

    @param max_machines, max_jobs  The row caps the player was launched with (-obsmaxmachines / -obsmaxjobs).
    """
    expected = obs_total_size(max_machines, max_jobs)
    assert raw.shape[-1] == expected, (
        f"Expected {expected} floats (row caps {max_machines} machines / {max_jobs} jobs), got "
        f"{raw.shape[-1]} -- player and env/config.py observation schemas differ (rebuild the player or "
        f"sync config.py)"
    )
    machines_end = SPATIAL_LEN + max_machines * MACHINE_FEATURES
    jobs_end = machines_end + max_jobs * JOB_FEATURES
    scalars_end = jobs_end + GLOBAL_SCALARS
    lead = raw.shape[:-1]
    return {
        "factory_grid": raw[..., :SPATIAL_LEN].reshape(
            *lead, GRID_CHANNELS, GRID_SIZE, GRID_SIZE).astype(np.float32),
        "machine_table": raw[..., SPATIAL_LEN:machines_end].reshape(
            *lead, max_machines, MACHINE_FEATURES).astype(np.float32),
        "job_table": raw[..., machines_end:jobs_end].reshape(
            *lead, max_jobs, JOB_FEATURES).astype(np.float32),
        "global_scalars": raw[..., jobs_end:scalars_end].astype(np.float32),
        "event_flags": raw[..., scalars_end:scalars_end + EVENT_FLAGS].astype(np.float32),
    }


class UnitySchedulingEnv:
    """@brief Single-agent wrapper around the ML-Agents Unity environment."""

    def __init__(self, file_name: Optional[str] = None,
                 reward_fn: Optional[RewardFunction] = None,
                 time_scale: float = 100.0, worker_id: int = 0,
                 timeout_wait: int = 300, no_graphics: bool = False,
                 decision_drain: bool = True, log_file: Optional[str] = None,
                 env_id: int = 0, seed_rng: Optional[np.random.Generator] = None,
                 capture_frame_rate: Optional[int] = 60,
                 target_frame_rate: Optional[int] = -1,
                 extra_args: Optional[list] = None,
                 scenario_generator: Optional[Callable[[int], dict]] = None,
                 obs_caps: Optional[Tuple[int, int]] = None):
        """
        @param obs_caps        Observation row caps (machine rows, job rows), passed to the player as
                               -obsmaxmachines / -obsmaxjobs. Pick them with config.obs_row_caps for the
                               largest floor the run will see; None uses the defaults (MAX_MACHINES,
                               MAX_JOBS), which fit every planned floor but pad small ones.
        @param reward_fn       Python reward function; None passes Unity's reward through.
        @param capture_frame_rate  Engine capture frame rate, as mlagents-learn sends. Without it Unity
                                   paces frames by wall-clock time and episodes simulated 2.9x slower
                                   (A/B 2026-09-14, identical results). None leaves Unity's default.
        @param target_frame_rate   Engine target frame rate (-1 = unlimited, as mlagents-learn sends).
        @param extra_args          Additional player command-line arguments, e.g. ["-decisionlogdir", dir].
        @param decision_drain  Launch the player with -rldecisiondrain (one Python step per decision).
        @param log_file        Player log path; without it Unity logs to stdout.
        @param env_id          Index within a vectorized wrapper (passed to the reward context).
        @param seed_rng        If set, every episode gets a training seed drawn from this RNG
                               (reproducible instances). None sends no seeds: Unity rebuilds the
                               factory each episode, so every episode replays the config's seed.
        @param scenario_generator  If set, called with each drawn seed to build that episode's
                                   scripted scenario (dict, ScenarioLoader schema); queued in
                                   lockstep with the same seed. Requires seed_rng. None (with a
                                   scenario separately queued via load_scenario/queue_scenarios)
                                   replays whatever scenario was last queued, every episode.
        """
        if scenario_generator is not None and seed_rng is None:
            raise ValueError("scenario_generator needs seed_rng, to draw the seed each variant is built from")
        self.engine_channel = EngineConfigurationChannel()
        self.config_channel = EpisodeConfigChannel()
        self.seed_channel = EpisodeSeedChannel()
        self.telemetry = EpisodeTelemetryChannel()

        self.obs_caps = (MAX_MACHINES, MAX_JOBS) if obs_caps is None else (int(obs_caps[0]), int(obs_caps[1]))
        additional_args = ["-obsmaxmachines", str(self.obs_caps[0]), "-obsmaxjobs", str(self.obs_caps[1])]
        if decision_drain:
            additional_args += ["-rldecisiondrain", "true"]
        if log_file is not None:
            # Must be absolute: the player does not resolve relative paths against Python's
            # working directory, and exits at startup ("Unable to open log file") if it can't open it.
            additional_args += ["-logFile", str(Path(log_file).resolve())]
        additional_args += [str(arg) for arg in (extra_args or [])]

        # Loopback-only gRPC (thesis section 7.1): upstream listens on every interface, unauthenticated.
        self.env = LoopbackUnityEnvironment(
            file_name=file_name,
            side_channels=[
                self.engine_channel,
                self.config_channel,
                self.seed_channel,
                self.telemetry,
            ],
            worker_id=worker_id,
            timeout_wait=timeout_wait,
            no_graphics=no_graphics,
            additional_args=additional_args,
        )
        self.engine_channel.set_configuration_parameters(
            time_scale=time_scale,
            capture_frame_rate=capture_frame_rate,
            target_frame_rate=target_frame_rate,
        )

        # Close the player if any startup check fails: otherwise its gRPC server thread keeps Python alive after
        # the exception, and a Slurm job hangs until its time limit (rnd02, 2026-09-26: 25 h idle on 3 nodes).
        try:
            self.env.reset()
            self.behavior_name = list(self.env.behavior_specs.keys())[0]
            self.spec = self.env.behavior_specs[self.behavior_name]
            self._policy_index, self._metrics_index = self._find_observation_indices(self.spec, self.obs_caps)
            branches = tuple(self.spec.action_spec.discrete_branches)
            if branches != ACTION_BRANCHES:
                raise RuntimeError(
                    f"Unity action branches {branches} do not match env/config.py {ACTION_BRANCHES} (job head x "
                    f"machine head). Rebuild the player with the two-branch SchedulingAgent.")
            if reward_fn is not None and self._metrics_index is None:
                raise RuntimeError(
                    f"A reward function was given but the Unity build has no '{SENSOR_NAME}' "
                    "sensor. Rebuild the player with RewardMetricsSensor."
                )
        except BaseException:
            self.env.close()
            raise

        self.reward_fn = reward_fn
        self.env_id = env_id
        self.seed_rng = seed_rng
        self.scenario_generator = scenario_generator
        ## @brief Total env steps across all envs; set by VectorizedUnityEnv for reward schedules.
        self.global_step = 0
        self.episodes_completed = 0

        self._pending_config = None
        self._prev_metrics: Optional[MetricsSnapshot] = None
        self._last_obs: Optional[Dict[str, np.ndarray]] = None   # last decision of the running episode
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_terms: Dict[str, float] = {}

    @staticmethod
    def _find_observation_indices(spec, obs_caps: Tuple[int, int]) -> Tuple[int, Optional[int]]:
        """@brief Locate the policy observation and the reward-metrics sensor by name."""
        policy_index, metrics_index = None, None
        for i, obs_spec in enumerate(spec.observation_specs):
            if obs_spec.name == SENSOR_NAME:
                metrics_index = i
            elif policy_index is None:
                policy_index = i

        shape = None if policy_index is None else tuple(spec.observation_specs[policy_index].shape)
        expected = obs_total_size(*obs_caps)
        assert shape == (expected,), (
            f"Unity VectorSensor size {shape} does not match the {expected} floats expected for row caps "
            f"{obs_caps[0]} machines / {obs_caps[1]} jobs. A player built before 2026-09-27 ignores "
            f"-obsmaxmachines / -obsmaxjobs: rebuild it."
        )
        return policy_index, metrics_index

    def send_config(self, config: dict):
        """
        Queue a config to be sent on the next reset().

        In deterministic mode (no stochastic block), Unity uses its default
        config (whatever is set in the Inspector / HeadlessBatchRunner).
        """
        self._pending_config = config

    def load_scenario(self, path):
        """@brief Replay a scripted scenario JSON (ScenarioLoader schema) every episode.

        @details Queues it once, cleared; Unity applies it from the next episode it starts and
        keeps replaying it (the queue runs dry after one item) until another scenario is queued.
        The episode already running when Python connects still uses the previous config.
        """
        self.queue_scenarios([str(Path(path).resolve())], clear=True)

    def queue_scenarios(self, items: Iterable[Union[str, dict]], clear: bool = False):
        """@brief Queue scripted scenarios; each episode Unity starts consumes one.

        @param items  Each element a dict (inline scenario) or a path string, resolved to
                      absolute here (see queue_seeds for why queueing ahead matters).
        @param clear  Empty Unity's scenario queue first.
        """
        resolved = [item if isinstance(item, dict) else str(Path(item).resolve()) for item in items]
        self.config_channel.queue_scenarios(resolved, clear=clear)

    def queue_seeds(self, seeds: Iterable[int], clear: bool = False):
        """@brief Queue instance seeds in Unity; each episode Unity starts consumes one.

        @details Messages reach Unity with the next reset() or step(). Because Unity starts
        the next episode as soon as one ends, a seed must already be queued before the
        preceding episode finishes in order to apply to it.

        @param clear  Empty Unity's queue and restart its seed index at 0 first.
        """
        self.seed_channel.queue_seeds(list(seeds), clear=clear)

    @property
    def current_metrics(self) -> Optional[MetricsSnapshot]:
        """@brief Latest snapshot of the running episode (after a rollover: the new episode's first)."""
        return self._prev_metrics

    def reset(self) -> Dict[str, np.ndarray]:
        """@brief Reset Unity and return the first observation of a fresh episode.

        @details Only needed at startup (or to apply a new config immediately); episodes
        roll over on their own inside @ref step. The episode already running when Python
        connects keeps going through reset and has no queued seed.
        """
        if self._pending_config is not None:
            self.config_channel.send_config(self._pending_config)
            self._pending_config = None
        if self.seed_rng is not None:
            self._refill_seeds_and_scenarios(SEED_BUFFER, clear=True)

        self.env.reset()
        decision, _ = self.env.get_steps(self.behavior_name)
        return self._begin_episode(self._wait_for_decision(decision))

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        """@brief Apply one scheduling action (job head, machine head) and advance to the next decision.

        @return (obs, reward, done, info). @c info["reward_terms"] holds this step's named
                reward terms. When @c done is True the episode just ended: @c reward is its
                final reward, @c info["episode"] summarizes it (including "truncated" — a
                deliberate time-cap cutoff, not a real terminal), @c info["terminal_obs"] is
                that ended episode's own last observation, and @c obs is already the first
                observation of the next episode.
        """
        self.env.set_actions(
            self.behavior_name,
            ActionTuple(discrete=np.asarray(action, dtype=np.int32).reshape(1, len(ACTION_BRANCHES))),
        )
        self.env.step()
        decision, terminal = self.env.get_steps(self.behavior_name)
        while len(decision) == 0 and len(terminal) == 0:
            self.env.step()
            decision, terminal = self.env.get_steps(self.behavior_name)

        # A terminal step and the next episode's first decision can arrive in the same batch.
        done = len(terminal) > 0
        steps = terminal if done else decision
        curr = self._extract_metrics(steps)
        reward, terms = self._compute_reward(curr, float(steps.reward[0]), done)

        self._episode_return += reward
        self._episode_length += 1
        for name, value in terms.items():
            self._episode_terms[name] = self._episode_terms.get(name, 0.0) + value

        info = {"reward_terms": terms, "dt": elapsed_sim_time(self._prev_metrics, curr)}
        if not done:
            self._prev_metrics = curr
            self._last_obs = self._extract_obs(decision)
            return self._last_obs, reward, False, info

        # Unity has already auto-reset by now: decision (if present) is the NEW episode's first
        # frame, not a continuation of the one that just ended. terminal_obs stands in for the
        # ended episode's final state, to bootstrap a truncated (not terminated) episode's value.
        # The terminal step's own observation is useless for that: the player has already set
        # episodeActive = false when it collects it, so SchedulingAgent.CollectObservations pads
        # it with zeros. The last decision's observation (one decision before the cap) is used
        # instead; it falls back to the padded one only if step() runs before reset().
        info["telemetry"] = self.telemetry.pop_payload()
        info["episode"] = self._episode_summary(curr, interrupted=bool(terminal.interrupted[0]),
                                                telemetry=info["telemetry"])
        info["terminal_obs"] = self._last_obs if self._last_obs is not None else self._extract_obs(terminal)
        self.episodes_completed += 1
        if self.seed_rng is not None:
            # Unity already consumed a seed (and scenario, if any) for the episode that just
            # started; replace it so the buffer stays full ahead of the next one.
            self._refill_seeds_and_scenarios(1, clear=False)
        next_obs = self._begin_episode(self._wait_for_decision(decision))
        return next_obs, reward, True, info

    def _draw_training_seeds(self, count: int) -> list:
        return self.seed_rng.integers(TRAIN_SEED_LOW, EpisodeSeedChannel.MAX_SEED, size=count).tolist()

    def _refill_seeds_and_scenarios(self, count: int, clear: bool):
        """@brief Queue @p count more training seeds, and — with scenario_generator set — the
        matching scenario variants, so both queues advance together one item per episode."""
        seeds = self._draw_training_seeds(count)
        self.queue_seeds(seeds, clear=clear)
        if self.scenario_generator is not None:
            self.queue_scenarios([self.scenario_generator(seed) for seed in seeds], clear=clear)

    def _compute_reward(self, curr: Optional[MetricsSnapshot], unity_reward: float,
                        done: bool) -> Tuple[float, Dict[str, float]]:
        if self.reward_fn is None or curr is None or self._prev_metrics is None:
            return unity_reward, {}
        ctx = RewardContext(global_step=self.global_step, episode=self.episodes_completed,
                            env_id=self.env_id, done=done)
        return self.reward_fn(self._prev_metrics, curr, ctx)

    def _wait_for_decision(self, decision):
        """@brief Advance Unity (no actions pending) until the agent requests a decision."""
        while len(decision) == 0:
            self.env.step()
            decision, _ = self.env.get_steps(self.behavior_name)
        return decision

    def _begin_episode(self, decision) -> Dict[str, np.ndarray]:
        self._prev_metrics = self._extract_metrics(decision)
        if self.reward_fn is not None and self._prev_metrics is not None:
            self.reward_fn.reset(self._prev_metrics)
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_terms = {}
        self._last_obs = self._extract_obs(decision)
        return self._last_obs

    def _episode_summary(self, final: Optional[MetricsSnapshot], interrupted: bool,
                         telemetry: Optional[dict] = None) -> dict:
        result = (telemetry or {}).get("result") or {}
        events = (telemetry or {}).get("events") or []
        summary = {
            "return": self._episode_return,
            "length": self._episode_length,
            "interrupted": interrupted,
            "reward_terms": dict(self._episode_terms),
            # Hash of the config Unity actually applied, and of config + initial jobs (ConfigFingerprint.cs);
            # None from a player built before 2026-09-29. applied_config is the canonical config text, only the
            # first time this player applies a given hash (train.py appends it to applied_configs.jsonl).
            "config_hash": result.get("configHash"),
            "instance_hash": result.get("instanceHash"),
            "applied_config": result.get("appliedConfig"),
            # Disruptions this episode. AGV totals come from the result (None from a player built before AGV
            # breakdowns existed); machine failures are counted from the telemetry events.
            "machine_failures": sum(1 for e in events if e.get("type") == "machine_failure") if telemetry else None,
            "agv_failures": result.get("agvFailures"),
            "agv_repair_time": result.get("agvRepairTime"),
            "agv_blocked_by_failure_time": result.get("agvBlockedByFailureTime"),
        }
        if final is not None:
            exited = final.jobs_exited
            summary.update(
                seed=int(final.episode_seed),
                seed_index=int(final.episode_seed_index),
                makespan=final.sim_time,
                jobs_exited=exited,
                jobs_total=final.jobs_total,
                total_flow_time=final.flow_time_exited_sum,
                mean_flow_time=final.flow_time_exited_sum / exited if exited else float("nan"),
                deadlock=bool(final.deadlock),
                timed_out=bool(final.timed_out),
                truncated=bool(final.truncated),
            )
        return summary

    def _extract_obs(self, steps) -> Dict[str, np.ndarray]:
        obs = slice_obs(steps.obs[self._policy_index][0], *self.obs_caps)
        obs["action_mask"] = action_mask_from(steps)
        return obs

    def _extract_metrics(self, steps) -> Optional[MetricsSnapshot]:
        if self._metrics_index is None:
            return None
        return MetricsSnapshot(steps.obs[self._metrics_index][0])

    def close(self):
        self.env.close()


class VectorizedUnityEnv:
    """@brief Manages multiple Unity instances for parallel data collection.

    @details Each Unity player is its own process, so envs are stepped concurrently from a
    thread pool: the Python side of a step mostly waits on the player (the ML-Agents
    communicator blocks on a pipe, releasing the GIL). Each env is only ever driven by one
    thread at a time, and results keep env order. Construction stays sequential so env i
    always gets worker_id base + i.
    """

    def __init__(self, num_envs: int, file_name: Optional[str] = None,
                 reward_spec=None, time_scale: float = 100.0,
                 base_worker_id: int = 0, timeout_wait: int = 300,
                 no_graphics: bool = False, decision_drain: bool = True,
                 log_dir: Optional[str] = None, train_seed: Optional[int] = None,
                 parallel: bool = True,
                 scenario_generator: Optional[Callable[[int], dict]] = None,
                 obs_caps: Optional[Tuple[int, int]] = None):
        """
        @param obs_caps     Observation row caps for every player (see UnitySchedulingEnv).
        @param reward_spec  Reward spec path or dict, or a @ref rewards.LoadedReward. Each env
                            gets its own reward instance. None passes Unity's reward through.
        @param log_dir      If set, instance i writes its player log to log_dir/Player-i.log.
        @param train_seed   If set, instance seeds come from per-env RNGs derived from it, so a
                            run's training instances are reproducible. None sends no seeds, so
                            every episode replays the config's own seed.
        @param parallel     Step and reset envs concurrently (default); False runs them one by one.
        @param scenario_generator  If set, each env gets a fresh scripted-scenario variant every
                                   episode, built by calling this with that episode's drawn seed
                                   (each env draws from its own per-env RNG, so envs see different
                                   variants). Requires train_seed.
        """
        if scenario_generator is not None and train_seed is None:
            raise ValueError("scenario_generator needs --train-seed, to draw the seed each variant is built from")

        loaded = None
        if reward_spec is not None:
            loaded = reward_spec if isinstance(reward_spec, LoadedReward) else load_reward(reward_spec)

        self.envs = []
        try:
            for i in range(num_envs):
                self.envs.append(UnitySchedulingEnv(
                    file_name=file_name,
                    reward_fn=loaded.build() if loaded is not None else None,
                    time_scale=time_scale,
                    worker_id=base_worker_id + i,
                    timeout_wait=timeout_wait,
                    no_graphics=no_graphics,
                    decision_drain=decision_drain,
                    log_file=None if log_dir is None else Path(log_dir) / f"Player-{i}.log",
                    env_id=i,
                    seed_rng=None if train_seed is None else np.random.default_rng([train_seed, i]),
                    scenario_generator=scenario_generator,
                    obs_caps=obs_caps,
                ))
        except BaseException:
            # Don't leave already-launched players running when a later one fails to start.
            for env in self.envs:
                env.close()
            raise

        self.num_envs = num_envs
        self.global_step = 0
        self._pool = (ThreadPoolExecutor(max_workers=num_envs, thread_name_prefix="unity-env")
                      if parallel and num_envs > 1 else None)

    def send_configs(self, configs: list):
        """Push one config per env. configs[i] applies to envs[i]."""
        for env, cfg in zip(self.envs, configs):
            if cfg is not None:
                env.send_config(cfg)

    def send_config_all(self, config: dict):
        """Push the same config to all envs."""
        for env in self.envs:
            env.send_config(config)

    def load_scenario_all(self, path):
        """Replay the same scripted scenario in every env (see UnitySchedulingEnv.load_scenario)."""
        for env in self.envs:
            env.load_scenario(path)

    def reset(self):
        results = self._map(lambda env: env.reset(), self.envs)
        return self._stack_obs(results), [{}] * self.num_envs

    def step(self, actions):
        """@brief Step every env once.

        @details Episode rollover happens inside each env, so a done env's returned
        observation already belongs to its next episode.
        """
        self.global_step += self.num_envs
        for env in self.envs:
            env.global_step = self.global_step
        results = self._map(lambda env, action: env.step(action), self.envs, actions)
        obs_list, rewards, dones, infos = zip(*results)
        truncateds = [bool(info["episode"]["truncated"]) if done else False
                     for done, info in zip(dones, infos)]

        return (
            self._stack_obs(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(dones),
            np.array(truncateds, dtype=bool),
            list(infos),
        )

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        for env in self.envs:
            env.close()

    def _map(self, fn, *iterables) -> list:
        """@brief Apply @p fn per env, concurrently when a pool exists, preserving env order."""
        if self._pool is None:
            return list(map(fn, *iterables))
        return list(self._pool.map(fn, *iterables))

    @staticmethod
    def _stack_obs(obs_list):
        keys = obs_list[0].keys()
        return {k: np.stack([o[k] for o in obs_list]) for k in keys}
