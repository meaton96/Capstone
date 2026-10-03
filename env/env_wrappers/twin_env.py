"""
@file twin_env.py
@brief The event-based twin (env/des_twin) behind UnitySchedulingEnv's interface, so env/train.py trains on it by
       switching the env class (--twin).

@details
Each step applies one (job head, machine head) action and advances the twin to its next decision, as the player
does under -rldecisiondrain. The floor is a des_floor.json exported by a player built after 2026-10-01 (it
carries the observation frame); one export serves every episode on that layout and fleet size. Each episode's
jobs come from the scenario generator (or a fixed scenario) resolved in Python (des_twin.scenario), seeded as
UnitySchedulingEnv seeds them, so the twin and the player can be given the same training instances.

What is kept identical to the Unity wrapper: per-episode training seeds (TRAIN_SEED_LOW and up), the warm-up
under the scenario's dispatchingRule, the steady-state time cap (truncation, not termination), rewards computed
from reward-metrics snapshots (env/rewards), the episode summary in info["episode"], auto-reset inside step(),
and info["terminal_obs"], the observation of the ended episode's last decision (the Unity wrapper keeps the same
one, since the player's own terminal observation is padded with zeros).

Not modelled by the twin, and refused per scenario: machine and AGV failures, Poisson arrivals, machine
flexibility, tiled floors (des_twin.scenario / des_twin.floor raise).
"""
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from config import ACTION_BRANCHES, JOB_HEAD_RULES, MACHINE_HEAD_RULES, MAX_JOBS, MAX_MACHINES
from des_twin import Floor
from des_twin.engine import Twin, TwinConfig
from des_twin.observation import ObservationBuilder
from des_twin.scenario import episode_settings, resolve_jobs
from rewards import (LoadedReward, MetricsSnapshot, RewardContext, RewardFunction, elapsed_sim_time, load_reward,
                     window_outcomes)

## @brief Same split as env_wrappers.unity_env (kept in sync by tests/test_twin_env.py): seeds below are evaluation.
TRAIN_SEED_LOW = 10_000
MAX_SEED = 1 << 24


def instance_seed_rng(train_seed: int, env_index: int, seed_stream: int = 0) -> np.random.Generator:
    """@brief Same as env_wrappers.unity_env.instance_seed_rng (kept in sync by tests/test_twin_env.py)."""
    key = [train_seed, env_index] if seed_stream == 0 else [train_seed, env_index, seed_stream]
    return np.random.default_rng(key)

## @brief Unity's FactoryOrchestrator.MAX_EPISODE_SIM_SECONDS: an episode past it ends as timed out.
MAX_EPISODE_SIM_SECONDS = 100_000.0


class TwinSchedulingEnv:
    """@brief One twin episode stream with UnitySchedulingEnv's step()/reset() contract."""

    def __init__(self, floor: Union[str, Path, Floor], transport: str = "kinematic",
                 reward_fn: Optional[RewardFunction] = None, env_id: int = 0,
                 seed_rng: Optional[np.random.Generator] = None,
                 scenario_generator: Optional[Callable[[int], dict]] = None,
                 scenario: Optional[dict] = None,
                 obs_caps: Optional[Tuple[int, int]] = None,
                 instant_fleet: Optional[int] = None):
        """
        @param floor        des_floor.json path (or a loaded Floor) for the layout and fleet size trained on.
        @param transport    "instant" (DES-0), "geometric" (DES-1g) or "kinematic" (DES-1k).
        @param seed_rng     Draws each episode's training seed; with scenario_generator the episode's scenario is
                            scenario_generator(seed). Without a generator, `scenario` is replayed every episode.
        @param obs_caps     (machine rows, job rows); None uses MAX_MACHINES / MAX_JOBS.
        @param instant_fleet  AGVs reported (parked, idle) in DES-0 observations; None: the floor's fleet.
        """
        if scenario_generator is not None and seed_rng is None:
            raise ValueError("scenario_generator needs seed_rng, to draw the seed each variant is built from")
        if scenario_generator is None and scenario is None:
            raise ValueError("the twin needs a scenario or a scenario_generator (it has no default floor job set)")
        self.floor = floor if isinstance(floor, Floor) else Floor.load(floor)
        self.transport = transport
        self.reward_fn = reward_fn
        self.env_id = env_id
        self.seed_rng = seed_rng
        self.scenario_generator = scenario_generator
        self.scenario = scenario
        self.obs_caps = (MAX_MACHINES, MAX_JOBS) if obs_caps is None else (int(obs_caps[0]), int(obs_caps[1]))
        fleet = len(self.floor.agvs) if instant_fleet is None else int(instant_fleet)
        self.obs_builder = ObservationBuilder(self.floor, *self.obs_caps, instant_fleet=fleet)
        self.global_step = 0
        self.episodes_completed = 0
        self._seed_index = -1
        self._twin: Optional[Twin] = None
        self._gen = None
        self._decision = None
        self._last_obs: Optional[Dict[str, np.ndarray]] = None   # last decision of the running episode
        self._seed = -1
        self._prev_metrics: Optional[MetricsSnapshot] = None
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_terms: Dict[str, float] = {}

    # ── Episodes ────────────────────────────────────────────────────────────────────────────────────

    def _check_scenario(self, scenario: dict):
        data = self.floor.data
        if scenario.get("layout") not in (None, data.get("layout")):
            raise ValueError(f"scenario layout {scenario['layout']!r} but the floor export is layout "
                             f"{data.get('layout')!r}")
        if scenario.get("parkingMethod") not in (None, data.get("parking_method")):
            raise ValueError(f"scenario parkingMethod {scenario['parkingMethod']!r} but the floor export uses "
                             f"{data.get('parking_method')!r}")
        agvs = scenario.get("agvCount")
        if agvs is not None and int(agvs) != len(self.floor.agvs):
            raise ValueError(f"scenario agvCount {agvs} but the floor export has {len(self.floor.agvs)} AGVs "
                             "(parking bays depend on the fleet size: export the floor at that size)")
        # Keys the twin does not model: refuse values it would otherwise silently ignore (audit P4.2).
        for key, field in (("agvMoveSpeed", "speed"), ("agvHandshakeDuration", "handshake")):
            value = scenario.get(key)
            if value is not None and float(value) != float(data["agv"][field]):
                raise ValueError(f"scenario {key} {value} but the floor export's AGVs use {field} "
                                 f"{data['agv'][field]} (the twin takes AGV motion from the export)")
        if float(scenario.get("travelPrice") or 0.0) > 0:
            raise ValueError(f"scenario travelPrice {scenario['travelPrice']}: the twin scores TECT without a "
                             "travel price")
        if scenario.get("ioDocks") not in (None, "corner"):
            raise ValueError(f"scenario ioDocks {scenario['ioDocks']!r}: the twin only models the default corner "
                             "docks (the floor export does not record its I/O docks)")
        for key, field in (("routingTrigger", "routing_trigger"), ("reservationProtocol", "reservation_protocol")):
            if scenario.get(key) is not None and data.get(field) and scenario[key] != data[field]:
                raise ValueError(f"scenario {key} {scenario[key]!r} but the floor export used {data[field]!r}")

    def _new_twin(self):
        if self.seed_rng is not None:
            self._seed = int(self.seed_rng.integers(TRAIN_SEED_LOW, MAX_SEED))
            self._seed_index += 1
        scenario = self.scenario_generator(self._seed) if self.scenario_generator is not None else self.scenario
        self._check_scenario(scenario)
        warmup, cap, rule = episode_settings(scenario)
        cfg = TwinConfig(rule=rule, transport=self.transport, routing_trigger=scenario.get("routingTrigger"),
                         max_sim_seconds=MAX_EPISODE_SIM_SECONDS, warmup_seconds=warmup,
                         episode_duration_seconds=cap)
        return Twin(self.floor, resolve_jobs(scenario, self.floor), cfg)

    def _start_episode(self) -> Dict[str, np.ndarray]:
        """Starts episodes until one reaches a decision (one that ends inside its warm-up gives the agent nothing
        to do; the player would not ask for an action either)."""
        for _ in range(100):
            self._twin = self._new_twin()
            self._gen = self._twin.agent_decisions()
            try:
                self._decision = next(self._gen)
            except StopIteration:
                continue
            self._prev_metrics = self._metrics()
            self._first_metrics = self._prev_metrics      # start of the agent window (window_outcomes)
            if self.reward_fn is not None:
                self.reward_fn.reset(self._prev_metrics)
            self._episode_return, self._episode_length, self._episode_terms = 0.0, 0, {}
            self._last_obs = self._obs()
            return self._last_obs
        raise RuntimeError("100 episodes in a row ended without an agent decision: check the warm-up and the cap")

    def reset(self) -> Dict[str, np.ndarray]:
        return self._start_episode()

    def _metrics(self) -> MetricsSnapshot:
        values = self._twin.metrics()
        values["decision_count"] = self._twin.decisions
        values["episode_seed"] = self._seed
        values["episode_seed_index"] = self._seed_index if self.seed_rng is not None else -1
        return MetricsSnapshot.from_dict(values)

    def _obs(self) -> Dict[str, np.ndarray]:
        return self.obs_builder.build(self._twin, self._decision)

    # ── Stepping ────────────────────────────────────────────────────────────────────────────────────

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        """@brief Apply (job head, machine head) and run to the next decision. Same return contract as
        UnitySchedulingEnv.step: on done, obs is the next episode's first observation."""
        job_head, machine_head = (int(a) for a in np.asarray(action).reshape(len(ACTION_BRANCHES)))
        rule = (JOB_HEAD_RULES[job_head], MACHINE_HEAD_RULES[machine_head])
        try:
            self._decision = self._gen.send(rule)
            done = False
        except StopIteration:
            done = True
        curr = self._metrics()
        reward, terms = self._compute_reward(curr, done)
        self._episode_return += reward
        self._episode_length += 1
        for name, value in terms.items():
            self._episode_terms[name] = self._episode_terms.get(name, 0.0) + value

        info = {"reward_terms": terms, "dt": elapsed_sim_time(self._prev_metrics, curr)}
        if not done:
            self._prev_metrics = curr
            self._last_obs = self._obs()
            return self._last_obs, reward, False, info
        info["telemetry"] = None
        info["episode"] = self._episode_summary(curr)
        info["terminal_obs"] = self._last_obs
        info["twin_summary"] = self._twin.summary()
        self.episodes_completed += 1
        return self._start_episode(), reward, True, info

    def _compute_reward(self, curr: MetricsSnapshot, done: bool) -> Tuple[float, Dict[str, float]]:
        if self.reward_fn is None or self._prev_metrics is None:
            return 0.0, {}
        ctx = RewardContext(global_step=self.global_step, episode=self.episodes_completed,
                            env_id=self.env_id, done=done)
        return self.reward_fn(self._prev_metrics, curr, ctx)

    def _episode_summary(self, final: MetricsSnapshot) -> dict:
        exited = final.jobs_exited
        return {
            "return": self._episode_return,
            "length": self._episode_length,
            "interrupted": False,
            "reward_terms": dict(self._episode_terms),
            "config_hash": None,
            "instance_hash": None,
            "applied_config": None,
            "machine_failures": 0,
            "agv_failures": None,
            "agv_repair_time": None,
            "agv_blocked_by_failure_time": None,
            "seed": int(final.episode_seed),
            "seed_index": int(final.episode_seed_index),
            "makespan": final.sim_time,
            "jobs_exited": exited,
            "jobs_total": final.jobs_total,
            "total_flow_time": final.flow_time_exited_sum,
            "mean_flow_time": final.flow_time_exited_sum / exited if exited else float("nan"),
            "deadlock": False,
            "timed_out": bool(final.timed_out),
            "truncated": bool(final.truncated),
            "tick_error": False,
            **window_outcomes(getattr(self, "_first_metrics", None), final),
        }

    @property
    def current_metrics(self) -> Optional[MetricsSnapshot]:
        return self._prev_metrics

    def close(self):
        self._gen = None


class VectorizedTwinEnv:
    """@brief VectorizedUnityEnv's interface over twin envs, stepped in this process one after another."""

    def __init__(self, num_envs: int, floor: Union[str, Path], transport: str = "kinematic", reward_spec=None,
                 train_seed: Optional[int] = None, scenario_generator: Optional[Callable[[int], dict]] = None,
                 scenario: Optional[dict] = None, obs_caps: Optional[Tuple[int, int]] = None,
                 instant_fleet: Optional[int] = None, seed_stream: int = 0):
        if scenario_generator is not None and train_seed is None:
            raise ValueError("scenario_generator needs --train-seed, to draw the seed each variant is built from")
        loaded = None
        if reward_spec is not None:
            loaded = reward_spec if isinstance(reward_spec, LoadedReward) else load_reward(reward_spec)
        shared = Floor.load(floor)   # route / hop caches are pure functions of the floor: share them
        self.envs = [TwinSchedulingEnv(
            shared, transport=transport, reward_fn=loaded.build() if loaded is not None else None, env_id=i,
            seed_rng=None if train_seed is None else instance_seed_rng(train_seed, i, seed_stream),
            scenario_generator=scenario_generator, scenario=scenario, obs_caps=obs_caps,
            instant_fleet=instant_fleet) for i in range(num_envs)]
        self.num_envs = num_envs
        self.global_step = 0

    def reset(self):
        return self._stack_obs([env.reset() for env in self.envs]), [{}] * self.num_envs

    def step(self, actions):
        self.global_step += self.num_envs
        for env in self.envs:
            env.global_step = self.global_step
        obs_list, rewards, dones, infos = zip(*(env.step(a) for env, a in zip(self.envs, actions)))
        truncateds = [bool(info["episode"]["truncated"]) if done else False for done, info in zip(dones, infos)]
        return (self._stack_obs(obs_list), np.array(rewards, dtype=np.float32), np.array(dones),
                np.array(truncateds, dtype=bool), list(infos))

    def close(self):
        for env in self.envs:
            env.close()

    @staticmethod
    def _stack_obs(obs_list):
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0].keys()}
