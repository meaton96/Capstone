"""@file test_unity_env.py
@brief UnitySchedulingEnv logic that runs without a player (the wrapper is built with __new__ and stubbed)."""
from types import SimpleNamespace

import numpy as np

from env_wrappers.unity_env import UnitySchedulingEnv


class _ScriptedEnv(UnitySchedulingEnv):
    """Episodes are a list of (seed_index, decisions); step() walks them like Unity's auto-reset."""

    def __init__(self, episodes):
        self.episodes, self.ep, self.t, self.actions = episodes, 0, 0, []
        self.seed_rng = np.random.default_rng(0)
        self.episodes_completed = 0

    @property
    def current_metrics(self):
        return SimpleNamespace(episode_seed_index=self.episodes[self.ep][0])

    def step(self, action):
        self.actions.append(tuple(action))
        self.t += 1
        if self.t < self.episodes[self.ep][1]:
            return {"ep": self.ep}, 0.0, False, {}
        self.ep, self.t = self.ep + 1, 0
        self.episodes_completed += 1
        return {"ep": self.ep}, -1.0, True, {}


def test_unseeded_startup_episode_is_played_out_and_dropped():
    env = _ScriptedEnv([(-1, 92), (0, 400), (1, 400)])
    obs = env._skip_unseeded_episodes({"ep": 0})
    assert obs == {"ep": 1}
    assert env.current_metrics.episode_seed_index == 0
    assert len(env.actions) == 92 and set(env.actions) == {(0, 0)}
    assert env.episodes_completed == 0


def test_a_seeded_first_episode_is_left_alone():
    env = _ScriptedEnv([(0, 400)])
    assert env._skip_unseeded_episodes({"ep": 0}) == {"ep": 0}
    assert env.actions == []


class _Steps(list):
    pass


class _FakeMlEnv:
    """get_steps returns (decision, terminal); the script lists (n_decisions, n_terminals) per env.step()."""

    def __init__(self, script):
        self.script = list(script)

    def step(self):
        self.now = self.script.pop(0)

    def get_steps(self, _behavior):
        d, t = self.now
        return _Steps([0] * d), _Steps([0] * t)


def test_a_decisionless_episode_drops_its_telemetry_and_replaces_its_seed():
    env = UnitySchedulingEnv.__new__(UnitySchedulingEnv)
    env.env_id, env.behavior_name = 0, "b"
    env.env = _FakeMlEnv([(0, 0), (0, 1), (1, 0)])
    popped, refills = [], []
    env.telemetry = SimpleNamespace(pop_payload=lambda: popped.append(1))
    env.seed_rng = np.random.default_rng(0)
    env._refill_seeds_and_scenarios = lambda count, clear: refills.append((count, clear))
    decision = env._wait_for_decision(_Steps())
    assert len(decision) == 1
    assert popped == [1] and refills == [(1, False)]
