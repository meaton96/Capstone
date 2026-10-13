"""
@file paired_slot_env.py
@brief Same-future difference advantage per slot (2026-10-07; credit trace item 1, handoff fix 5): slot actions as
       SlotActionEnv, and at each slot the advantage of the chosen pair against a default pair, measured on the same
       redrawn futures (common random numbers).

@details One slot choice moves the discounted return by about 1/20 to 1/40 of its spread over redrawn futures, and a
state-only critic cannot remove that spread (docs/experiments/review_1007/CREDIT_ASSIGNMENT_TRACE_1007.md). Pairing
removes it: from the identical state, the taken pair and the default pair play the slot on the same future, then the
default pair plays a tail of `tail_seconds`, and the difference of the two discounted returns is the slot's advantage
(about 46x less variance than unpaired returns at a 3,000 s horizon, measured). A single future predicts the sign on
other futures only about half the time, so the target is the mean over `futures` redrawn futures.

What it estimates: A(s, a) = Q_d(s, a) - Q_d(s, d), the value of playing a for one slot instead of the default d, with
d afterwards. That is a policy-improvement step over the default pair (the action-prior branch learns deviations from
MDD-TECT), not the advantage under the current policy's own continuation.

How the state is reproduced: the twin runs on nested generators and cannot be copied, but it is deterministic given
the scenario and the actions. A branch replays the episode from its start: same scenario up to the slot start, the
recorded pair of every earlier slot, then the branch's own choices. A redrawn future keeps the instance's jobs that
arrived by the slot start and splices in the jobs arriving later from another generator seed (the splice of
docs/experiments/review_1007/scripts/branch.py, checked there to reach the identical state). Every branch checks that
it reaches the slot start at the same sim time and decision count as the real run; a branch that does not is
reported as a mismatch and the step keeps its GAE advantage.

info["paired_advantage"] is the mean difference (reward units, discounted to the slot start with gamma_per_second)
or None when it was not computed (mismatch); it is exactly 0.0 when the chosen pair is the default (nothing to
simulate). Twin only; the cost is 2 x futures replays per non-default slot.
"""
import copy
import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from config import JOB_HEAD_RULES, MACHINE_HEAD_RULES
from env_wrappers.slot_env import SlotActionEnv
from env_wrappers.twin_env import MAX_SEED, TRAIN_SEED_LOW, TwinSchedulingEnv


def splice_future(scenario: dict, future: dict, t0: float) -> dict:
    """@brief @p scenario with its jobs arriving after @p t0 replaced by @p future's (ids offset by 100,000, so they
    cannot collide); everything else (warm-up, cap, AGV schedule, rules) is the instance's own."""
    out = copy.deepcopy(scenario)
    keep = [j for j in scenario["jobs"] if j["arrivalTime"] <= t0]
    add = []
    for j in future["jobs"]:
        if j["arrivalTime"] > t0 + 1.0:
            j = copy.deepcopy(j)
            j["id"] = int(j["id"]) + 100_000
            add.append(j)
    out["jobs"] = keep + add
    return out


class _BranchTwin(TwinSchedulingEnv):
    """A queued twin env for branch replays: no observations (they are not needed) and no auto-restart at the
    episode end."""

    def _obs(self):
        return None

    def run_once(self, seed: int, scenario: dict):
        self.queue_seeds([seed], clear=True)
        self.queue_scenarios([scenario], clear=True)
        self._started = False
        return self.reset()

    def _start_episode(self):
        if getattr(self, "_started", False):
            return None
        self._started = True
        return super()._start_episode()


class PairedSlotEnv(SlotActionEnv):
    """@brief SlotActionEnv over a TwinSchedulingEnv that also reports the same-future paired advantage per slot."""

    def __init__(self, env: TwinSchedulingEnv, slot_seconds: float, gamma_per_second: float,
                 default_action: Tuple[int, int], futures: int = 2, tail_seconds: float = 3600.0):
        """
        @param env             The (non-queued) TwinSchedulingEnv the agent plays.
        @param gamma_per_second  Discount per simulated second for the branch returns (training's gamma_s).
        @param default_action  (job head, machine head) of the default pair (e.g. MDD-TECT).
        @param futures         Redrawn futures per slot; 0 = the instance's own future only (one paired sample).
        @param tail_seconds    Default-pair tail after the slot, before the branch stops.
        """
        if gamma_per_second is None:
            raise ValueError("the paired advantage discounts over simulated time: needs gamma_per_second")
        if futures > 0 and env.scenario_generator is None:
            raise ValueError("redrawn futures need the env's scenario generator (pass futures=0 for a fixed scenario)")
        super().__init__(env, slot_seconds, gamma_per_second)
        self.default_action = (int(default_action[0]), int(default_action[1]))
        self.futures = int(futures)
        self.tail_seconds = float(tail_seconds)
        branch_reward = copy.deepcopy(env.reward_fn)
        self._branch = _BranchTwin(env.floor, transport=env.transport, reward_fn=branch_reward,
                                   obs_caps=env.obs_caps, queued=True)
        self._history: Dict[int, Tuple[int, int]] = {}
        self._t_start = 0.0
        self._scenario_cache: Tuple[Optional[int], Optional[dict]] = (None, None)
        ## @brief Counters for logging: branch runs, mismatched branches.
        self.branch_runs = 0
        self.mismatches = 0

    def _start_slots(self):
        super()._start_slots()
        self._t_start = self._now()
        self._history = {}

    def _scenario(self) -> dict:
        seed = int(self.env._seed)
        if self._scenario_cache[0] != seed:
            gen = self.env.scenario_generator
            self._scenario_cache = (seed, gen(seed) if gen is not None else self.env.scenario)
        return self._scenario_cache[1]

    def _branch_return(self, scenario: dict, g0: int, first: Tuple[int, int], t0: float,
                       decisions0: int) -> Optional[float]:
        """Discounted return from t0 to the end of slot g0 plus the tail: earlier slots replay the history, slot g0
        plays @p first, later slots the default. None if the replay does not reach the slot start identically."""
        env = self._branch
        env.run_once(int(self.env._seed), scenario)
        self.branch_runs += 1
        boundary, grid = self._t_start + self.slot_seconds, 0
        t_stop, total, reached = math.inf, 0.0, False
        while True:
            m = env.current_metrics
            t = float(m.sim_time)
            while boundary <= t:          # SlotActionEnv's grid, with the same float steps
                boundary += self.slot_seconds
                grid += 1
            if grid == g0 and not reached:
                if t != t0 or env._twin.decisions != decisions0:
                    self.mismatches += 1
                    return None
                reached, t_stop = True, boundary + self.tail_seconds
            if grid > g0 and not reached:     # skipped past the slot start: the replay diverged
                self.mismatches += 1
                return None
            if t >= t_stop:
                return total
            if grid < g0:
                action = self._history.get(grid, self.default_action)
            else:
                action = first if grid == g0 else self.default_action
            _, reward, done, _ = env.step(action)
            if reached:
                total += self.gamma_per_second ** (t - t0) * float(reward)
            if done:
                return total if reached else None

    def paired_advantage(self, action: Tuple[int, int]) -> Optional[float]:
        """@brief Mean over futures of G(action, then default) - G(default) from the current slot start."""
        if action == self.default_action:
            return 0.0
        scenario = self._scenario()
        t0 = self._now()
        decisions0 = int(self.env._twin.decisions)
        g0 = self._grid
        if self.futures == 0:
            worlds = [scenario]
        else:
            rng = np.random.default_rng([int(self.env._seed), g0, 7919])
            gen = self.env.scenario_generator
            worlds = [splice_future(scenario, gen(int(rng.integers(TRAIN_SEED_LOW, MAX_SEED))), t0)
                      for _ in range(self.futures)]
        diffs = []
        for world in worlds:
            a = self._branch_return(world, g0, action, t0, decisions0)
            d = self._branch_return(world, g0, self.default_action, t0, decisions0)
            if a is None or d is None:
                return None
            diffs.append(a - d)
        return float(np.mean(diffs))

    def step(self, action):
        action = tuple(int(x) for x in np.asarray(action).reshape(2))
        self._history[self._grid] = action
        adv = self.paired_advantage(action)
        obs, total, done, info = super().step(action)
        return obs, total, done, dict(info, paired_advantage=adv)


def default_action_for(pair: str) -> Tuple[int, int]:
    """@brief (job head, machine head) of a "JOB-MACHINE" pair name."""
    job, machine = pair.split("-")
    return JOB_HEAD_RULES.index(job), MACHINE_HEAD_RULES.index(machine)
