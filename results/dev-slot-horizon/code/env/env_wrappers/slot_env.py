"""
@file slot_env.py
@brief Slot actions (2026-10-05): the agent picks a (job head, machine head) rule pair once per time slot, and the
       pair is held for every decision until the slot ends, as the switching oracle does (env/switch_oracle.py).

@details eval-due-twin found that per-decision actions let the policy mix rules decision by decision (ATC, SPT, MDD,
SRT, EDD about 20% each), an incoherent priority order that loses to one fixed pair by 11-17%. With slots, a policy
that keeps one pair reproduces that fixed rule exactly, and each agent step's effect on tardiness is large enough to
learn from.

SlotActionEnv wraps one env with the UnitySchedulingEnv / TwinSchedulingEnv interface (step(action) -> obs, reward,
done, info; current_metrics.sim_time; auto-reset on done):
  - slots of `slot_seconds` start at the first agent decision of the episode (the agent window), like the oracle's
    stages; the agent acts at the first decision at or after each boundary;
  - within a slot the held action is replayed at every decision; the slot's reward is the decision rewards discounted
    to the slot start, sum_i gamma_s^(tau_i - tau_0) r_i, and info["dt"] is the slot's total duration, so the SMDP
    discounting in RolloutBuffer stays exact at slot level (per-decision gamma when gamma_per_second is None);
  - the observation at a slot start has an all-ones action mask: the held pair applies to routing and dispatch
    decisions alike, so both heads always matter;
  - an episode end returns at once (done=True) with the inner info (episode summary, terminal_obs).
Everything else (queue_seeds, queue_scenarios, current_metrics, close, ...) is passed through, so evaluate.py can wrap
its env too: a fixed pair through the wrapper gives exactly the fixed-pair run.
"""
from typing import Optional

import numpy as np


class SlotActionEnv:
    def __init__(self, env, slot_seconds: float, gamma_per_second: Optional[float] = None):
        if slot_seconds <= 0:
            raise ValueError("slot_seconds must be > 0")
        self.env = env
        self.slot_seconds = float(slot_seconds)
        self.gamma_per_second = gamma_per_second
        self._boundary = None

    def __getattr__(self, name):      # queue_seeds, current_metrics, close, episodes_completed, ...
        return getattr(self.env, name)

    @property
    def global_step(self):
        return self.env.global_step

    @global_step.setter
    def global_step(self, value):
        self.env.global_step = value

    def _now(self) -> float:
        m = self.env.current_metrics
        return float(m.sim_time) if m is not None else 0.0

    def _start_slots(self):
        self._boundary = self._now() + self.slot_seconds

    @staticmethod
    def _full_mask(obs):
        if obs is None or "action_mask" not in obs:
            return obs
        out = dict(obs)
        out["action_mask"] = np.ones_like(obs["action_mask"])
        return out

    def reset(self):
        obs = self.env.reset()
        self._start_slots()
        return self._full_mask(obs)

    def step(self, action):
        total, discount, elapsed = 0.0, 1.0, 0.0
        terms = {}
        while True:
            obs, reward, done, info = self.env.step(action)
            dt = float(info.get("dt") or 0.0)
            total += discount * float(reward)
            for k, v in (info.get("reward_terms") or {}).items():
                terms[k] = terms.get(k, 0.0) + discount * float(v)
            elapsed += dt
            if self.gamma_per_second is not None:
                discount *= self.gamma_per_second ** dt
            if done:
                info = dict(info, dt=elapsed, reward_terms=terms)
                if info.get("terminal_obs") is not None:
                    info["terminal_obs"] = self._full_mask(info["terminal_obs"])
                self._start_slots()            # the inner env has auto-reset into the next episode
                return self._full_mask(obs), total, True, info
            t = self._now()
            if t >= self._boundary:
                while self._boundary <= t:
                    self._boundary += self.slot_seconds
                return self._full_mask(obs), total, False, dict(info, dt=elapsed, reward_terms=terms)
