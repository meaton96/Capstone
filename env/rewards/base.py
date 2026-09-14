"""
@file base.py
@brief Reward function base class, per-call context, and parameter schedules.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Tuple

from rewards.metrics import MetricsSnapshot


@dataclass
class RewardContext:
    """@brief Extra information passed to every reward call."""

    ## @brief Environment steps taken so far across all envs (drives schedules).
    global_step: int = 0
    ## @brief Episodes completed so far by this env.
    episode: int = 0
    ## @brief Index of the env inside the vectorized wrapper.
    env_id: int = 0
    ## @brief True when @c curr is the episode's terminal snapshot.
    done: bool = False


class RewardFunction:
    """@brief Base class for reward functions.

    @details
    Subclasses implement @ref compute and return named reward terms; the scalar reward is
    their sum. Constructor keyword arguments come from the spec's @c params. Each env gets
    its own instance, so per-episode state is allowed (clear it in @ref reset).
    """

    def __init__(self, **params):
        self.params = params

    def reset(self, first: MetricsSnapshot) -> None:
        """@brief Called with the first snapshot of every episode."""

    def compute(self, prev: MetricsSnapshot, curr: MetricsSnapshot,
                ctx: RewardContext) -> Dict[str, float]:
        """@brief Return named reward terms for the transition @p prev → @p curr."""
        raise NotImplementedError

    def __call__(self, prev: MetricsSnapshot, curr: MetricsSnapshot,
                 ctx: RewardContext) -> Tuple[float, Dict[str, float]]:
        terms = {name: float(value) for name, value in self.compute(prev, curr, ctx).items()}
        return float(sum(terms.values())), terms


def resolve(value, global_step: int) -> float:
    """@brief Evaluate a parameter that may be a schedule over training steps.

    @param value  One of:
                  - a number (constant);
                  - a linear schedule @c {"start": a, "end": b, "steps": n, "begin": 0},
                    moving from @c a to @c b over @c n steps starting at @c begin;
                  - a piecewise-linear schedule @c {"piecewise": [[step, value], ...]},
                    held constant before the first and after the last point.
    @param global_step  Current training step.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported schedule value: {value!r}")

    if "piecewise" in value:
        points = sorted((float(step), float(v)) for step, v in value["piecewise"])
        if global_step <= points[0][0]:
            return points[0][1]
        for (s0, v0), (s1, v1) in zip(points, points[1:]):
            if global_step <= s1:
                return v1 if s1 == s0 else v0 + (v1 - v0) * (global_step - s0) / (s1 - s0)
        return points[-1][1]

    start, end = float(value["start"]), float(value["end"])
    begin, steps = float(value.get("begin", 0)), float(value["steps"])
    frac = 1.0 if steps <= 0 else min(max((global_step - begin) / steps, 0.0), 1.0)
    return start + (end - start) * frac
