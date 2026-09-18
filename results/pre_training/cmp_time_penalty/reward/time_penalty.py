"""
@file time_penalty.py
@brief Makespan-style reward: penalize simulated time elapsed after each decision.

@details
Summed over an episode this equals -makespan / time_scale. Unlike the former C# reward,
each penalty is credited to the action that preceded the elapsed time, and it does not
depend on Unity's time scale.
"""

from rewards.base import RewardFunction, resolve


class TimePenaltyReward(RewardFunction):
    """@param time_scale  Sim-seconds per unit of penalty (schedulable)."""

    def __init__(self, time_scale=100.0):
        super().__init__(time_scale=time_scale)

    def compute(self, prev, curr, ctx):
        scale = resolve(self.params["time_scale"], ctx.global_step)
        return {"time": -curr.delta(prev, "sim_time") / scale}
