"""
@file flow_time.py
@brief Flow-time reward: penalize total job time in system.

@details
@c time_in_system_sum grows by (jobs in system) × (elapsed time), so its per-step delta
charges every job that is still waiting while time passes. Summed over a completed
episode this equals -(total flow time) / time_scale. Optional bonus per finished job, and
a terminal penalty when the episode ends by deadlock or timeout.
"""

from rewards.base import RewardFunction, resolve


class FlowTimeReward(RewardFunction):
    """
    @param time_scale        Job-seconds per unit of penalty (schedulable).
    @param completion_bonus  Reward per job that exits the system (schedulable).
    @param failure_penalty   Penalty when the episode ends by deadlock or timeout (schedulable).
    """

    def __init__(self, time_scale=1000.0, completion_bonus=0.0, failure_penalty=10.0):
        super().__init__(time_scale=time_scale, completion_bonus=completion_bonus,
                         failure_penalty=failure_penalty)

    def compute(self, prev, curr, ctx):
        step = ctx.global_step
        terms = {
            "flow_time": -curr.delta(prev, "time_in_system_sum") / resolve(self.params["time_scale"], step),
        }

        bonus = resolve(self.params["completion_bonus"], step)
        if bonus:
            terms["completion"] = bonus * curr.delta(prev, "jobs_exited")

        if ctx.done and (curr.deadlock or curr.timed_out):
            terms["failure"] = -resolve(self.params["failure_penalty"], step)
        return terms
