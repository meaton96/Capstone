"""
@file tardiness.py
@brief Tardiness reward: penalize the time jobs spend past their due date.

@details
@c tardiness_sum (reward metrics v5) grows by (open jobs already past their due date) x (elapsed time), so its
per-step delta charges every late job still in the shop, the late-WIP integral. Summed over an episode that drains
it equals -(total tardiness) / time_scale; over a capped window it counts the lateness accrued inside the window,
finished or not, as flow_time does for time in system. Jobs without a due date never count.

Needs due dates on the jobs (scenario "dueDate"; RandomizedParams.due_date_allowance > 0 writes total-work-content
ones) and a schema-v5 player. A seeded episode (episode_seed >= 0, i.e. one built from a queued scenario) whose jobs
carry none raises instead of training on a zero reward. The player's unseeded start-up episode (its default config,
seed -1, before any queued scenario applies) has no due dates by design and just scores 0.
Background and the screen that chose this objective: docs/Plans/PDR_RULE_SET_PLAN_2026-10-02.md section 9,
results/rq2-twin-due/README.md.
"""

from rewards.base import RewardFunction, resolve


class TardinessReward(RewardFunction):
    """
    @param time_scale       Job-seconds per unit of penalty (schedulable).
    @param flow_weight      Weight of an added time-in-system term (the flow_time reward's), 0 = pure tardiness
                            (schedulable).
    @param failure_penalty  Penalty when the episode ends by deadlock or timeout (schedulable).
    """

    def __init__(self, time_scale=1000.0, flow_weight=0.0, failure_penalty=10.0):
        super().__init__(time_scale=time_scale, flow_weight=flow_weight, failure_penalty=failure_penalty)

    def compute(self, prev, curr, ctx):
        step = ctx.global_step
        if curr.episode_seed >= 0 and curr.jobs_total > 0 and curr.jobs_with_due_date == 0:
            raise ValueError("the tardiness reward needs due dates, but no job in this episode has one (generate "
                             "with RandomizedParams.due_date_allowance > 0, or give the jobs a \"dueDate\")")
        scale = resolve(self.params["time_scale"], step)
        terms = {"tardiness": -curr.delta(prev, "tardiness_sum") / scale}

        weight = resolve(self.params["flow_weight"], step)
        if weight:
            terms["flow_time"] = -weight * curr.delta(prev, "time_in_system_sum") / scale

        if ctx.done and (curr.deadlock or curr.timed_out):
            terms["failure"] = -resolve(self.params["failure_penalty"], step)
        return terms
