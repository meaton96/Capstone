"""Twin side of the rules API (rules_api, 2026-10-11): the decision view built from the event-based twin's state.

des_twin.engine calls routing_view / dispatch_view at a decision whose rule is a Python RulePair (TwinConfig.py_rule as
the fixed / warm-up rule, or a RulePair sent by the agent: gen.send(pair) instead of (job_rule, machine_rule)). The view
reads the same state the built-in rules read: the twin, or its degraded ObservedState under the uncertainty layer (S).
Remaining work, WINQ and the candidate machines are computed lazily, only when a rule reads them, from the same twin
functions the built-in rules call (Twin.remaining_work, all_machine_loads + work_in_next_queue, Twin._machine_lists),
so a ported rule sees bit-identical inputs.
"""
from .views import DecisionView, JobView, MachineView


class _TwinSource:
    __slots__ = ("tw", "S", "_loads")

    def __init__(self, tw, S):
        self.tw, self.S, self._loads = tw, S, None

    def remaining_work(self, jv):
        return self.S.remaining_work(jv.id)

    def winq(self, jv):
        if self._loads is None:
            self._loads = self.S.all_machine_loads()
        return self.S.work_in_next_queue(jv.id, self._loads)

    def machines(self, jv):
        cands, times, loads, utils, travel = self.tw._machine_lists(self.tw.jobs[jv.id], self.S)
        return tuple(MachineView(*row) for row in zip(cands, times, loads, utils, travel))


def _globals(tw):
    n = len(tw.agvs)
    on = n
    for t, k in tw.cfg.agv_schedule or ():
        if t > tw.now:
            break
        on = k
    return len(tw._live_d), min(on, n), n


def _job_view(S, jid, now, p, src):
    j = S.jobs[jid]
    op = j.cur_op
    op_due = None if j.op_due is None or op >= len(j.op_due) else j.op_due[op]
    return JobView(jid, op, j.total_ops, j.arrival, j.since, now, j.due, op_due, p, src=src)


def routing_view(tw, S, ids):
    """View of a routing decision over the pool `ids` (job p = the fastest eligible machine's time)."""
    src = _TwinSource(tw, S)
    now = S.now
    jobs = [_job_view(S, i, now, S.jobs[i].min_proc(), src) for i in ids]
    wip, fleet, n = _globals(tw)
    return DecisionView("routing", now, tw.decisions, jobs, -1, wip, fleet, n, tw.ptime_params(), 0.0)


def dispatch_view(tw, S, mid, queue):
    """View of a dispatch decision at machine `mid` over its queue (job p = the time on mid)."""
    src = _TwinSource(tw, S)
    now = S.now
    jobs = [_job_view(S, i, now, S.jobs[i].proc(mid), src) for i in queue]
    wip, fleet, n = _globals(tw)
    return DecisionView("dispatch", now, tw.decisions, jobs, mid, wip, fleet, n, tw.ptime_params(), 0.0)
