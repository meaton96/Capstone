"""The neutral decision view a Python rule reads (rules_api, 2026-10-11).

Both simulators build the same view at each decision: the twin from its state (rules_api.twin_adapter), Unity from the
JSON the player sends under -pyrules (rules_api.unity_adapter). A rule never touches the simulator, only these objects.

Units are sim-seconds. Values that are float32 in the simulators (processing times, loads, remaining work, WINQ,
utilization, travel) are numpy float32 here, so float32 arithmetic in a rule (e.g. `m.load + m.ptime`) rounds as the
C# and twin rules do. Clock-like values (now, arrival, since, due, op_due) are Python floats; a job without a due date
has due = op_due = +inf.

DecisionView (passed to every rule as `ctx`)
  kind          "routing" (pick a job from the pool and a machine for it) or "dispatch" (pick a job from an idle
                machine's queue; no machine choice)
  now           sim time of the decision
  decision      the simulator's decision index (Unity: DecisionRequest.DecisionIndex; twin: Twin.decisions)
  machine       dispatch: the idle machine's id; routing: -1
  jobs          the candidate jobs (JobView), in the simulator's order (ties go to the first)
  wip           jobs in the system (arrived, not exited)
  fleet         AGVs on duty now (fleet schedule); n_agvs the fleet size
  pt            PECT / PTECT parameters {"alpha", "shape", "tau", "reroute", "avail"} of this episode
  travel_price  TECT's travel price lambda (Unity FJSSPConfig.TravelPrice; the twin has none: 0)
  p_mean        mean of job.p over the candidates (ATC's scale), float
  cache         a dict that lives for this decision only (precompute something once for all candidates)

JobView (one candidate job)
  id, op (index of the current operation), ops_left (operations left, the current one included), total_ops
  arrival, since (time it entered its current state = the queue it waits in), time_in_queue = float32(now - since)
  due, op_due (the current operation's due date; twin only, Unity has none: +inf)
  p             the job rule's processing time: at dispatch the time on this machine, at routing the minimum over
                the eligible machines (as SPT / LPT / PTWINQ use)
  remaining_work  sum over the remaining operations of the minimum processing time (SRT's w)
  winq          least load among the machines eligible for the next operation, 0 on the last (PTWINQ's WINQ)
  machines      routing only: the candidate machines for this job (MachineView), in floor order; () at dispatch

MachineView (one candidate machine of one job, routing only)
  id
  ptime         this job's processing time on the machine (p)
  load          queued + committed work incl. the in-process remainder (q; JobStore.GetMachineLoad / machine_load)
  util          busy time / operational time so far (MMUR)
  travel        loaded AGV travel estimate from the job's location, seconds (TECT's tr; float32 max = no route)
"""
import math

import numpy as np

f32 = np.float32


class MachineView:
    __slots__ = ("id", "ptime", "load", "util", "travel")

    def __init__(self, id, ptime, load, util, travel):
        self.id, self.ptime, self.load, self.util, self.travel = int(id), ptime, load, util, travel

    def __repr__(self):
        return (f"MachineView(id={self.id}, ptime={float(self.ptime):g}, load={float(self.load):g}, "
                f"util={float(self.util):g}, travel={float(self.travel):g})")


class JobView:
    """One candidate job. remaining_work, winq and machines may be filled lazily by an adapter (the twin computes
    them only when a rule reads them, as its built-in rules do); a rule just reads the attribute."""
    __slots__ = ("id", "op", "ops_left", "total_ops", "arrival", "since", "time_in_queue", "due", "op_due", "p",
                 "_rw", "_winq", "_machines", "_src")

    def __init__(self, id, op, total_ops, arrival, since, now, due, op_due, p, remaining_work=None, winq=None,
                 machines=None, src=None):
        self.id, self.op, self.total_ops = int(id), int(op), int(total_ops)
        self.ops_left = self.total_ops - self.op
        self.arrival, self.since = float(arrival), float(since)
        self.time_in_queue = f32(now - since)
        self.due = math.inf if due is None else float(due)
        self.op_due = math.inf if op_due is None else float(op_due)
        self.p = p
        self._rw, self._winq, self._machines, self._src = remaining_work, winq, machines, src

    @property
    def remaining_work(self):
        if self._rw is None:
            self._rw = self._src.remaining_work(self)
        return self._rw

    @property
    def winq(self):
        if self._winq is None:
            self._winq = self._src.winq(self)
        return self._winq

    @property
    def machines(self):
        if self._machines is None:
            self._machines = () if self._src is None else self._src.machines(self)
        return self._machines

    def __repr__(self):
        return f"JobView(id={self.id}, op={self.op}/{self.total_ops}, p={float(self.p):g}, due={self.due:g})"


class DecisionView:
    __slots__ = ("kind", "now", "decision", "machine", "jobs", "wip", "fleet", "n_agvs", "pt", "travel_price",
                 "cache", "_p_mean")

    def __init__(self, kind, now, decision, jobs, machine=-1, wip=0, fleet=0, n_agvs=0, pt=None, travel_price=0.0):
        if kind not in ("routing", "dispatch"):
            raise ValueError(f"decision kind must be routing or dispatch, not {kind!r}")
        self.kind, self.now, self.decision, self.machine = kind, float(now), int(decision), int(machine)
        self.jobs = list(jobs)
        self.wip, self.fleet, self.n_agvs = int(wip), int(fleet), int(n_agvs)
        self.pt = pt
        self.travel_price = float(travel_price)
        self.cache = {}
        self._p_mean = None

    @property
    def p_mean(self):
        """Mean of job.p over the candidates, in float64 summed in candidate order (ATC's p_mean)."""
        if self._p_mean is None:
            self._p_mean = sum(float(j.p) for j in self.jobs) / len(self.jobs)
        return self._p_mean

    def job(self, jid):
        for j in self.jobs:
            if j.id == jid:
                return j
        raise KeyError(f"job {jid} is not a candidate of this decision")

    def __repr__(self):
        return (f"DecisionView({self.kind}, t={self.now:g}, #{self.decision}, machine={self.machine}, "
                f"jobs={[j.id for j in self.jobs]})")
