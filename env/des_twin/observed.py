"""
@file observed.py
@brief The controllers' view of the twin under uncertainty (2026-10-11, dev-twin-uncertainty; plan §2.3-2.4):
       ObservedState exposes the duck-typed state API that des_twin.rules and des_twin.observation read from a Twin,
       but built from *reported* events, so rules, the planner and the policy all see the same degraded information.

@details
**Event log.** Twin._to (the single place a job's state changes) hands every transition to the view core with the
job's location / target / current op at that moment; arrivals are applied at once (an arriving job is a local trigger:
it requests routing itself). A transition becomes visible at UncertaintyTrace.visible_time (event time + O1 booking
delay, frozen by an O2 outage of the machine it concerns, rounded up to the next O1p snapshot). Entries are applied
lazily, in visibility order, whenever the view is queried; an entry older than the job's last applied one (bookings
overtaking each other) is dropped.

**Reported state.** Per job: state, location, target, current op, `since` (the booking's time stamp: the event's
time by default, o1_stamp="booking" uses the booking time). Per machine: the job it processes (from reported starts /
finishes), the start stamp and tick, busy time from reported start-finish pairs. Machine load = observed remaining
time + the planned times of the jobs reported queued at / heading to the machine, summed in arrival order as the
engine does. Observed remaining time = the controller's estimate, max(g x p_hat - elapsed since the reported start, 0)
on Unity's float32 countdown, with g the O3 factor and p_hat = p_plan (I0) or p_plan x b_hat (I1): a planned countdown,
so an overrun shows as busy with remaining 0 (observable in obs v3, machine columns 6 and 9).

**Local truth, global staleness (§1.2).** Decisions are still triggered on the true state. At a decision the
deciding entities are true: the candidate jobs (routing pool, or the deciding machine's physical queue) report their
true state / location / current op, and the deciding machine its true idle state. `since` of a local job whose entry
into its current state is not booked yet is `now` (time in queue 0). Everything else (other machines' loads, status,
remaining time, utilization, other jobs) is the reported state. AGV telemetry and failure status stay exact (the
failure flags are local alarms; failures are off in the uncertainty study).

**Information level I1.** A per-machine estimate of the execution bias from reported completions (actual / planned of
finished ops, log scale, exponential forgetting with half-life cfg.bias_half_life, shrunk to 1 by cfg.bias_prior
pseudo-observations). At I1 every planned duration the view hands out becomes p_hat = f32(p_plan x b_hat): rules,
remaining work, loads, the observation (unless cfg.info_obs is False) and the planner's belief. Every estimator read
syncs first, so b_hat includes every completion visible by now whoever read the view before (fix of 2026-10-17: until
then `bias_estimate` / `bhat` / `_ops` read it unsynced, and the estimate was as fresh as the last syncing read: an
agent's observation build synced at every decision, a bare rule play or the warm-up did not;
results/rq4-uncert-v5-policy, "Root cause").

**Exactness.** With all delays 0 every entry is applied before any query in the same tick, the reported state equals
the true state and every value is computed with the engine's float32 operations in the engine's order, so a zero-
severity configuration reproduces the noise-free twin bit for bit (results/dev-twin-uncertainty/check_exact.py).
"""
import heapq
import math
from bisect import bisect_left, insort

import numpy as np

from .motion import f32_countdown

f32 = np.float32
NEEDS_ROUTING, WAITING_PICKUP, IN_TRANSIT, QUEUED, PROCESSING, EXITED = range(6)


class _Rep:
    """A job's reported state."""
    __slots__ = ("state", "loc", "target", "cur_op", "since", "n", "seq", "load")

    def __init__(self, state, loc, target, cur_op, since, seq):
        self.state, self.loc, self.target, self.cur_op, self.since = state, loc, target, cur_op, since
        self.n, self.seq, self.load = 0, seq, -1


class _Ord:
    """Job ids kept in arrival order (seq)."""
    __slots__ = ("keys", "items")

    def __init__(self):
        self.keys, self.items = [], {}

    def add(self, seq, jid):
        insort(self.keys, seq)
        self.items[seq] = jid

    def remove(self, seq):
        del self.keys[bisect_left(self.keys, seq)]
        del self.items[seq]

    def ids(self):
        items = self.items
        return [items[k] for k in self.keys]


class ViewCore:
    """Shared reported state of one twin (one per Twin; the facades ObservedState read it)."""

    def __init__(self, twin, trace):
        self.tw, self.tr = twin, trace
        self.cfg = trace.cfg
        self.booking_stamp = self.cfg.o1_stamp == "booking"
        self.heap, self.seqno = [], 0
        self.ntrue, self.occ, self.rep = {}, {}, {}
        self.live = _Ord()
        self.load = {m.id: _Ord() for m in twin.machines}
        self.proc = {m.id: None for m in twin.machines}       # (job, op, start tick, start stamp)
        self.busy = {m.id: 0.0 for m in twin.machines}
        self.local_jobs, self.local_machine = frozenset(), -1
        # I1 estimator per machine: [weight, weighted sum of log ratios, time of the last update]
        self.est = {m.id: [0.0, 0.0, 0.0] for m in twin.machines}
        self.n_updates = 0
        self.zero_steps = {}            # estimate -> ticks until its float32 countdown reaches 0
        self.stats ={"bookings": 0, "dropped_stale": 0, "delay_sum": 0.0}

    # ── events ──────────────────────────────────────────────────────────────────────────────────────

    def on_arrive(self, job):
        r = _Rep(job.state, job.location, job.target, job.cur_op, job.since, job.seq)
        self.rep[job.id] = r
        self.ntrue[job.id] = 0
        self.live.add(job.seq, job.id)
        self._index_load(job.id, r)

    def on_event(self, job, old, state):
        tw = self.tw
        jid = job.id
        n = self.ntrue[jid] = self.ntrue[jid] + 1
        op = job.cur_op
        key = (jid, op, state)
        occ = self.occ.get(key, 0)
        self.occ[key] = occ + 1
        if state in (QUEUED, PROCESSING) or old == PROCESSING or state == NEEDS_ROUTING:
            mid = job.location
        elif state in (WAITING_PICKUP, IN_TRANSIT):
            mid = job.target
        else:
            mid = -1
        now = tw.now
        v = self.tr.visible_time(now, jid, op, state, occ, mid)
        stamp, k = (v, int(math.ceil(v / tw.dt - 1e-9))) if self.booking_stamp else (now, tw.k)
        self.stats["bookings"] += 1
        self.stats["delay_sum"] += v - now
        heapq.heappush(self.heap, (v, self.seqno, (jid, n, state, job.location, job.target, op, old, k, stamp)))
        self.seqno += 1

    def sync(self):
        heap = self.heap
        if heap and heap[0][0] <= self.tw.now:
            now = self.tw.now
            while heap and heap[0][0] <= now:
                self._apply(heapq.heappop(heap)[2])

    def _apply(self, e):
        jid, n, state, loc, target, op, old, k, stamp = e
        r = self.rep[jid]
        if n <= r.n:
            self.stats["dropped_stale"] += 1
            return
        if r.state == PROCESSING:
            m = r.loc
            cur = self.proc.get(m)
            if cur is not None and cur[0] == jid:
                if op > r.cur_op:                                   # a completed operation (not a failure restart)
                    self.busy[m] += stamp - cur[3]
                    if old == PROCESSING and self.cfg.info == "I1":
                        plan = float(self.tw.jobs[jid].ops[r.cur_op][m])
                        if plan > 0 and stamp > cur[3]:
                            self._update_est(m, stamp, (stamp - cur[3]) / plan)
                self.proc[m] = None
        if state == PROCESSING:
            self.proc[loc] = (jid, op, k, stamp)
        r.state, r.loc, r.target, r.cur_op, r.since, r.n = state, loc, target, op, stamp, n
        if state == EXITED:
            self.live.remove(r.seq)
        self._index_load(jid, r)

    def _index_load(self, jid, r):
        if r.state == QUEUED:
            mid = r.loc
        elif r.state in (WAITING_PICKUP, IN_TRANSIT) and r.target >= 0:
            mid = r.target
        else:
            mid = -1
        if mid != r.load:
            if r.load >= 0:
                self.load[r.load].remove(r.seq)
            if mid >= 0:
                self.load[mid].add(r.seq, jid)
            r.load = mid

    # ── I1 estimator ────────────────────────────────────────────────────────────────────────────────

    def _update_est(self, mid, t, ratio):
        e = self.est[mid]
        h = self.cfg.bias_half_life
        decay = 0.5 ** ((t - e[2]) / h) if h > 0 and t > e[2] else 1.0
        e[0] = e[0] * decay + 1.0
        e[1] = e[1] * decay + math.log(ratio)
        e[2] = max(t, e[2])
        self.n_updates += 1

    def bias_estimate(self, mid, t=None):
        """(b_hat, effective number of observations) of machine mid at time t (default now), from every completion
        visible by now (it syncs first; see the module docstring, I1)."""
        self.sync()
        e = self.est[mid]
        if e[0] <= 0.0:
            return 1.0, 0.0
        t = self.tw.now if t is None else t
        h = self.cfg.bias_half_life
        decay = 0.5 ** ((t - e[2]) / h) if h > 0 and t > e[2] else 1.0
        w, s = e[0] * decay, e[1] * decay
        return math.exp(s / (w + self.cfg.bias_prior)), w

    def bhat(self, mid):
        if self.cfg.info == "I2":
            return self.tr.bias_at(mid, self.tw.now)
        self.sync()                     # n_updates as of now (2026-10-17 fix)
        return self.bias_estimate(mid)[0] if self.n_updates else 1.0

    def estimate(self, jid, op, mid, plan, i1=None):
        """The controller's estimate of an operation's duration: f32(p_plan x g x b_hat) (p_plan when both are 1)."""
        g = self.tr.est_factor(jid, op)
        b = self.bhat(mid) if (self.cfg.adaptive if i1 is None else i1) else 1.0
        if g == 1.0 and b == 1.0:
            return plan
        return f32(float(plan) * g * b)


class _JobMap:
    __slots__ = ("v", "cache")

    def __init__(self, v):
        self.v, self.cache = v, {}

    def __getitem__(self, jid):
        p = self.cache.get(jid)
        if p is None:
            p = self.cache[jid] = JobProxy(self.v, self.v.tw.jobs[jid])
        return p

    def get(self, jid, default=None):
        if jid not in self.v.tw.jobs:
            return default
        return self[jid]

    def __contains__(self, jid):
        return jid in self.v.tw.jobs

    def values(self):
        return [self[j] for j in self.v.tw.jobs]


class JobProxy:
    """A job as the controllers see it (reported state; true for the decision's local jobs)."""
    __slots__ = ("_v", "_j", "id", "arrival", "due", "op_due", "types")

    def __init__(self, view, job):
        self._v, self._j = view, job
        self.id, self.arrival, self.due, self.op_due, self.types = job.id, job.arrival, job.due, job.op_due, job.types

    @property
    def seq(self):
        return self._j.seq

    def _r(self):
        core = self._v.core
        if self.id in core.local_jobs:
            return None
        core.sync()
        return core.rep.get(self.id)

    @property
    def total_ops(self):
        return len(self._j.ops)

    @property
    def ops(self):
        return self._v._ops(self._j)

    @property
    def state(self):
        r = self._r()
        return self._j.state if r is None else r.state

    @property
    def location(self):
        r = self._r()
        return self._j.location if r is None else r.loc

    @property
    def target(self):
        r = self._r()
        return self._j.target if r is None else r.target

    @property
    def cur_op(self):
        r = self._r()
        return self._j.cur_op if r is None else r.cur_op

    completed_ops = cur_op

    @property
    def since(self):
        core = self._v.core
        core.sync()
        r = core.rep.get(self.id)
        if self.id in core.local_jobs and (r is None or r.n < core.ntrue.get(self.id, 0)):
            return self._v.tw.now                  # entry into the current state not booked yet
        return self._j.since if r is None else r.since

    @property
    def assigned_agv(self):
        return self._j.assigned_agv

    @property
    def pre_agv(self):
        return self._j.pre_agv

    def proc(self, mid):
        ops, cur = self.ops, self.cur_op
        return ops[cur].get(mid, f32(0)) if cur < len(ops) else f32(0)

    def min_proc(self):
        return min(self.ops[self.cur_op].values())


class MachineProxy:
    __slots__ = ("_v", "_m", "id", "info")

    def __init__(self, view, m):
        self._v, self._m, self.id, self.info = view, m, m.id, m.info

    @property
    def idle(self):
        core = self._v.core
        if self.id == core.local_machine:
            return self._m.idle
        core.sync()
        return core.proc[self.id] is None

    @property
    def job(self):
        core = self._v.core
        if self.id == core.local_machine:
            return self._m.job
        core.sync()
        p = core.proc[self.id]
        return -1 if p is None else p[0]

    @property
    def down(self):
        return self._m.down


class ObservedState:
    """@brief The degraded view with the Twin's duck-typed read API (rules.rank_jobs, ObservationBuilder.build,
    Twin._route / _dispatch_decision). Attributes it does not override are the twin's (AGVs, clock, floor, ...)."""

    def __init__(self, core, i1):
        self.core, self.tw, self.i1 = core, core.tw, bool(i1)
        self.jobs = _JobMap(self)
        self.machines = [MachineProxy(self, m) for m in core.tw.machines]
        self.mach = {m.id: m for m in self.machines}
        self._ops_cache = {}

    def __getattr__(self, name):
        if name.startswith("__") or name in ("core", "tw"):
            raise AttributeError(name)
        return getattr(self.tw, name)

    # ── decision-local truth ─────────────────────────────────────────────────────────────────────────

    def set_local(self, jobs, machine=-1):
        self.core.local_jobs, self.core.local_machine = frozenset(jobs), machine

    def clear_local(self):
        self.core.local_jobs, self.core.local_machine = frozenset(), -1

    # ── planned durations at the information level ──────────────────────────────────────────────────

    def _ops(self, job):
        if self.i1:
            self.core.sync()            # n_updates / b_hat as of now (2026-10-17 fix)
        if not self.i1 or (not self.core.n_updates and self.core.cfg.info == "I1"):
            return job.ops
        key = (self.core.n_updates, self.tw.now)
        hit = self._ops_cache.get(job.id)
        if hit is not None and hit[0] == key:
            return hit[1]
        bh = {}
        out = []
        for op in job.ops:
            d = {}
            for mid, p in op.items():
                b = bh.get(mid)
                if b is None:
                    b = bh[mid] = self.core.bhat(mid)
                d[mid] = p if b == 1.0 else f32(float(p) * b)
            out.append(d)
        self._ops_cache[job.id] = (key, out)
        return out

    # ── the Twin read API ───────────────────────────────────────────────────────────────────────────

    def _live(self):
        core = self.core
        core.sync()
        seqs = core.live.keys
        local = core.local_jobs
        if local:
            extra = [self.tw.jobs[j].seq for j in local if self.tw.jobs[j].seq not in core.live.items]
            if extra:
                seqs = sorted(set(seqs) | set(extra))
        order = self.tw.order
        return iter([self.jobs[order[s].id] for s in seqs])

    def remaining_proc(self, m):
        core = self.core
        mid = m.id
        tw = self.tw
        if mid == core.local_machine:
            return tw.remaining_proc(tw.mach[mid])
        core.sync()
        p = core.proc[mid]
        if p is None:
            return f32(0)
        jid, op, k0, _ = p
        plan = tw.jobs[jid].ops[op][mid]
        est = core.estimate(jid, op, mid, plan, self.i1)
        steps = max(0, tw.k - k0)
        n0 = core.zero_steps.get(est)
        if n0 is None:
            n0 = core.zero_steps[est] = f32_countdown(est, tw.dt, until=0.0)[0]
        if steps >= n0:            # overrun: the countdown is <= 0 (walking it below 0 is slow and gives 0 anyway)
            return f32(0)
        return f32(max(0.0, float(f32_countdown(est, tw.dt, steps=steps)[1])))

    def machine_load(self, mid):
        load = self.remaining_proc(self.mach[mid])
        for jid in self.core.load[mid].ids():
            load = f32(load + self.jobs[jid].proc(mid))
        return load

    def all_machine_loads(self):
        loads = {m.id: self.remaining_proc(m) for m in self.machines}
        for mid, idx in self.core.load.items():
            if idx.keys:
                load = loads[mid]
                for jid in idx.ids():
                    load = f32(load + self.jobs[jid].proc(mid))
                loads[mid] = load
        return loads

    def work_in_next_queue(self, jid, loads):
        job = self.jobs[jid]
        nxt = job.cur_op + 1
        if nxt >= job.total_ops:
            return f32(0)
        best = None
        for mid in job.ops[nxt]:
            if mid in loads and (best is None or loads[mid] < best):
                best = loads[mid]
        return f32(0) if best is None else best

    def remaining_work(self, jid):
        job = self.jobs[jid]
        ops = job.ops
        total = f32(0)
        for o in range(job.cur_op, len(ops)):
            total = f32(total + min(ops[o].values()))
        return total

    def utilization(self, m):
        core, tw = self.core, self.tw
        mid = m.id
        core.sync()
        p = core.proc[mid]
        busy = core.busy[mid] + (tw.now - p[3] if p is not None else 0.0)
        if tw.fail is not None:
            up = tw.now - tw.downtime_so_far(tw.mach[mid])
            return f32(busy / up) if up > 0 else f32(0)
        return f32(busy / tw.now) if tw.now > 0 else f32(0)

    def remaining_repair(self, m):
        return self.tw.remaining_repair(self.tw.mach[m.id])

    def downtime_so_far(self, m):
        return self.tw.downtime_so_far(self.tw.mach[m.id])

    def bias_estimates(self):
        """{machine: (b_hat, n_eff)} (the I1 estimator; (1, 0) before any reported completion)."""
        self.core.sync()
        return {m.id: self.core.bias_estimate(m.id) for m in self.machines}


def make_views(twin, trace):
    """(core, rules view, observation view) for a twin built with TwinConfig.uncertainty = trace."""
    core = ViewCore(twin, trace)
    i1 = trace.cfg.adaptive
    rules = ObservedState(core, i1)
    obs = rules if (not i1 or trace.cfg.info_obs) else ObservedState(core, False)
    return core, rules, obs
