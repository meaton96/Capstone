"""Next-event engine of the twin.

The clock jumps between orchestrator ticks at which something happens; within a tick the steps run in
FactoryOrchestrator.FixedUpdate's order (heuristic drain mode, "-baselinedrain"):

  1. machine completions   (FlagHarvester.HarvestMachineFlags, machine order)
  2. AGV pickups/deliveries (HarvestAGVFlags, AGV order)
  3. pre-dispatch           (HarvestAlmostDoneFlags)
  4. AssignAGVs
  5. decision drain         (routing before dispatch, as DecisionCoordinator.FindNextDecision)
  6. scripted arrivals      (TickScriptedArrivals: a job arriving at tick k is routed from tick k + 1)

With a fixed rule (run / run_twin) every decision is drained with it, as "-baselinedrain". Agent mode
(Twin.agent_decisions, used by env_wrappers/twin_env.py) stops at each decision after the warm-up, as the player's
"-rldecisiondrain": the generator yields a Decision (Unity's DecisionRequest) and takes the rule halves to apply.

Machines and AGVs act after the orchestrator in a tick, so work started at tick k runs from tick k and a flag
raised at tick m is harvested at m + 1. AGV failures and multi-tile floors are not modelled: compare only against
Unity runs without them.

Machine failures (2026-10-09, dev-twin-failures): with TwinConfig.failures (a des_twin.failures.FailureTrace) a step 0
runs first in every tick, as FailureCoordinator.HarvestFailureFlags does before every other harvest: in machine order,
a failure harvested at this tick (`_fail`, FailureCoordinator.HandleMachineFailure steps 0-6) or a repair harvested
at this tick (`_repair_done`, then RetryDeferredJobs). Orphaned pre-dispatches are released after the AGV milestones
(FlagHarvester.ReleaseOrphanedPreDispatches). Down machines are never routing candidates, a job whose eligible
machines are all down is deferred, TryAssignAgv sends a job whose target went down back to routing, and a delivery to
a machine that is down is re-routed (audit B3). With failures None every one of these paths is skipped and the engine
is bit-identical to 84e221c3 (results/dev-twin-failures/check_exact.py).

Quirk fixes (2026-10-09, dev-csharp-failures, mirrored in the C# of the v5 player): a job re-routed off a failed machine
without being started (failure step 2, B3) frees its incoming-belt slot, and a transit with no operational alternate is
delivered to the down machine and re-routed there by B3 instead of being aborted with a stale location.
TwinConfig.legacy_failure_quirks restores the old handling (player flag -legacyfailurequirks). A task cancelled or
redirected mid-handshake re-plans from the heading the AGV has turned to by then (motion.dock_yaw_after). Unity runs on
an injected failureSchedule (FailureTrace.from_schedule) match the twin event for event at 1 AGV
(results/dev-csharp-failures); failures-off behaviour is still bit-identical (check_exact.py there, vs 3ac76c2b).

Indexes (2026-10-09, dev-twin-speed): the engine used to find live jobs, routable jobs, machine queues and machine loads
by scanning every job that ever arrived (`self.order`), several times per tick; at 100 machines that was ~90% of the
run time. It now keeps the live jobs (`_live_d`), the jobs per state and per machine queue / load target as indexes
ordered by arrival position (`Job.seq` = index in `self.order`), so every scan visits the same jobs in the same order
as before and the results are bit-identical (float32 sums included). `_to` is the only place a job's state changes and
every location / target change happens before the `_to` that follows it, so `_to` keeps the indexes current. metrics()
caches its left fold over the leading exited jobs (they never change). Exactness check: results/dev-twin-speed.
"""
import heapq
import math
from bisect import bisect_left, insort
from dataclasses import dataclass

import numpy as np

from .failures import OnlineReliability, harvest_tick, remaining_countdown
from .floor import Floor
from .motion import Kinematics, dock_yaw_after, f32_countdown, handshake_done, heading, plan_leg
from .rules import PTIME_RULES, RELIABILITY_RULES, f32, parse_rule, rank_jobs, select_machine

NEEDS_ROUTING, WAITING_PICKUP, IN_TRANSIT, QUEUED, PROCESSING, EXITED = range(6)
TRANSPORTS = ("instant", "geometric", "kinematic")


@dataclass
class TwinConfig:
    rule: str                      # fixed rule; in agent mode the warm-up rule
    transport: str = "kinematic"
    agv_count: int = None          # None: the exported fleet; fewer uses its first n AGVs and bays
    routing_trigger: str = None    # None: as exported ("onTransport" / "onReady")
    max_sim_seconds: float = 500000.0
    warmup_seconds: float = 0.0    # Stochastic.WarmupSeconds: decisions before it use `rule` (agent mode)
    episode_duration_seconds: float = 0.0   # Stochastic.EpisodeDurationSeconds: truncate after warm-up + this
    # Fleet schedule (2026-10-04, regime blocks): ((t, n), ...) sorted by t; from sim time t only the first n AGVs take
    # new tasks. An AGV going off duty finishes its task and its trip back to the bay (as for charging). () = the whole
    # fleet throughout. The scenario's "agvSchedule" (des_twin.scenario.agv_schedule) gives it.
    agv_schedule: tuple = ()
    # Machine failures (2026-10-09): a des_twin.failures.FailureTrace (exogenous, calendar-time events), None = off.
    failures: object = None
    # Unity's failure handling before 2026-10-09 ("-legacyfailurequirks"): a job re-routed off a failed machine keeps its
    # incoming-belt slot, and a transit with no operational alternate is aborted (job collected later from the machine
    # it came from). False (default) = the fixed handling of the v5 player: the slot is freed and such a transit is
    # delivered to the down machine and re-routed there (results/dev-csharp-failures).
    legacy_failure_quirks: bool = False
    # Reliability-aware machine rules (RECT, RTECT, RECTA, RTECTA, HAECT; twin only): the information they get -
    # "static" (long-run per-machine parameters), "online" (estimated from the failures observed so far) or "oracle"
    # (the true current hazard: hidden regime and age) - the risk-term weight and the re-route cost (seconds).
    reliability: str = "static"
    reliability_kappa: float = 1.0
    reroute_cost: float = 90.0
    # R-online estimator (failures.OnlineReliability, reliability="online" only): forgetting half-life (seconds) and
    # prior strength (pseudo-failures at the pooled floor rate).
    reliability_half_life: float = 7200.0
    reliability_prior: float = 2.0
    # Machine discrimination of the reliability rules (dev-hetero-tune, 10-10): each candidate's failure rate and repair
    # time become omega x its own + (1 - omega) x the floor mean at the information level. 1 = the per-machine values
    # (the original rules); 0 = blind (every candidate gets the floor values, so kappa / R only weight the processing-
    # time exposure). Decouples the exposure weight (kappa, R) from discrimination. HAECT ignores it.
    reliability_mix: float = 1.0
    # Processing-time-weighted machine rules (PECT, PTECT; twin only, dev-pect-rule 10-10, rules.PTIME_RULES): weight
    # alpha of the penalty g(p), its shape (lin / quad / ratio / exp), the exp shape's time scale tau and re-route cost
    # R (seconds), and the availability A that divides the queue (1 = the plain queue).
    ptime_alpha: float = 1.0
    ptime_shape: str = "lin"
    ptime_tau: float = 5544.0
    ptime_reroute: float = 0.0
    ptime_avail: float = 1.0


@dataclass
class Decision:
    """DecisionRequest as the agent sees it (agent mode). Routing: `job` is the oldest routable job, a placeholder
    the job half re-selects among `job_candidates` unless `selected_by_rule`; `machines` are its candidates.
    Dispatch: `machine` picks from `queue`."""
    kind: str                      # "routing" / "dispatch"
    job: int = -1
    job_candidates: tuple = ()
    selected_by_rule: bool = True
    machines: tuple = ()
    machine: int = -1
    queue: tuple = ()

    def heads_that_matter(self):
        """DispatchingEngine.HeadsThatMatter: (job head, machine head) can change the outcome."""
        if self.kind == "dispatch":
            return len(self.queue) > 1, False
        pool = not self.selected_by_rule and len(self.job_candidates) > 1
        return pool, pool or len(self.machines) > 1


class _OrdIdx:
    """Jobs kept in arrival order (Job.seq): iterating it visits them as a scan of Twin.order would."""
    __slots__ = ("keys", "items")

    def __init__(self):
        self.keys, self.items = [], {}

    def add(self, job):
        insort(self.keys, job.seq)
        self.items[job.seq] = job

    def remove(self, job):
        del self.keys[bisect_left(self.keys, job.seq)]
        del self.items[job.seq]

    def jobs(self):
        items = self.items
        return [items[k] for k in self.keys]

    def __len__(self):
        return len(self.keys)


class Job:
    __slots__ = ("id", "arrival", "ops", "types", "cur_op", "completed_ops", "state", "since", "t_state",
                 "location", "target", "assigned_agv", "pre_agv", "exit_time", "op_rows", "due", "op_due",
                 "seq", "ix_load", "ix_queue")

    def __init__(self, jid, arrival, ops, types, due=None, op_due=None):
        self.id, self.arrival, self.ops, self.types = jid, arrival, ops, types
        self.due, self.op_due = due, op_due      # due date and per-operation due dates; None = no due dates
        self.cur_op = self.completed_ops = 0
        self.state, self.since = NEEDS_ROUTING, arrival
        self.t_state = [0.0] * 6
        self.location = self.target = self.assigned_agv = self.pre_agv = -1
        self.exit_time = None
        self.op_rows = []
        self.seq = -1              # index in Twin.order once arrived
        self.ix_load = -1          # machine whose load index holds the job (-1: none)
        self.ix_queue = -1         # machine whose queue index holds the job (-1: none)

    @property
    def total_ops(self):
        return len(self.ops)

    def proc(self, mid):
        """JobData.GetProcessingTime: the current op's time on mid (0 when past the last op)."""
        return self.ops[self.cur_op].get(mid, f32(0)) if self.cur_op < len(self.ops) else f32(0)

    def min_proc(self):
        return min(self.ops[self.cur_op].values())


class MachineRT:
    __slots__ = ("id", "idle", "job", "start_tick", "dur", "proc_start", "busy", "belts", "finish_tick",
                 "almost_tick", "info", "down", "fail_tick", "repair_tick", "repair_dur", "repair_start", "fptr",
                 "events", "n_fail", "repair_sum", "downtime", "down_since", "lost_work", "life_start", "age0")

    def __init__(self, info):
        self.info, self.id = info, info.id
        self.idle, self.job = True, -1
        self.start_tick, self.dur, self.proc_start, self.busy = 0, f32(0), None, 0.0
        self.belts = [[cap, set()] for _, cap in info.in_belts]
        self.finish_tick = self.almost_tick = None
        self.down = False                       # Failed or Repairing (failures only)
        self.fail_tick = self.repair_tick = None


class AgvRT:
    __slots__ = ("id", "park_pos", "park_zone", "state", "pos", "yaw", "zone", "leg", "job", "source", "target",
                 "drop_pos", "dock", "milestone", "pre_arrive", "pre_yaw", "path", "idle_since", "idle_time",
                 "trips", "hs")

    def __init__(self, a):
        self.id = a["id"]
        self.park_pos, self.park_zone = tuple(a["park_pos"]), a["park_zone"]
        self.state = "idle"
        self.pos, self.yaw, self.zone = tuple(a["pos"]), float(a["yaw"]), a["zone"]
        self.leg = None
        self.job = self.source = self.target = -1
        self.drop_pos = self.dock = self.milestone = None
        self.pre_arrive, self.pre_yaw = None, 0.0
        self.path, self.idle_since, self.idle_time, self.trips = 0.0, 0.0, 0.0, 0
        # the current dock handshake (yaw in, goal, first tick, align first), kept with failures only: the heading
        # of an AGV whose task is cancelled or redirected mid-handshake (Twin._leg_pose)
        self.hs = None


class Twin:
    def __init__(self, floor, jobs_data, cfg):
        if cfg.transport not in TRANSPORTS:
            raise ValueError(f"transport must be one of {TRANSPORTS}")
        self.floor, self.cfg = floor, cfg
        self.dt = floor.dt
        self.job_rule, self.machine_rule = parse_rule(cfg.rule)
        self.instant = cfg.transport == "instant"
        self.kin = None if self.instant else Kinematics(floor.agv, self.dt, cfg.transport)
        trigger = cfg.routing_trigger or floor.routing_trigger
        self.on_transport = trigger == "onTransport" and not self.instant
        self.lead = floor.pre_dispatch_lead
        self.machines = [MachineRT(m) for m in floor.machines]
        self.mach = {m.id: m for m in self.machines}
        n_agv = len(floor.agvs) if cfg.agv_count is None else cfg.agv_count
        if n_agv > len(floor.agvs):
            raise ValueError(f"agv_count {n_agv} exceeds the {len(floor.agvs)} bays exported in des_floor.json")
        self.agvs = [] if self.instant else [AgvRT(a) for a in floor.agvs[:n_agv]]
        self._agv_pos = {id(a): i for i, a in enumerate(self.agvs)}   # self.agvs.index(a)
        self.incoming_zone = floor.incoming_zone()

        self.jobs, self.order, self.pending = {}, [], []
        self._live_d = {}                                  # live (arrived, not exited) jobs by seq, in arrival order
        self._by_state = [_OrdIdx() for _ in range(6)]     # live jobs per state (EXITED unused)
        self._queued_at = {m.id: _OrdIdx() for m in self.machines}   # QUEUED jobs by location
        self._load_at = {m.id: _OrdIdx() for m in self.machines}     # jobs counted in machine_load(mid)
        self._mpos = {m.id: i for i, m in enumerate(self.machines)}
        self._dispatchable = set()                         # positions of idle machines with a non-empty queue
        self._prearrive = {}                               # trace entry "arrive_prepickup" -> its indexes
        self._mprefix = None                               # metrics(): fold over the leading exited jobs
        for jd in jobs_data["jobs"]:
            ops = [{int(m): f32(d) for m, d in op["eligible"]} for op in jd["ops"]]
            due, op_due = jd.get("due"), jd.get("op_due")
            if due is not None and op_due is None:
                # Operation due dates for MOD: the allowance split by work, as scenario.assign_due_dates does
                # (d_io = r + (d - r) x work of ops 0..o / total work).
                work = [float(min(op.values())) for op in ops]
                total, cum, op_due = sum(work), 0.0, []
                for w in work:
                    cum += w
                    op_due.append(float(jd["arrival"]) + (float(due) - float(jd["arrival"])) * (cum / total if total > 0 else 1.0))
            job = Job(int(jd["id"]), float(jd["arrival"]), ops, [op["type"] for op in jd["ops"]],
                      None if due is None else float(due), op_due)
            self.jobs[job.id] = job
            if job.arrival > 0.0:
                self.pending.append(job)
            else:
                self._arrive(job)
        self.pending.sort(key=lambda j: (j.arrival, j.id))
        self.pending_i = 0

        self.k, self.now = 0, 0.0
        self._heap, self._queued = [], set()
        self.decisions = 0
        self.decision_rows = []
        self.trace = []                # (time, agv, event, job) at agent ticks, as agv_events.csv
        self.timed_out = False
        self.truncated = False
        self.makespan = None
        self.agent = False
        self.forced = None             # (job, machine) for the next decision only; see force()
        # FixedUpdate ends the episode at the first tick with SimTime > warm-up end + cap, before harvesting
        self.cap_tick = None
        if cfg.episode_duration_seconds > 0:
            limit = cfg.warmup_seconds + cfg.episode_duration_seconds
            k = math.floor(limit / self.dt) + 1
            while k > 1 and (k - 1) * self.dt > limit:
                k -= 1
            while k * self.dt <= limit:
                k += 1
            self.cap_tick = k
        self.fail = cfg.failures
        if self.fail is not None:
            self._init_failures()

    # ── Clock ───────────────────────────────────────────────────────────────────────────────────

    def _wake(self, tick):
        if tick not in self._queued:
            self._queued.add(tick)
            heapq.heappush(self._heap, tick)

    def run(self):
        """Runs the episode with the fixed rule (cfg.rule) at every decision."""
        for _ in self._events():
            raise RuntimeError("a fixed-rule run never stops for a decision")
        return self

    def agent_decisions(self):
        """Agent mode: a generator that yields a Decision at each decision after the warm-up and takes the rule
        halves to apply, gen.send((job_rule, machine_rule)), e.g. ("SPT", "ECT"). Between the yield and the send
        the twin's state is the one the decision is made in (for the observation). Ends with the episode."""
        self.agent = True
        return self._events()

    def force(self, job=None, machine=None):
        """Agent mode, while paused at a decision: the next decision takes this job (and, for a routing, this
        machine) instead of the rule halves' pick; None leaves that half to the rule. Used for rollouts
        (des_twin.branch); equivalent to rq2-direct-headroom's FORCE wrapper on rank_jobs / select_machine."""
        self.forced = (job, machine)

    @property
    def in_warmup(self):
        return self.now < self.cfg.warmup_seconds

    @property
    def done(self):
        return self.makespan is not None

    def _events(self):
        self._wake(1)
        for job in self.pending:
            self._wake(max(1, math.ceil(job.arrival / self.dt - 1e-9)))
        if self.cap_tick is not None:
            self._wake(self.cap_tick)
        max_tick = int(self.cfg.max_sim_seconds / self.dt)
        while self._heap:
            k = heapq.heappop(self._heap)
            self._queued.discard(k)
            if k > max_tick:
                self.timed_out = True
                self.k, self.now = max_tick, max_tick * self.dt
                break
            self.k, self.now = k, k * self.dt
            if self.cap_tick is not None and k >= self.cap_tick:
                self.truncated = True
                break
            if (yield from self._tick(k)):
                self.makespan = self.now
                break
        if self.makespan is None:
            self.makespan = self.now

    def _tick(self, k):
        if self.fail is not None:
            self._harvest_failures(k)
        for m in self.machines:
            if m.finish_tick == k:
                self._finish(m)
        for a in self.agvs:
            if a.milestone is not None and a.milestone[0] == k:
                self._milestone(a)
        if self.fail is not None and self._orphans:
            self._release_orphans()
        waiting = None
        for m in self.machines:
            if m.almost_tick == k:
                m.almost_tick = None
                if self.on_transport and waiting is None:
                    waiting = self._any_waiting_for_transport()
                self._almost_done(m, waiting)
        self._assign_agvs()
        while (yield from self._next_decision()):
            pass
        added = False
        while self.pending_i < len(self.pending) and self.pending[self.pending_i].arrival <= self.now + 1e-9:
            job = self.pending[self.pending_i]
            self.pending_i += 1
            self._arrive(job)
            added = True
        if added:
            self._wake(k + 1)
        return self.pending_i >= len(self.pending) and not self._live_d

    # ── Job bookkeeping ─────────────────────────────────────────────────────────────────────────

    def _arrive(self, job):
        job.seq = len(self.order)
        self.order.append(job)
        self._live_d[job.seq] = job
        self._by_state[job.state].add(job)
        self._index_load(job)

    def _to(self, job, state):
        dt = self.now - job.since
        if dt > 0:
            job.t_state[job.state] += dt
        old = job.state
        job.state, job.since = state, self.now
        # indexes (callers set location / target before the _to that follows)
        self._by_state[old].remove(job)
        if job.ix_queue >= 0:
            q = self._queued_at[job.ix_queue]
            q.remove(job)
            if not q.items:
                self._dispatchable.discard(self._mpos[job.ix_queue])
            job.ix_queue = -1
        if state == EXITED:
            del self._live_d[job.seq]
        else:
            self._by_state[state].add(job)
        if state == QUEUED:
            job.ix_queue = job.location
            self._queued_at[job.location].add(job)
            m = self.mach[job.location]
            if m.idle and not m.down:
                self._dispatchable.add(self._mpos[job.location])
        self._index_load(job)

    def _index_load(self, job):
        if job.state == QUEUED:
            mid = job.location
        elif job.state in (WAITING_PICKUP, IN_TRANSIT) and job.target >= 0:
            mid = job.target
        else:
            mid = -1
        if mid != job.ix_load:
            if job.ix_load >= 0:
                self._load_at[job.ix_load].remove(job)
            if mid >= 0:
                self._load_at[mid].add(job)
            job.ix_load = mid

    def _live(self):
        return iter(self._live_d.values())

    def remaining_work(self, jid):
        job = self.jobs[jid]
        total = f32(0)
        for o in range(job.cur_op, job.total_ops):
            total = f32(total + min(job.ops[o].values()))
        return total

    def remaining_proc(self, m):
        if m.idle:
            return f32(0)
        # PhysicalMachine.remainingTime after one float32 decrement per tick since the start
        return f32(max(0.0, float(f32_countdown(m.dur, self.dt, steps=self.k - m.start_tick)[1])))

    def machine_load(self, mid):
        load = self.remaining_proc(self.mach[mid])
        # QUEUED at mid, or WAITING_PICKUP / IN_TRANSIT to mid, in arrival order (float32 sum order)
        for j in self._load_at[mid].jobs():
            load = f32(load + j.proc(mid))
        return load

    def all_machine_loads(self):
        loads = {m.id: self.remaining_proc(m) for m in self.machines}
        for mid, idx in self._load_at.items():   # per machine in arrival order, as the scan added them
            if idx.keys:
                load = loads[mid]
                for j in idx.jobs():
                    load = f32(load + j.proc(mid))
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

    def utilization(self, m):
        """DecisionCoordinator.MachineUtilization: busy time over operational time (sim time less downtime so far);
        work lost to a failure is not busy (audit G10)."""
        busy = m.busy + (self.now - m.proc_start if m.proc_start is not None else 0.0)
        if self.fail is not None:
            up = self.now - self.downtime_so_far(m)
            return f32(busy / up) if up > 0 else f32(0)
        return f32(busy / self.now) if self.now > 0 else f32(0)

    # ── Machines ────────────────────────────────────────────────────────────────────────────────

    def _start(self, m, job):
        dur = job.proc(m.id)
        m.idle, m.job, m.start_tick, m.dur, m.proc_start = False, job.id, self.k, dur, self.now
        # PhysicalMachine.TickProcessing: float32 remainingTime -= dt each tick from this one; the almost-done flag at
        # remainingTime <= PreDispatchLeadTime, the finished flag at <= 0, each harvested the next tick
        m.finish_tick = self.k + max(1, f32_countdown(dur, self.dt, until=0.0)[0])
        m.almost_tick = self.k + max(1, f32_countdown(dur, self.dt, until=self.lead)[0])
        for belt in m.belts:
            belt[1].discard(job.id)
        self._dispatchable.discard(self._mpos[m.id])
        self._wake(m.finish_tick)
        self._wake(m.almost_tick)

    def _finish(self, m):
        job = self.jobs[m.job]
        m.busy += self.now - m.proc_start
        m.proc_start, m.idle, m.job, m.finish_tick = None, True, -1, None
        if self._queued_at[m.id].keys:
            self._dispatchable.add(self._mpos[m.id])
        job.op_rows[-1]["end"] = self.now
        job.completed_ops += 1
        job.cur_op += 1
        job.location = m.id
        if job.completed_ops >= job.total_ops:
            if self.instant:
                self._exit(job)
            else:
                job.target = -1
                self._to(job, WAITING_PICKUP)
        else:
            self._to(job, NEEDS_ROUTING)

    def _exit(self, job):
        self._to(job, EXITED)
        job.exit_time = self.now
        job.location = -1

    def _dropoff_pos(self, m):
        """PhysicalMachine.GetDropoffPosition: the first incoming belt with room (primary when all are full)."""
        if not m.belts:
            return m.info.dropoff_pos
        for (pos, _), (cap, jobs) in zip(m.info.in_belts, m.belts):
            if len(jobs) < cap:
                return pos
        return m.info.in_belts[0][0]

    @staticmethod
    def _release_slot(m, jid):
        """PhysicalMachine.ReleaseIncomingSlot: a job re-routed off the machine without being started frees its slot."""
        for belt in m.belts:
            belt[1].discard(jid)

    def _place_on_belt(self, m, jid):
        for cap, jobs in m.belts:
            if len(jobs) < cap:
                jobs.add(jid)
                return

    # ── AGVs ────────────────────────────────────────────────────────────────────────────────────

    def _status(self, a):
        """Settles a returning AGV that reached its bay before this tick."""
        if a.state == "returning" and a.leg.arrive_tick <= self.k - 1:
            a.state = "idle"
            a.pos, a.yaw = a.leg.end_pos, a.leg.end_yaw
            a.zone = a.park_zone if self.floor.data.get("parking_bayed") else -1
            a.idle_since = (a.leg.arrive_tick) * self.dt
            a.leg = None
        return a.state

    def _on_duty(self, a):
        """The AGV may take a new task now (agv_schedule): its index is below the active fleet size."""
        sched = self.cfg.agv_schedule
        if not sched:
            return True
        n = len(self.agvs)
        for t, k in sched:
            if t > self.now:
                break
            n = k
        return self._agv_pos[id(a)] < n

    def _available(self, a):
        return self._status(a) in ("idle", "returning") and self._on_duty(a)

    def _pose(self, a, cancel):
        """Pose after agent tick k - 1. cancel applies CancelCurrentRoute: an AGV already inside the zone it was
        heading for commits to it before re-planning."""
        if a.state == "returning":
            x, z, yaw, zone, ri = a.leg.state_after(self.k - 1, self.kin)
            if cancel and ri < len(a.leg.route):
                ahead = a.leg.route[ri]
                if ahead != zone and self.floor.zone_by_id[ahead].contains(x, z):
                    zone = ahead
            return (x, z), yaw, zone
        zone = a.zone if a.zone >= 0 else self.floor.zone_at(*a.pos)
        return a.pos, a.yaw, zone

    def _nearest_agv(self, src):
        pickup_zones = src.info.pickup_zones if src is not None else [self.incoming_zone]
        hops = self.floor.hops_to(pickup_zones)
        # Nearest over idle AND returning AGVs, idle winning ties (AGVPool.GetNearestAvailableAGV since 2026-10-03;
        # before, any idle AGV beat a returning one).
        best, best_d, best_idle = None, math.inf, False
        for a in self.agvs:
            st = self._status(a)
            if st not in ("idle", "returning") or not self._on_duty(a):
                continue
            if st == "returning":
                zone = a.leg.state_after(self.k - 1, self.kin)[3]
            else:
                zone = a.zone if a.zone >= 0 else self.floor.zone_at(*a.pos)
            d = hops.get(zone, math.inf)
            idle = st == "idle"
            if best is None or d < best_d or (d == best_d and idle and not best_idle):
                best, best_d, best_idle = a, d, idle
        return best

    def _drive(self, a, route_to, final_pos, kind):
        pos, yaw, zone = self._pose(a, cancel=True)
        if a.state == "idle":
            a.idle_time += self.now - a.idle_since
        route = self.floor.route(zone, route_to)
        if not route:
            raise RuntimeError(f"AGV {a.id}: no route from zone {zone} to {route_to} at t={self.now:.2f}")
        leg = plan_leg(self.kin, self.floor, self.k, pos, yaw, zone, route, final_pos, kind)
        a.path += leg.path_length
        return leg

    def _pickup_dock(self, src):
        if src is not None:
            return self.floor.machine_dock(src.id, src.info.pickup_pos)
        return self.floor.special_dock(self.floor.incoming_key)

    def _dispatch(self, a, job, src, target, drop_pos):
        zid, dock = self._pickup_dock(src)
        leg = self._drive(a, zid, dock.approach, "dock")
        done, yaw = handshake_done(self.kin, leg.end_yaw, heading(*dock.facing), leg.arrive_tick, True)
        if self.fail is not None:
            a.hs = (leg.end_yaw, heading(*dock.facing), leg.arrive_tick, True)
        a.state, a.leg, a.dock = "to_pickup", leg, dock
        a.job, a.source, a.target, a.drop_pos = job.id, src.id if src else -1, target.id if target else -1, drop_pos
        a.pos, a.yaw, a.zone = leg.end_pos, yaw, leg.end_zone
        a.milestone = (done + 1, "pickup")
        self._wake(done + 1)
        self._trace(self.k, a, "dispatch", job.id)
        self._trace(leg.arrive_tick, a, "arrive_pickup", job.id)
        self._trace(done, a, "pickup", job.id)

    def _trace(self, tick, a, ev, jid):
        row = (round(tick * self.dt, 2), a.id, ev, jid)
        if self.fail is not None:          # per-AGV entries, so a cancelled task's future entries can be marked
            idx = self._agv_tr.setdefault(a.id, [])
            idx.append(len(self.trace))
            if len(idx) > 16:
                del idx[:8]
        if ev == "arrive_prepickup":
            self._prearrive.setdefault(row, []).append(len(self.trace))
        self.trace.append(row)

    def _predispatch(self, a, job, m):
        zid, dock = self._pickup_dock(m)
        leg = self._drive(a, zid, dock.approach, "dock")
        a.state, a.leg, a.dock = "to_prepickup", leg, dock
        a.job, a.source, a.target = job.id, m.id, -2
        a.pre_arrive = leg.arrive_tick
        a.pre_yaw = self.kin.rotate(leg.end_yaw, heading(*dock.facing))   # one turn on arrival
        a.pos, a.zone = leg.end_pos, leg.end_zone
        self._trace(self.k, a, "predispatch", job.id)
        self._trace(leg.arrive_tick, a, "arrive_prepickup", job.id)

    def _finalize(self, a, job, target, drop_pos):
        goal = heading(*a.dock.facing)
        if a.pre_arrive <= self.k - 1:   # already waiting at the dock: re-armed handshake from this tick
            done, yaw = handshake_done(self.kin, a.pre_yaw, goal, self.k, False)
            if self.fail is not None:
                a.hs = (a.pre_yaw, goal, self.k, False)
        else:                            # still on the way: the normal arrival
            done, yaw = handshake_done(self.kin, a.leg.end_yaw, goal, a.pre_arrive, True)
            if self.fail is not None:
                a.hs = (a.leg.end_yaw, goal, a.pre_arrive, True)
            pre = (round(a.pre_arrive * self.dt, 2), a.id, "arrive_prepickup", job.id)
            at = self._prearrive.get(pre)
            if at:                        # it now arrives as MovingToPickup, as Unity logs it (first match)
                self.trace[at.pop(0)] = pre[:2] + ("arrive_pickup", job.id)
        a.state, a.target, a.drop_pos, a.yaw = "to_pickup", target.id if target else -1, drop_pos, yaw
        a.milestone = (done + 1, "pickup")
        self._wake(done + 1)
        self._trace(self.k, a, "finalize", job.id)
        self._trace(done, a, "pickup", job.id)

    def _milestone(self, a):
        _, kind = a.milestone
        a.milestone = None
        job = self.jobs[a.job]
        if kind == "pickup":
            if job.state == WAITING_PICKUP:
                self._to(job, IN_TRANSIT)
            if a.target >= 0:
                zid, dock = self.floor.machine_dock(a.target, a.drop_pos)
            else:
                zid, dock = self.floor.special_dock(self.floor.outgoing_key)
            route = self.floor.route(a.zone, zid)
            if not route:
                raise RuntimeError(f"AGV {a.id}: no route to dropoff zone {zid}")
            leg = plan_leg(self.kin, self.floor, self.k, a.pos, a.yaw, a.zone, route, dock.approach, "dock")
            a.path += leg.path_length
            done, yaw = handshake_done(self.kin, leg.end_yaw, heading(*dock.facing), leg.arrive_tick, True)
            if self.fail is not None:
                a.hs = (leg.end_yaw, heading(*dock.facing), leg.arrive_tick, True)
            a.state, a.leg, a.dock = "to_dropoff", leg, dock
            a.pos, a.yaw, a.zone = leg.end_pos, yaw, leg.end_zone
            a.milestone = (done + 1, "dropoff")
            self._wake(done + 1)
            self._trace(leg.arrive_tick, a, "arrive_dropoff", job.id)
            self._trace(done, a, "dropoff", job.id)
            return
        # dropoff, done at agent tick k - 1: the belt takes the job there, the orchestrator sees it now
        a.trips += 1
        if a.target >= 0:
            m = self.mach[a.target]
            self._place_on_belt(m, job.id)
            job.location = m.id
            job.op_rows[-1]["queued"] = self.now
            self._to(job, QUEUED)
            if m.down:                    # delivered to a machine that is down (audit B3): re-routed
                self._to(job, NEEDS_ROUTING)
                self.fstats["delivered_to_down"] += 1
                if not self.cfg.legacy_failure_quirks:     # not started here: its belt slot is freed
                    self._release_slot(m, job.id)
        else:
            self._exit(job)
        job.assigned_agv = -1
        a.job = a.target = a.source = -1
        route = self.floor.route(a.zone, a.park_zone)
        if route:
            leg = plan_leg(self.kin, self.floor, self.k, a.pos, a.yaw, a.zone, route, a.park_pos, "park")
            a.path += leg.path_length
            a.state, a.leg = "returning", leg
            self._trace(leg.arrive_tick, a, "park(planned)", -1)
        else:
            a.state, a.leg, a.idle_since = "idle", None, self.now

    def _try_assign(self, job):
        """FlagHarvester.TryAssignAgv: the nearest available AGV takes the job; False when none is free."""
        if self.fail is not None and job.target >= 0 and self.mach[job.target].down:
            job.target = -1                # the routed destination went down: back to routing
            self._to(job, NEEDS_ROUTING)
            self.fstats["target_down_rerouted"] += 1
            return True
        src = self.mach[job.location] if job.location >= 0 else None
        a = self._nearest_agv(src)
        if a is None:
            return False
        target = self.mach[job.target] if job.target >= 0 else None
        drop_pos = self._dropoff_pos(target) if target else self.floor.outgoing_pos
        job.assigned_agv = a.id
        self._dispatch(a, job, src, target, drop_pos)
        return True

    def _assign_agvs(self):
        if self.instant:
            return
        for job in [j for j in self._by_state[WAITING_PICKUP].jobs() if j.assigned_agv == -1 and j.pre_agv < 0]:
            if not self._try_assign(job):
                break

    def _any_waiting_for_transport(self):
        if self.fail is not None and self.deferred:
            if any(j.id not in self.deferred for j in self._by_state[NEEDS_ROUTING].items.values()):
                return True
            return any(j.assigned_agv == -1 and j.pre_agv < 0 for j in self._by_state[WAITING_PICKUP].items.values())
        return len(self._by_state[NEEDS_ROUTING]) > 0 or \
            any(j.assigned_agv == -1 and j.pre_agv < 0 for j in self._by_state[WAITING_PICKUP].items.values())

    def _almost_done(self, m, waiting):
        if m.job < 0 or self.instant or m.down:
            return
        job = self.jobs[m.job]
        if job.state != PROCESSING or job.pre_agv >= 0 or job.completed_ops == job.total_ops - 1:
            return
        if self.on_transport and waiting:
            return
        a = self._nearest_agv(m)
        if a is None:
            return
        self._predispatch(a, job, m)
        job.pre_agv = a.id

    # ── Decisions ───────────────────────────────────────────────────────────────────────────────

    def _next_decision(self):
        ready = self._by_state[NEEDS_ROUTING].jobs()
        if ready:
            open_ = not self.on_transport or any(self._available(a) for a in self.agvs)
            if self.fail is None:
                routable = [j for j in ready if open_ or j.pre_agv >= 0]
            else:
                routable = []
                for j in ready:
                    if not any(mid in self.mach and not self.mach[mid].down for mid in j.ops[j.cur_op]):
                        if j.id not in self.deferred:    # every eligible machine is down: deferred
                            self.deferred[j.id] = self.now
                    elif open_ or j.pre_agv >= 0:
                        routable.append(j)
            if routable:
                yield from self._route(routable)
                return True
        if self._dispatchable:            # the first machine in floor order that is idle with a queue
            yield from self._dispatch_decision(self.machines[min(self._dispatchable)])
            return True
        return False

    def _rule_for(self, make_decision):
        """The rule halves for this decision: the fixed rule, or (agent mode, after the warm-up) the agent's."""
        if not self.agent or self.in_warmup:
            return self.job_rule, self.machine_rule
        rule = yield make_decision()
        job_rule, machine_rule = (r.upper() for r in rule)
        parse_rule(f"{job_rule}_{machine_rule}")
        return job_rule, machine_rule

    def candidate_machines(self, job):
        """BuildRoutingDecision: the job's eligible machines in floor order that are operational."""
        if self.fail is None:
            return [m.id for m in self.machines if m.id in job.ops[job.cur_op]]
        return [m.id for m in self.machines if m.id in job.ops[job.cur_op] and not m.down]

    def _route(self, routable):
        self.decisions += 1
        ids = [j.id for j in routable]
        job_rule, machine_rule = yield from self._rule_for(lambda: Decision(
            "routing", job=ids[0], job_candidates=tuple(ids), selected_by_rule=len(ids) == 1,
            machines=tuple(self.candidate_machines(routable[0]))))
        forced, self.forced = self.forced, None
        if forced is not None and forced[0] is not None:
            if forced[0] not in ids:
                raise ValueError(f"forced job {forced[0]} is not routable now ({ids})")
            jid = forced[0]
        else:
            jid = rank_jobs(job_rule, ids, self, lambda i: self.jobs[i].min_proc())
        job = self.jobs[jid]
        cands = self.candidate_machines(job)
        times = [job.proc(mid) for mid in cands]
        loads = [self.machine_load(mid) for mid in cands]
        utils = [self.utilization(self.mach[mid]) for mid in cands]
        if self.instant:
            travel = [f32(0)] * len(cands)
        else:
            speed = self.floor.agv["speed"]
            travel = [f32(self.floor.estimate_path_length(job.location, mid) / speed) for mid in cands]
        if forced is not None and forced[1] is not None:
            if forced[1] not in cands:
                raise ValueError(f"forced machine {forced[1]} is not eligible for job {jid} ({cands})")
            mid = forced[1]
        elif machine_rule in RELIABILITY_RULES:
            mid = select_machine(machine_rule, cands, times, loads, utils, travel,
                                 rel=self.reliability_terms(cands))
        elif machine_rule in PTIME_RULES:
            mid = select_machine(machine_rule, cands, times, loads, utils, travel, pt=self.ptime_params())
        else:
            mid = select_machine(machine_rule, cands, times, loads, utils, travel)
        self.decision_rows.append((self.now, "routing", job.id, mid, len(cands), len(ids)))
        job.op_rows.append({"job": job.id, "op": job.cur_op, "machine": mid, "routed": self.now,
                            "from": job.location, "queued": None, "start": None, "end": None})
        if self.fail is not None:
            job.op_rows[-1]["interrupted"] = None
        job.target = mid
        self._to(job, WAITING_PICKUP)
        if self.instant:
            job.location = mid
            job.op_rows[-1]["queued"] = self.now
            self._to(job, QUEUED)
        elif job.pre_agv >= 0:
            a = self.agvs[job.pre_agv]
            target = self.mach[mid]
            self._finalize(a, job, target, self._dropoff_pos(target))
            job.assigned_agv, job.pre_agv = a.id, -1
        elif self.on_transport:
            self._try_assign(job)
        else:
            self._wake(self.k + 1)       # onReady: AssignAGVs picks it up next tick

    def _dispatch_decision(self, m):
        self.decisions += 1
        queue = [j.id for j in self._queued_at[m.id].jobs()]
        job_rule, _ = yield from self._rule_for(lambda: Decision("dispatch", machine=m.id, queue=tuple(queue)))
        forced, self.forced = self.forced, None
        if forced is not None and forced[0] is not None:
            if forced[0] not in queue:
                raise ValueError(f"forced job {forced[0]} is not queued at machine {m.id} ({queue})")
            jid = forced[0]
        else:
            jid = rank_jobs(job_rule, queue, self, lambda i: self.jobs[i].proc(m.id))
        job = self.jobs[jid]
        self.decision_rows.append((self.now, "dispatch", m.id, jid, len(queue), 0))
        job.op_rows[-1]["start"] = self.now
        self._to(job, PROCESSING)
        self._start(m, job)

    # ── Machine failures (FailureCoordinator) ───────────────────────────────────────────────────────

    def _init_failures(self):
        tr = self.fail
        self.deferred = {}                 # JobStore.DeferredJobIds (job id -> time first deferred), insertion order
        self._orphans = False              # a pre-dispatch claim was cleared: ReleaseOrphanedPreDispatches runs
        self._agv_tr = {}
        self.fail_log = []                 # (time, machine, event, job / -1, repair duration)
        # Unity's machine_failures.csv rows (FailureCoordinator, "-destrace"), same columns
        self.fail_rows = []
        self.fstats = {k: 0 for k in ("jobs_interrupted", "queued_rerouted", "pickups_cancelled", "redirects",
                                       "aborts", "held_to_down", "redirect_no_route", "predispatch_cleared",
                                       "orphans_released", "delivered_to_down", "target_down_rerouted")}
        self.deferred_seconds = 0.0
        self.reliability_info = (OnlineReliability(self, self.cfg.reliability_half_life, self.cfg.reliability_prior)
                                 if self.cfg.reliability == "online" else None)
        for m in self.machines:
            mt = tr.machines.get(m.id)
            m.events = mt.events if mt is not None else []
            m.fptr, m.n_fail, m.repair_sum, m.downtime, m.down_since, m.lost_work = 0, 0, 0.0, 0.0, None, 0.0
            m.repair_dur, m.repair_start = 0.0, None
            m.life_start, m.age0 = 1, (mt.age0 if mt is not None else 0.0)
            if m.events:
                m.fail_tick = max(1, m.events[0][0])
                self._wake(m.fail_tick)

    def replace_failures(self, trace):
        """Swaps the failure trace mid-episode, e.g. for a rollout over a redrawn future
        (FailureTrace.redraw_after(twin.k, seed)): failures already harvested and a repair in progress stay as they
        are; each up machine's next failure comes from the new trace."""
        if self.fail is None:
            raise ValueError("the twin was built without failures (TwinConfig.failures is None)")
        self.fail = trace
        for m in self.machines:
            mt = trace.machines.get(m.id)
            m.events = mt.events if mt is not None else []
            m.fptr = sum(1 for f, _ in m.events if f <= self.k)
            if m.down:
                m.fail_tick = None
            else:
                nxt = m.events[m.fptr][0] if m.fptr < len(m.events) else None
                m.fail_tick = None if nxt is None else max(nxt, self.k + 1)
                if m.fail_tick is not None:
                    self._wake(m.fail_tick)

    def _harvest_failures(self, k):
        """FailureCoordinator.HarvestFailureFlags: machine order; a failure, else a repair (then RetryDeferredJobs)."""
        for m in self.machines:
            if m.fail_tick == k:
                self._fail(m)
            elif m.repair_tick == k:
                self._repair_done(m)
                self._retry_deferred()
                self.fail_rows[-1]["deferred_jobs"] = len(self.deferred)

    def downtime_so_far(self, m):
        """EpisodeTracker.DowntimeSoFar: closed repair intervals plus the open one."""
        return m.downtime + (self.now - m.down_since if m.down_since is not None else 0.0)

    def remaining_repair(self, m):
        """PhysicalMachine.RemainingRepairTime at this tick (counted down from the harvest tick on)."""
        if not m.down:
            return 0.0
        return remaining_countdown(m.repair_dur, self.dt, self.k - m.repair_start)

    def machine_age(self, m):
        """Operating age since the last renewal (Unity's _ageSinceLastRepair), seconds; 0 while down."""
        if m.down:
            return 0.0
        return (m.age0 if m.life_start == 1 else 0.0) + (self.k - m.life_start) * self.dt

    def _fail(self, m):
        """FailureCoordinator.HandleMachineFailure, steps 0-6, at the harvest tick."""
        k = self.k
        _, rep = m.events[m.fptr]
        m.fptr += 1
        m.fail_tick = None
        m.n_fail += 1
        m.repair_sum += rep
        m.down, m.down_since = True, self.now
        m.repair_dur, m.repair_start = rep, k
        m.repair_tick = harvest_tick(k, rep, self.dt)
        self._wake(m.repair_tick)
        self.fail_log.append((self.now, m.id, "fail", m.job, rep))
        row = self._fail_row(m, "fail", m.job, rep)
        if self.reliability_info is not None:
            self.reliability_info.on_fail(m)
        # 0. the interrupted operation's elapsed time is not busy time (G10)
        if not m.idle and m.proc_start is not None:
            m.lost_work += self.now - m.proc_start
        # 1. the job in process restarts: NeedsRouting at the failed machine (picked up from its dock later)
        if m.job >= 0:
            job = self.jobs[m.job]
            if job.state == PROCESSING:
                job.op_rows[-1]["interrupted"] = self.now
                job.location = m.id
                self._to(job, NEEDS_ROUTING)
                self.fstats["jobs_interrupted"] += 1
        # AcknowledgeFailure's machine reset (done before the other steps here; nothing below reads it)
        m.idle, m.job, m.proc_start, m.dur = True, -1, None, f32(0)
        m.finish_tick = m.almost_tick = None          # no stale finish / almost-done fires (TickProcessing stopped)
        self._dispatchable.discard(self._mpos[m.id])
        # 2. jobs queued at the machine are re-routed and free their incoming-belt slots (PhysicalMachine.
        # ReleaseIncomingSlot; with legacy_failure_quirks they keep them, as Unity did before 2026-10-09)
        legacy = self.cfg.legacy_failure_quirks
        for job in self._queued_at[m.id].jobs():
            self._to(job, NEEDS_ROUTING)
            if not legacy:
                self._release_slot(m, job.id)
            self.fstats["queued_rerouted"] += 1
            row["queued_rerouted"] += 1
        # 3. pickups bound for the machine: the AGV turns back, the job is re-routed. An AGV whose handshake ended
        # last tick is already MovingToDropoff in Unity (its PickedUpFlag is harvested after this step): skipped.
        for a in self.agvs:
            if a.state != "to_pickup" or (a.milestone is not None and a.milestone[0] == k):
                continue
            job = self.jobs[a.job]
            if job.state != WAITING_PICKUP or job.target != m.id:
                continue
            self._abandon(a, "cancel")
            job.target, job.assigned_agv = -1, -1
            self._to(job, NEEDS_ROUTING)
            self.fstats["pickups_cancelled"] += 1
            row["pickups_cancelled"] += 1
        # 4. deliveries in transit to the machine: redirected to the least-loaded operational eligible machine
        # (a fixed heuristic, not the policy). With no alternate the delivery carries on to the down machine, where
        # the B3 path re-routes the job on arrival (legacy_failure_quirks: aborted, the AGV goes home and the job
        # keeps its previous location). An AGV that unloaded last tick is already returning in Unity: skipped (its
        # delivery is re-routed by B3 in the AGV harvest).
        for a in self.agvs:
            if a.state != "to_dropoff" or (a.milestone is not None and a.milestone[0] == k):
                continue
            job = self.jobs[a.job]
            if job.state != IN_TRANSIT or job.target != m.id:
                continue
            alt = self._best_alternate(job, m.id)
            if alt is None and not legacy:
                self.fstats["held_to_down"] += 1
                row["held"] += 1
                continue
            if alt is not None:
                job.target = alt
                self._index_load(job)
                job.op_rows[-1]["machine"] = alt   # the operation now goes to the alternate
                if self._redirect(a, job, self.mach[alt]):
                    self.fstats["redirects"] += 1
                    row["redirected"] += 1
                    continue
                self.fstats["redirect_no_route"] += 1
            job.target, job.assigned_agv = -1, -1
            self._to(job, NEEDS_ROUTING)          # location unchanged (Unity drops the job where the AGV is)
            row["aborted"] += 1
            if alt is None:
                self._abandon(a, "abort")
                self.fstats["aborts"] += 1
        # 5. pre-dispatches aimed at the machine lose their claim; the AGV is released after the AGV harvest
        for job in self._live_d.values():
            if job.pre_agv >= 0 and job.target == m.id:
                job.pre_agv = -1
                self._orphans = True
                self.fstats["predispatch_cleared"] += 1
                row["predispatch_cleared"] += 1

    def _fail_row(self, m, event, job, rep):
        """A machine_failures.csv row (FailureCoordinator.EventRecords), appended to fail_rows."""
        row = {"sim_time": round(self.now, 2), "tick": self.k, "machine_id": m.id, "event": event, "active_job": job,
               "repair_duration": float(np.float32(rep)), "queued_rerouted": 0, "pickups_cancelled": 0,
               "redirected": 0, "held": 0, "aborted": 0, "predispatch_cleared": 0, "deferred_jobs": 0}
        self.fail_rows.append(row)
        return row

    def _best_alternate(self, job, failed):
        """FailureCoordinator.FindBestAlternateMachine: the operational eligible machine (eligible-list order) with
        the lowest GetMachineLoad, first minimum."""
        best, best_load = None, f32(np.finfo(np.float32).max)
        for mid in job.ops[job.cur_op]:
            if mid == failed or mid not in self.mach or self.mach[mid].down:
                continue
            load = self.machine_load(mid)
            if load < best_load:
                best, best_load = mid, load
        return best

    def _repair_done(self, m):
        """HandleMachineRepairComplete: operational again, a fresh life from this tick."""
        k = self.k
        m.down, m.repair_tick = False, None
        m.downtime += self.now - m.down_since
        m.down_since = None
        m.life_start = k
        self.fail_log.append((self.now, m.id, "repair_done", -1, m.repair_dur))
        self._fail_row(m, "repair_done", -1, m.repair_dur)
        if self.reliability_info is not None:
            self.reliability_info.on_repair(m)
        if m.fptr < len(m.events):
            m.fail_tick = max(m.events[m.fptr][0], k + 1)
            self._wake(m.fail_tick)
        if m.idle and self._queued_at[m.id].keys:
            self._dispatchable.add(self._mpos[m.id])

    def _retry_deferred(self):
        """FailureCoordinator.RetryDeferredJobs: a deferred job with an operational eligible machine is routable."""
        for jid in list(self.deferred):
            job = self.jobs[jid]
            if job.state == EXITED or job.cur_op >= job.total_ops:
                del self.deferred[jid]
                continue
            if any(mid in self.mach and not self.mach[mid].down for mid in job.ops[job.cur_op]):
                self.deferred_seconds += self.now - self.deferred.pop(jid)

    def _leg_pose(self, a):
        """Pose after agent tick k - 1 on the AGV's current leg, with CancelCurrentRoute's commit to a zone it is
        already inside, and the leg's undriven length (path_length was added in full when the leg was planned)."""
        leg = a.leg
        x, z, yaw, zone, ri = leg.state_after(self.k - 1, self.kin)
        if self.k - 1 >= leg.arrive_tick:      # at the dock (handshake): turned toward the dock as far as by now
            if a.state == "to_prepickup":
                yaw = a.pre_yaw
            elif a.hs is not None:
                yaw = dock_yaw_after(self.kin, a.hs[0], a.hs[1], a.hs[2], self.k - 1, a.hs[3])
            else:
                yaw = a.yaw
        if ri < len(leg.route):
            ahead = leg.route[ri]
            if ahead != zone and self.floor.zone_by_id[ahead].contains(x, z):
                zone = ahead
        driven = 0.0
        for ph in leg.phases:
            if ph[0] != "drive" or ph[1] > self.k - 1:
                continue
            n = min(self.k - 1, ph[2]) - ph[1] + 1
            driven += min(n * self.kin.step, ph[8])
        return (x, z), yaw, zone, max(0.0, leg.path_length - driven)

    def _mark_cancelled(self, a):
        """The AGV's trace entries for ticks it will now never reach are marked "(cancelled)"."""
        t_now = round(self.k * self.dt, 2)
        for i in self._agv_tr.get(a.id, ()):
            row = self.trace[i]
            if row[0] >= t_now and row[1] == a.id and not row[2].endswith("(cancelled)"):
                if row[2] == "arrive_prepickup":
                    lst = self._prearrive.get(row)
                    if lst and i in lst:
                        lst.remove(i)
                self.trace[i] = row[:2] + (row[2] + "(cancelled)", row[3])

    def _abandon(self, a, event):
        """CancelPickup / AbortTransit / ReleaseOrphanedPrePickup: the task is dropped and the AGV drives back to its
        bay from where it is."""
        pos, yaw, zone, undriven = self._leg_pose(a)
        a.path -= undriven
        self._mark_cancelled(a)
        self._trace(self.k, a, event, a.job)
        a.job = a.source = a.target = -1
        a.milestone, a.dock, a.drop_pos, a.pre_arrive = None, None, None, None
        a.pos, a.yaw, a.zone = pos, yaw, zone
        route = self.floor.route(zone, a.park_zone)
        if route:
            leg = plan_leg(self.kin, self.floor, self.k, pos, yaw, zone, route, a.park_pos, "park")
            a.path += leg.path_length
            a.state, a.leg = "returning", leg
            self._trace(leg.arrive_tick, a, "park(planned)", -1)
        else:
            a.state, a.leg, a.idle_since = "idle", None, self.now

    def _redirect(self, a, job, target):
        """AGVController.RedirectDropoff: re-plan from where the AGV is to the new machine's dropoff dock. False when
        there is no route (Unity: FullReset; here the AGV drives back to its bay)."""
        pos, yaw, zone, undriven = self._leg_pose(a)
        drop_pos = self._dropoff_pos(target)
        zid, dock = self.floor.machine_dock(target.id, drop_pos)
        route = self.floor.route(zone, zid) if dock is not None else None
        if not route:
            self._abandon(a, "redirect_failed")
            return False
        a.path -= undriven
        self._mark_cancelled(a)
        self._trace(self.k, a, "redirect", job.id)
        leg = plan_leg(self.kin, self.floor, self.k, pos, yaw, zone, route, dock.approach, "dock")
        a.path += leg.path_length
        done, yaw2 = handshake_done(self.kin, leg.end_yaw, heading(*dock.facing), leg.arrive_tick, True)
        a.hs = (leg.end_yaw, heading(*dock.facing), leg.arrive_tick, True)
        a.leg, a.dock, a.target, a.drop_pos = leg, dock, target.id, drop_pos
        a.pos, a.yaw, a.zone = leg.end_pos, yaw2, leg.end_zone
        a.milestone = (done + 1, "dropoff")
        self._wake(done + 1)
        self._trace(leg.arrive_tick, a, "arrive_dropoff", job.id)
        self._trace(done, a, "dropoff", job.id)
        return True

    def _release_orphans(self):
        """FlagHarvester.ReleaseOrphanedPreDispatches: a pre-dispatched AGV whose job no longer claims it goes back."""
        self._orphans = False
        for a in self.agvs:
            if a.state == "to_prepickup" and self.jobs[a.job].pre_agv != a.id:
                self._abandon(a, "orphan")
                self.fstats["orphans_released"] += 1

    # ── Reliability information for the reliability-aware rules ──────────────────────────────────────

    def _rel_info(self, mid, info):
        """(k, lam, mtbf, mttr, age or None) of one machine at an information level."""
        if self.fail is None or self.fail.model is None:
            return 1.0, math.inf, math.inf, 0.0, 0.0
        if info == "static":
            return self.fail.model.static_rel(mid) + (self.machine_age(self.mach[mid]),)
        m = self.mach[mid]
        if info == "oracle":
            return self.fail.model.current_rel(mid, self.now, self.fail) + (self.machine_age(m),)
        if info == "online":
            return self.reliability_info.estimate(m) + (self.machine_age(m),)
        raise ValueError(f"reliability information must be static / online / oracle, not {info!r}")

    def ptime_params(self):
        """Parameters of the processing-time-weighted machine rules (rules.select_machine's pt=)."""
        c = self.cfg
        return {"alpha": c.ptime_alpha, "shape": c.ptime_shape, "tau": c.ptime_tau, "reroute": c.ptime_reroute,
                "avail": c.ptime_avail}

    def reliability_terms(self, cands):
        """Inputs of the reliability-aware machine rules (rules.select_machine's rel=): per candidate (A, mtbf, mttr,
        k, lam, age) at the information level cfg.reliability, the floor's median availability, kappa and the
        re-route cost. Without failures every machine is perfectly reliable, so the rules reduce to ECT / TECT."""
        info = self.cfg.reliability
        omega = self.cfg.reliability_mix
        terms = []
        for mid in cands:
            k, lam, mtbf_, mttr_, age = self._rel_info(mid, info)
            a = mtbf_ / (mtbf_ + mttr_) if math.isfinite(mtbf_) else 1.0
            terms.append((a, mtbf_, mttr_, k, lam, age))
        avail, rates, mttrs = [], [], []
        for m in self.machines:
            _, _, mtbf_, mttr_, _ = self._rel_info(m.id, info)
            avail.append(mtbf_ / (mtbf_ + mttr_) if math.isfinite(mtbf_) else 1.0)
            rates.append(1.0 / mtbf_)
            mttrs.append(mttr_)
        if omega != 1.0 and self.fail is not None and self.fail.model is not None:
            # shrink each candidate's failure rate and repair time towards the floor mean at this information level
            r_floor, m_floor = sum(rates) / len(rates), sum(mttrs) / len(mttrs)
            mixed = []
            for a, mtbf_, mttr_, k, lam, age in terms:
                rate = omega / mtbf_ + (1.0 - omega) * r_floor
                mt = omega * mttr_ + (1.0 - omega) * m_floor
                mb = 1.0 / rate if rate > 0 else math.inf
                mixed.append((mb / (mb + mt) if math.isfinite(mb) else 1.0, mb, mt, k,
                              mb / math.gamma(1.0 + 1.0 / k) if math.isfinite(mb) else math.inf, age))
            terms = mixed
        avail.sort()
        n = len(avail)
        median = avail[n // 2] if n % 2 else 0.5 * (avail[n // 2 - 1] + avail[n // 2])
        return {"terms": terms, "kappa": self.cfg.reliability_kappa, "reroute": self.cfg.reroute_cost,
                "median_A": median}

    def machine_rows(self):
        """Per machine: failures, sampled repair time, downtime, availability, lost work (Unity's machine CSV)."""
        rows = []
        for m in self.machines:
            on = self.fail is not None
            down = self.downtime_so_far(m) if on else 0.0
            rows.append({"machine_id": m.id, "machine_type": m.info.type,
                         "failure_count": m.n_fail if on else 0, "total_repair_time": m.repair_sum if on else 0.0,
                         "downtime": down, "availability_rate": 1.0 - down / self.now if self.now > 0 else 1.0,
                         "time_processing": m.busy, "lost_work": m.lost_work if on else 0.0})
        return rows

    def failure_summary(self):
        """Episode failure metrics (names as Unity's results.csv where it has them)."""
        rows = self.machine_rows()
        s = {"episode_failures": sum(r["failure_count"] for r in rows),
             "total_repair_time": sum(r["total_repair_time"] for r in rows),
             "machine_downtime": sum(r["downtime"] for r in rows),
             "machine_availability_mean": sum(r["availability_rate"] for r in rows) / len(rows) if rows else 1.0,
             "lost_work": sum(r["lost_work"] for r in rows)}
        if self.fail is not None:
            s.update(self.fstats)
            s["deferred_job_seconds"] = self.deferred_seconds + sum(self.now - t for t in self.deferred.values())
        return s

    # ── Results ─────────────────────────────────────────────────────────────────────────────────

    def summary(self):
        done = [j for j in self.jobs.values() if j.exit_time is not None]
        flows = sorted(j.exit_time - j.arrival for j in done)

        def pct(p):
            if not flows:
                return 0.0
            return flows[min(max(math.ceil(p * len(flows)) - 1, 0), len(flows) - 1)]

        n = len(done)
        for a in self.agvs:
            if self._status(a) == "idle":
                a.idle_time += self.makespan - a.idle_since
                a.idle_since = self.makespan
        s = {
            "rule": self.cfg.rule,
            "transport": self.cfg.transport,
            "agv_count": len(self.agvs),
            "makespan": self.makespan,
            "timed_out": self.timed_out,
            "jobs": len(self.jobs),
            "jobs_completed": n,
            "decisions": self.decisions,
            "mean_flow_time": sum(flows) / n if n else 0.0,
            "p95_flow_time": pct(0.95),
            "max_flow_time": flows[-1] if flows else 0.0,
            "mean_transport_wait": (sum(j.t_state[WAITING_PICKUP] + j.t_state[IN_TRANSIT] for j in done) / n) if n else 0.0,
            "mean_time_queued": (sum(j.t_state[QUEUED] for j in done) / n) if n else 0.0,
            "agv_path_length": sum(a.path for a in self.agvs),
            "agv_trips": sum(a.trips for a in self.agvs),
            "agv_busy_fraction": (1.0 - sum(a.idle_time for a in self.agvs) / (len(self.agvs) * self.makespan))
            if self.agvs and self.makespan else 0.0,
            "machine_util_mean": sum(m.busy for m in self.machines) / (len(self.machines) * self.makespan)
            if self.makespan else 0.0,
        }
        if self.fail is not None:
            s.update(self.failure_summary())
        return s

    def metrics(self):
        """RewardMetrics.Fill (env/rewards/metrics.py names) for the current state, as float32 like the sensor.
        AGV travel time is the non-idle time (handshakes included) and there are no route waits or zone blocks:
        the twin has no reservations."""
        jobs = self.order                 # AllJobs: arrived jobs only
        # The sums are a left fold over self.order. Its leading run of exited jobs never changes, so the fold up to
        # the first live job is cached (_mprefix) and only the rest is re-summed: the same additions in the same
        # order, so the same bits.
        pre = self._mprefix
        if pre is None:
            pre = self._mprefix = [0, 0, 0.0, 0.0, 0, 0, 0, [0.0] * 6, 0, 0.0, 0.0]
        p, with_due, tard, tard_exited, late_exited, ops_total, ops_done, t_state, exited, flow, tis = pre
        t_state = list(t_state)
        while p < len(jobs) and jobs[p].state == EXITED:
            j = jobs[p]
            if j.due is not None:
                with_due += 1
                late = j.exit_time - j.due
                if late > 0:
                    tard += late
                    tard_exited += late
                    late_exited += 1
            ops_total += j.total_ops
            ops_done += j.completed_ops
            for s in range(5):
                t_state[s] += j.t_state[s]
            exited += 1
            f = j.exit_time - j.arrival
            flow += f
            tis += f
            p += 1
        self._mprefix = [p, with_due, tard, tard_exited, late_exited, ops_total, ops_done, list(t_state), exited,
                         flow, tis]
        n_state = [0] * 6
        late_open = 0
        for j in jobs[p:]:
            if j.due is not None:
                with_due += 1
                if j.state == EXITED:
                    late = j.exit_time - j.due
                    if late > 0:
                        tard += late
                        tard_exited += late
                        late_exited += 1
                else:
                    late = self.now - j.due
                    if late > 0:
                        tard += late
                        late_open += 1
            ops_total += j.total_ops
            ops_done += j.completed_ops
            for s in range(5):
                t_state[s] += j.t_state[s]
            if j.state == EXITED:
                exited += 1
                f = j.exit_time - j.arrival
                flow += f
                tis += f
            else:
                n_state[j.state] += 1
                t_state[j.state] += max(0.0, self.now - j.since)
                tis += max(0.0, self.now - j.arrival)
        idle_agvs = 0
        idle_time = 0.0
        for a in self.agvs:
            idle_time += a.idle_time
            if self._status(a) == "idle":
                idle_agvs += 1
                idle_time += max(0.0, self.now - a.idle_since)
        n_agv = len(self.agvs)
        values = {
            "sim_time": self.now, "episode_active": 0.0 if self.done else 1.0, "decision_count": self.decisions,
            "jobs_total": len(jobs), "jobs_exited": exited, "wip": len(jobs) - exited,
            "ops_total": ops_total, "ops_completed": ops_done,
            "flow_time_exited_sum": flow, "time_in_system_sum": tis,
            "jobs_needs_routing": n_state[NEEDS_ROUTING], "jobs_waiting_pickup": n_state[WAITING_PICKUP],
            "jobs_in_transit": n_state[IN_TRANSIT], "jobs_queued": n_state[QUEUED],
            "jobs_processing": n_state[PROCESSING],
            "time_needs_routing_sum": t_state[NEEDS_ROUTING], "time_waiting_pickup_sum": t_state[WAITING_PICKUP],
            "time_in_transit_sum": t_state[IN_TRANSIT], "time_queued_sum": t_state[QUEUED],
            "time_processing_sum": t_state[PROCESSING],
            "machines_total": len(self.machines), "machines_busy": sum(not m.idle for m in self.machines),
            "agvs_total": n_agv, "agvs_idle": idle_agvs,
            "agv_time_traveling_sum": n_agv * self.now - idle_time, "agv_time_idle_sum": idle_time,
            "agv_trips_total": sum(a.trips for a in self.agvs),
            "timed_out": float(self.timed_out),
            "all_jobs_exited": float(len(jobs) > 0 and exited == len(jobs)),
            "truncated": float(self.truncated),
            "tardiness_sum": tard, "tardiness_exited_sum": tard_exited, "jobs_late": late_open,
            "jobs_exited_late": late_exited, "jobs_with_due_date": with_due,
        }
        return {k: float(np.float32(v)) for k, v in values.items()}

    def job_rows(self):
        return [{"job_id": j.id, "arrival_time": j.arrival, "exit_time": j.exit_time,
                 "flow_time": (j.exit_time - j.arrival) if j.exit_time is not None else None,
                 "total_operations": j.total_ops, "completed_ops": j.completed_ops,
                 "time_needs_routing": j.t_state[NEEDS_ROUTING], "time_waiting_pickup": j.t_state[WAITING_PICKUP],
                 "time_in_transit": j.t_state[IN_TRANSIT], "time_queued": j.t_state[QUEUED],
                 "time_processing": j.t_state[PROCESSING], "due_date": j.due,
                 "tardiness": (max(0.0, j.exit_time - j.due) if j.exit_time is not None and j.due is not None
                               else None)} for j in self.jobs.values()]

    def op_rows(self):
        return [r for j in self.jobs.values() for r in j.op_rows]


def run_twin(floor, jobs_data, cfg):
    """Runs one episode; returns the finished Twin (summary(), job_rows(), op_rows(), decision_rows)."""
    if not isinstance(floor, Floor):
        floor = Floor(floor)
    return Twin(floor, jobs_data, cfg).run()
