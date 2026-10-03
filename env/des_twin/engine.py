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
raised at tick m is harvested at m + 1. Machine failures, AGV failures and multi-tile floors are not modelled:
compare only against Unity runs without them.
"""
import heapq
import math
from dataclasses import dataclass

import numpy as np

from .floor import Floor
from .motion import Kinematics, f32_countdown, handshake_done, heading, plan_leg
from .rules import f32, parse_rule, rank_jobs, select_machine

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


class Job:
    __slots__ = ("id", "arrival", "ops", "types", "cur_op", "completed_ops", "state", "since", "t_state",
                 "location", "target", "assigned_agv", "pre_agv", "exit_time", "op_rows")

    def __init__(self, jid, arrival, ops, types):
        self.id, self.arrival, self.ops, self.types = jid, arrival, ops, types
        self.cur_op = self.completed_ops = 0
        self.state, self.since = NEEDS_ROUTING, arrival
        self.t_state = [0.0] * 6
        self.location = self.target = self.assigned_agv = self.pre_agv = -1
        self.exit_time = None
        self.op_rows = []

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
                 "almost_tick", "info")

    def __init__(self, info):
        self.info, self.id = info, info.id
        self.idle, self.job = True, -1
        self.start_tick, self.dur, self.proc_start, self.busy = 0, f32(0), None, 0.0
        self.belts = [[cap, set()] for _, cap in info.in_belts]
        self.finish_tick = self.almost_tick = None


class AgvRT:
    __slots__ = ("id", "park_pos", "park_zone", "state", "pos", "yaw", "zone", "leg", "job", "source", "target",
                 "drop_pos", "dock", "milestone", "pre_arrive", "pre_yaw", "path", "idle_since", "idle_time",
                 "trips")

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
        self.incoming_zone = floor.incoming_zone()

        self.jobs, self.order, self.pending = {}, [], []
        for jd in jobs_data["jobs"]:
            ops = [{int(m): f32(d) for m, d in op["eligible"]} for op in jd["ops"]]
            job = Job(int(jd["id"]), float(jd["arrival"]), ops, [op["type"] for op in jd["ops"]])
            self.jobs[job.id] = job
            if job.arrival > 0.0:
                self.pending.append(job)
            else:
                self.order.append(job)
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
        for m in self.machines:
            if m.finish_tick == k:
                self._finish(m)
        for a in self.agvs:
            if a.milestone is not None and a.milestone[0] == k:
                self._milestone(a)
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
            self.order.append(job)
            added = True
        if added:
            self._wake(k + 1)
        return self.pending_i >= len(self.pending) and all(j.state == EXITED for j in self.order)

    # ── Job bookkeeping ─────────────────────────────────────────────────────────────────────────

    def _to(self, job, state):
        dt = self.now - job.since
        if dt > 0:
            job.t_state[job.state] += dt
        job.state, job.since = state, self.now

    def _live(self):
        return (j for j in self.order if j.state != EXITED)

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
        for j in self._live():
            if (j.state == QUEUED and j.location == mid) or \
                    (j.target == mid and j.state in (WAITING_PICKUP, IN_TRANSIT)):
                load = f32(load + j.proc(mid))
        return load

    def all_machine_loads(self):
        loads = {m.id: self.remaining_proc(m) for m in self.machines}
        for j in self._live():
            if j.state == QUEUED:
                mid = j.location
            elif j.state in (WAITING_PICKUP, IN_TRANSIT):
                mid = j.target
            else:
                continue
            if mid >= 0:
                loads[mid] = f32(loads.get(mid, f32(0)) + j.proc(mid))
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
        busy = m.busy + (self.now - m.proc_start if m.proc_start is not None else 0.0)
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
        self._wake(m.finish_tick)
        self._wake(m.almost_tick)

    def _finish(self, m):
        job = self.jobs[m.job]
        m.busy += self.now - m.proc_start
        m.proc_start, m.idle, m.job, m.finish_tick = None, True, -1, None
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

    def _available(self, a):
        return self._status(a) in ("idle", "returning")

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
            if st not in ("idle", "returning"):
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
        a.state, a.leg, a.dock = "to_pickup", leg, dock
        a.job, a.source, a.target, a.drop_pos = job.id, src.id if src else -1, target.id if target else -1, drop_pos
        a.pos, a.yaw, a.zone = leg.end_pos, yaw, leg.end_zone
        a.milestone = (done + 1, "pickup")
        self._wake(done + 1)
        self._trace(self.k, a, "dispatch", job.id)
        self._trace(leg.arrive_tick, a, "arrive_pickup", job.id)
        self._trace(done, a, "pickup", job.id)

    def _trace(self, tick, a, ev, jid):
        self.trace.append((round(tick * self.dt, 2), a.id, ev, jid))

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
        else:                            # still on the way: the normal arrival
            done, yaw = handshake_done(self.kin, a.leg.end_yaw, goal, a.pre_arrive, True)
            pre = (round(a.pre_arrive * self.dt, 2), a.id, "arrive_prepickup", job.id)
            if pre in self.trace:         # it now arrives as MovingToPickup, as Unity logs it
                self.trace[self.trace.index(pre)] = pre[:2] + ("arrive_pickup", job.id)
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
        for job in [j for j in self._live() if j.state == WAITING_PICKUP and j.assigned_agv == -1 and j.pre_agv < 0]:
            if not self._try_assign(job):
                break

    def _any_waiting_for_transport(self):
        return any(j.state == NEEDS_ROUTING or
                   (j.state == WAITING_PICKUP and j.assigned_agv == -1 and j.pre_agv < 0) for j in self._live())

    def _almost_done(self, m, waiting):
        if m.job < 0 or self.instant:
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
        ready = [j for j in self._live() if j.state == NEEDS_ROUTING]
        if ready:
            open_ = not self.on_transport or any(self._available(a) for a in self.agvs)
            routable = [j for j in ready if open_ or j.pre_agv >= 0]
            if routable:
                yield from self._route(routable)
                return True
        for m in self.machines:
            if m.idle and any(j.state == QUEUED and j.location == m.id for j in self._live()):
                yield from self._dispatch_decision(m)
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
        """BuildRoutingDecision: the job's eligible machines in floor order (all operational in the twin)."""
        return [m.id for m in self.machines if m.id in job.ops[job.cur_op]]

    def _route(self, routable):
        self.decisions += 1
        ids = [j.id for j in routable]
        job_rule, machine_rule = yield from self._rule_for(lambda: Decision(
            "routing", job=ids[0], job_candidates=tuple(ids), selected_by_rule=len(ids) == 1,
            machines=tuple(self.candidate_machines(routable[0]))))
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
        mid = select_machine(machine_rule, cands, times, loads, utils, travel)
        self.decision_rows.append((self.now, "routing", job.id, mid, len(cands), len(ids)))
        job.op_rows.append({"job": job.id, "op": job.cur_op, "machine": mid, "routed": self.now,
                            "from": job.location, "queued": None, "start": None, "end": None})
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
        queue = [j.id for j in self._live() if j.state == QUEUED and j.location == m.id]
        job_rule, _ = yield from self._rule_for(lambda: Decision("dispatch", machine=m.id, queue=tuple(queue)))
        jid = rank_jobs(job_rule, queue, self, lambda i: self.jobs[i].proc(m.id))
        job = self.jobs[jid]
        self.decision_rows.append((self.now, "dispatch", m.id, jid, len(queue), 0))
        job.op_rows[-1]["start"] = self.now
        self._to(job, PROCESSING)
        self._start(m, job)

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
        return {
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

    def metrics(self):
        """RewardMetrics.Fill (env/rewards/metrics.py names) for the current state, as float32 like the sensor.
        AGV travel time is the non-idle time (handshakes included) and there are no route waits or zone blocks:
        the twin has no reservations."""
        jobs = self.order                 # AllJobs: arrived jobs only
        n_state = [0] * 6
        t_state = [0.0] * 6
        flow = tis = 0.0
        exited = ops_total = ops_done = 0
        for j in jobs:
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
        }
        return {k: float(np.float32(v)) for k, v in values.items()}

    def job_rows(self):
        return [{"job_id": j.id, "arrival_time": j.arrival, "exit_time": j.exit_time,
                 "flow_time": (j.exit_time - j.arrival) if j.exit_time is not None else None,
                 "total_operations": j.total_ops, "completed_ops": j.completed_ops,
                 "time_needs_routing": j.t_state[NEEDS_ROUTING], "time_waiting_pickup": j.t_state[WAITING_PICKUP],
                 "time_in_transit": j.t_state[IN_TRANSIT], "time_queued": j.t_state[QUEUED],
                 "time_processing": j.t_state[PROCESSING]} for j in self.jobs.values()]

    def op_rows(self):
        return [r for j in self.jobs.values() for r in j.op_rows]


def run_twin(floor, jobs_data, cfg):
    """Runs one episode; returns the finished Twin (summary(), job_rows(), op_rows(), decision_rows)."""
    if not isinstance(floor, Floor):
        floor = Floor(floor)
    return Twin(floor, jobs_data, cfg).run()
