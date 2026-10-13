"""Observation schema v3 and action masks from the twin's state (port of ObservationBuilder.cs and
DispatchingEngine.HeadsThatMatter), so a policy can be trained on the twin and run on the Unity player.

Built at a decision, between the twin's yield and the agent's answer (engine.Twin.agent_decisions): the state then is
the one Unity observes, after the tick's harvests and earlier decisions and with the AGVs' poses after the
previous agent tick (Leg.state_after(k - 1)). Values are float32 as in C#; squashes are computed in double and
cast, as C# does.

Differences from the player, all by design:
- Job positions are logical (GetJobWorldPosition's fallback: the job's machine, its AGV, or the incoming belt).
  Unity prefers the job's visual, which sits on a belt slot or lerps per rendered frame, so the job channel of
  the grid and the focus distance (machine column 15) can differ by a cell or a few metres.
- No machine failures: operational is always 1, repair time 0, no deferred jobs, no machine down.
- In DES-0 ("instant") there are no vehicles; `instant_fleet` reports that many AGVs parked and idle in their
  bays, so the AGV features keep the values a policy sees on the physical floor with nothing to move.

The floor export must carry the observation frame (des_floor.json "obs", machine "pos3"/"cell", tile
"incoming_pos3"/"incoming_cell"), written by players built after 2026-10-01.
"""
import math

import numpy as np

from config import JOB_HEAD_RULES, MACHINE_HEAD_RULES

from .engine import EXITED, IN_TRANSIT, NEEDS_ROUTING, PROCESSING, QUEUED, WAITING_PICKUP

GRID = 64
CHANNELS = 3
MACHINE_FEATURES = 16
JOB_FEATURES = 21
GLOBAL_SCALARS = 18
EVENT_FLAGS = 6
JOB_HEAD = len(JOB_HEAD_RULES)          # sizes of the two action branches (env/config.py)
MACHINE_HEAD = len(MACHINE_HEAD_RULES)

TIME_SCALE, AGE_SCALE, HORIZON_SCALE = 300.0, 1200.0, 7200.0
COUNT_SCALE, OPS_SCALE, WIP_SCALE, MACHINE_COUNT_SCALE = 5.0, 3.0, 30.0, 100.0

TYPE_INDEX = {"Mill": 0, "Lathe": 1, "Weld": 2, "Inspect": 3, "Assemble": 4}
STATE_SLOT = {NEEDS_ROUTING: 0, WAITING_PICKUP: 1, IN_TRANSIT: 2, QUEUED: 3, PROCESSING: 4}
AGV_VALUE = {"idle": 0.25, "returning": 0.30, "to_pickup": 0.50, "to_prepickup": 0.50, "to_dropoff": 0.75}

f32 = np.float32


def squash(x, scale):
    """ObservationBuilder.Squash: x / (x + scale) in double, cast to float."""
    x = float(x)
    return f32(0.0) if x <= 0.0 else f32(x / (x + scale))


def signed_squash(x, scale):
    """ObservationBuilder.SignedSquash: x / (|x| + scale), sign kept."""
    x = float(x)
    return f32(-squash(-x, scale)) if x < 0.0 else squash(x, scale)


def clamp01(x):
    return f32(min(max(float(x), 0.0), 1.0))


def action_mask(decision):
    """SchedulingAgent.WriteDiscreteActionMask as the wrapper reports it: 1 = enabled, branches concatenated; a head
    that cannot change the outcome allows only its action 0."""
    job, machine = decision.heads_that_matter()
    mask = np.zeros(JOB_HEAD + MACHINE_HEAD, dtype=np.float32)
    mask[:JOB_HEAD if job else 1] = 1.0
    mask[JOB_HEAD:JOB_HEAD + (MACHINE_HEAD if machine else 1)] = 1.0
    return mask


class ObservationBuilder:
    def __init__(self, floor, max_machines, max_jobs, instant_fleet=0):
        frame = floor.obs_frame
        missing = [m.id for m in floor.machines if m.pos3 is None or m.cell is None]
        if frame is None or missing or floor.incoming_cell is None:
            raise ValueError("des_floor.json has no observation frame (obs / machine pos3, cell / incoming_cell): "
                             "re-export it with a player built after 2026-10-01 (-destrace)")
        self.floor = floor
        self.max_machines, self.max_jobs = int(max_machines), int(max_jobs)
        self.cx, self.cz = (f32(v) for v in frame["grid_centre"])
        self.hw, self.hd = (f32(v) for v in frame["grid_half"])
        self.diag = max(math.hypot(*frame["floor_size"]), 1.0)
        self.instant_fleet = int(instant_fleet)
        self.mpos = {m.id: m.pos3 for m in floor.machines}
        self.mcell = {m.id: tuple(m.cell) for m in floor.machines}
        self.mcaps = {m.id: [TYPE_INDEX[c] for c in (m.capabilities or (m.type,))] for m in floor.machines}

    # ── Grid frame (ObservationBuilder.WorldToGrid, float32) ──────────────────────────────────────────────

    def cell(self, x, z):
        nx = (f32(x) - self.cx + self.hw) / (f32(2.0) * self.hw)
        nz = (f32(z) - self.cz + self.hd) / (f32(2.0) * self.hd)
        gx = min(max(math.floor(f32(nx * f32(GRID))), 0), GRID - 1)
        gy = min(max(math.floor(f32(nz * f32(GRID))), 0), GRID - 1)
        return gx, gy

    # ── Positions ─────────────────────────────────────────────────────────────────────────────────────

    def agv_view(self, twin):
        """[(state, (x, z))] for every AGV, as transform.position after agent tick k - 1."""
        if twin.instant:
            return [("idle", tuple(a["park_pos"])) for a in self.floor.agvs[:self.instant_fleet]]
        out = []
        for a in twin.agvs:
            st = twin._status(a)
            if st != "idle" and a.leg is not None:
                x, z = a.leg.state_after(twin.k - 1, twin.kin)[:2]
            else:
                x, z = a.pos
            out.append((st, (x, z)))
        return out

    def job_cell(self, job, agv_xz):
        """Logical GetJobWorldPosition, as a grid cell."""
        if job.state in (PROCESSING, QUEUED, WAITING_PICKUP, NEEDS_ROUTING) and job.location >= 0:
            return self.mcell[job.location]
        if job.state == IN_TRANSIT and job.assigned_agv >= 0 and job.assigned_agv in agv_xz:
            return self.cell(*agv_xz[job.assigned_agv])
        return tuple(self.floor.incoming_cell)

    def focus_pos3(self, job):
        """Position of a routing focus job (NeedsRouting: its machine, or the incoming belt)."""
        return self.mpos[job.location] if job.location >= 0 else self.floor.incoming_pos3

    # ── Streams ───────────────────────────────────────────────────────────────────────────────────────

    def build(self, twin, decision):
        """Observation dict (env_wrappers.unity_env.slice_obs keys plus "action_mask") at this decision."""
        live = [j for j in twin.order if j.state != EXITED]
        agvs = self.agv_view(twin)
        agv_ids = [a.id for a in twin.agvs] if not twin.instant else list(range(len(agvs)))
        agv_xz = {i: xz for i, (_, xz) in zip(agv_ids, agvs)}
        loads = twin.all_machine_loads()
        queued = {}
        for j in live:
            if j.state == QUEUED and j.location >= 0:
                queued[j.location] = queued.get(j.location, 0) + 1
        return {
            "factory_grid": self._grid(twin, live, agvs, agv_xz),
            "machine_table": self._machines(twin, decision, loads, queued),
            "job_table": self._jobs(twin, decision, live),
            "global_scalars": self._scalars(twin, decision, live, agvs, loads),
            "event_flags": self._flags(twin, decision, agvs, queued),
            "action_mask": action_mask(decision),
        }

    def _grid(self, twin, live, agvs, agv_xz):
        g = np.zeros((CHANNELS, GRID, GRID), dtype=np.float32)
        for m in twin.machines:   # machines overwrite (SetGrid), the others keep the max
            gx, gy = self.mcell[m.id]
            g[0, gy, gx] = 0.25 if m.idle else 0.75
        for j in live:
            gx, gy = self.job_cell(j, agv_xz)
            progress = f32(1.0) - f32(f32(j.completed_ops) / f32(j.total_ops)) if j.total_ops > 0 else f32(0.5)
            g[1, gy, gx] = max(g[1, gy, gx], progress)
        for st, (x, z) in agvs:
            gx, gy = self.cell(x, z)
            g[2, gy, gx] = max(g[2, gy, gx], AGV_VALUE[st])
        return g

    def _machines(self, twin, dec, loads, queued):
        t = np.zeros((self.max_machines, MACHINE_FEATURES), dtype=np.float32)
        cands, focus, fpos = set(), None, None
        if dec.kind == "routing":
            cands.update(dec.machines)
            focus = twin.jobs.get(dec.job)
            if focus is not None:
                fpos = self.focus_pos3(focus)
        else:
            cands.add(dec.machine)
        for i, m in enumerate(twin.machines[:self.max_machines]):
            r = t[i]
            r[0] = 1.0
            for c in self.mcaps[m.id]:
                r[1 + c] = 1.0
            r[6] = 0.0 if m.idle else 1.0
            r[7] = 1.0
            r[9] = squash(twin.remaining_proc(m), TIME_SCALE)
            r[10] = squash(loads.get(m.id, f32(0)), TIME_SCALE)
            r[11] = squash(queued.get(m.id, 0), COUNT_SCALE)
            r[12] = clamp01(twin.utilization(m))
            r[13] = 1.0 if m.id in cands else 0.0
            if focus is not None and focus.cur_op < focus.total_ops and m.id in focus.ops[focus.cur_op]:
                r[14] = squash(focus.ops[focus.cur_op][m.id], TIME_SCALE)
                r[15] = clamp01(f32(math.dist(fpos, self.mpos[m.id])) / f32(self.diag))
        return t

    def _jobs(self, twin, dec, live):
        t = np.zeros((self.max_jobs, JOB_FEATURES), dtype=np.float32)
        focus_id, cands, disp = -1, set(), -1
        if dec.kind == "routing":
            focus_id = dec.job
            cands.update(dec.job_candidates)
            cands.add(dec.job)
        else:
            disp = dec.machine
            cands.update(dec.queue)
        focus = [j for j in live if j.id == focus_id]
        rows = focus + [j for j in live if j.id != focus_id and j.id in cands] + \
            [j for j in live if j.id != focus_id and j.id not in cands]
        now = twin.now
        for r, j in zip(t, rows[:self.max_jobs]):
            r[0] = 1.0
            r[1 + STATE_SLOT[j.state]] = 1.0
            r[7] = squash(now - float(f32(j.arrival)), AGE_SCALE)
            r[8] = squash(now - j.since, TIME_SCALE)
            r[9] = squash(j.total_ops - j.cur_op, OPS_SCALE)
            if j.cur_op < j.total_ops:
                el = j.ops[j.cur_op]
                if el:
                    r[10] = squash(min(el.values()), TIME_SCALE)
                    r[12] = squash(len(el), COUNT_SCALE)
                    r[13] = 1.0
                r[11] = squash(twin.remaining_work(j.id), AGE_SCALE)
            r[14] = 1.0 if j.id == focus_id else 0.0
            r[15] = 1.0 if j.id in cands else 0.0
            if disp >= 0 and j.state == QUEUED and j.location == disp:
                r[16] = squash(j.proc(disp), TIME_SCALE)
            if j.due is not None:   # C#: float DueDate - double SimTime - float remaining work, in double
                to_due = float(f32(j.due)) - now
                r[17] = 1.0
                r[18] = signed_squash(to_due - float(twin.remaining_work(j.id)), AGE_SCALE)
                r[19] = 1.0 if to_due < 0.0 else 0.0
                r[20] = signed_squash(to_due, AGE_SCALE)
        return t

    def _scalars(self, twin, dec, live, agvs, loads):
        s = np.zeros(GLOBAL_SCALARS, dtype=np.float32)
        by_state = [0] * 5
        for j in live:
            by_state[STATE_SLOT[j.state]] += 1
        active = len(live)
        s[0] = squash(twin.now, HORIZON_SCALE)
        s[1] = squash(active, WIP_SCALE)
        if active:
            for k in range(5):
                s[2 + k] = f32(by_state[k]) / f32(active)
        n = len(twin.machines)
        if n:
            load = f32(0)
            for m in twin.machines:
                load = f32(load + loads.get(m.id, f32(0)))
            s[7] = f32(sum(not m.idle for m in twin.machines)) / f32(n)
            s[10] = squash(f32(load / f32(n)), TIME_SCALE)
            s[12] = clamp01(f32(n) / f32(MACHINE_COUNT_SCALE))
        if agvs:
            s[9] = f32(sum(st != "idle" for st, _ in agvs)) / f32(len(agvs))
            if n:   # AGVs on duty (fleet schedule) per machine, as ObservationBuilder since 2026-10-04
                on_duty = len(agvs) if twin.instant else sum(twin._on_duty(a) for a in twin.agvs)
                s[13] = clamp01(f32(on_duty) / f32(n))
        if active > self.max_jobs:
            s[11] = f32(active - self.max_jobs) / f32(active)
        options = len(dec.machines) if dec.kind == "routing" else len(dec.queue)
        s[15] = squash(options, COUNT_SCALE)
        due = [j for j in live if j.due is not None]
        if due:
            now = twin.now
            late = sum(float(f32(j.due)) - now < 0.0 for j in due)
            behind = sum(float(f32(j.due)) - now - float(twin.remaining_work(j.id)) < 0.0 for j in due)
            s[16] = f32(late) / f32(len(due))
            s[17] = f32(behind) / f32(len(due))
        return s

    def _flags(self, twin, dec, agvs, queued):
        f = np.zeros(EVENT_FLAGS, dtype=np.float32)
        f[0] = 1.0 if dec.kind == "dispatch" else 0.0
        f[1] = 1.0 if dec.kind == "routing" else 0.0
        # AGVPool.GetIdleAGV: an idle AGV that is in service (on duty under the fleet schedule, 2026-10-04).
        on_duty = [True] * len(agvs) if twin.instant else [twin._on_duty(a) for a in twin.agvs]
        f[2] = 1.0 if any(st == "idle" and ok for (st, _), ok in zip(agvs, on_duty)) else 0.0
        f[3] = 1.0 if any(m.idle and queued.get(m.id, 0) > 0 for m in twin.machines) else 0.0
        return f
