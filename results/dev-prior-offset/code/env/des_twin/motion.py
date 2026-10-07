"""Free-flow AGV motion: AGVController's movement with no other vehicles on the floor.

A leg is planned once, when it starts, as tick-exact phases (turn in place, drive straight), so its arrival tick
is known in advance and the AGV's pose at any earlier tick can be recovered if it is redirected mid-leg (a
returning AGV taking a new job). Times are agent ticks: AGVs move after the orchestrator within a FixedUpdate,
so a leg commanded at orchestrator tick k moves from agent tick k, and a milestone reached at agent tick m is
seen by the orchestrator at tick m + 1.

Per tick, AGVController.UpdateMovement either detects arrival at a zone centre (within waypoint_arrival_dist; no
movement that tick), turns in place when the heading is off by more than path_turn_threshold, or snaps the
heading and drives speed * dt. At the final dock the AGV turns to the dock's facing direction and counts the
handshake down only while aligned. "geometric" drops the turning: path length / speed plus handshakes.

Turns follow Quaternion.RotateTowards, which asks Slerp for maxDegrees / remaining of the remaining angle.
Unity's Slerp falls back to a normalised lerp once the two rotations are within acos(0.95) of each other
(about 36 degrees of yaw), and a lerp step turns less than its share of the angle, so the last ~36 degrees of
every turn run slightly slow (about 0.15 degrees in total). Ignoring this ("exact") lands about half of the
uncontended Unity legs 1-3 ticks early; emulating it reproduces them to the tick (env/tests/test_des_twin.py).
"""
import math
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


def heading(dx, dz):
    """Unity yaw in degrees of a flat direction (yaw 0 faces +z, 90 faces +x)."""
    return math.degrees(math.atan2(dx, dz)) % 360.0


def angdiff(a, b):
    """Signed shortest rotation from yaw a to yaw b, in (-180, 180]."""
    d = (b - a + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


SLERP_LERP_DOT = 0.95                        # Unity Slerp: below this quaternion dot it is a true slerp
QUAT_EQUAL_DEG = 2.0 * math.degrees(math.acos(1.0 - 1e-6))   # Quaternion.Angle reports 0 below this


def rotate_toward(yaw, goal, max_deg, unity_slerp=True):
    """One Quaternion.RotateTowards step of yaw toward goal (yaw-only rotations)."""
    d = angdiff(yaw, goal)
    th = abs(d)
    if th <= max_deg or th < QUAT_EQUAL_DEG:
        return goal % 360.0
    step = max_deg
    if unity_slerp:
        half = math.radians(th / 2.0)
        if math.cos(half) >= SLERP_LERP_DOT:
            t = max_deg / th
            step = 2.0 * math.degrees(math.atan2(t * math.sin(half), (1.0 - t) + t * math.cos(half)))
    return (yaw + math.copysign(step, d)) % 360.0


def _binade(x):
    return math.frexp(x)[1]


def f32_countdown(start, step, until=None, steps=None):
    """Unity's float32 timer `t -= step` per tick, from t = start. Returns (n, t): the first n with t <= until, or
    the value after `steps` ticks. Exact, but jumps: inside one binary exponent range a float32 subtraction of a
    fixed step removes the same amount every tick once the rounding has settled, so after two equal steps within
    a binade the walk advances many ticks at once (a 2000 s timer is ~100k ticks)."""
    x, c = np.float32(start), np.float32(step)
    n, prev_s, prev_b = 0, None, None
    while True:
        if until is not None and x <= until:
            return n, x
        if steps is not None and n >= steps:
            return n, x
        nx = np.float32(x - c)
        s, b = float(x) - float(nx), (_binade(float(x)), _binade(float(nx)))
        x, n = nx, n + 1
        if s == prev_s and b[0] == b[1] == prev_b[1] and float(x) > 0:
            lower = math.ldexp(0.5, b[1])                  # smallest value of this binade
            k = math.floor((float(x) - float(c) - lower) / s) - 1
            if until is not None:
                k = min(k, math.ceil((float(x) - until) / s) - 1)
            if steps is not None:
                k = min(k, steps - n)
            if k > 0:
                x, n = np.float32(float(x) - k * s), n + k
        prev_s, prev_b = s, b


def countdown_ticks(duration, dt):
    """Ticks until a float32 timer started at duration, decremented by dt each tick, reaches <= 0."""
    return f32_countdown(duration, dt, until=0.0)[0]


class Kinematics:
    def __init__(self, agv, dt, mode="kinematic", slerp="unity"):
        if mode not in ("kinematic", "geometric"):
            raise ValueError(mode)
        if slerp not in ("unity", "exact"):
            raise ValueError(slerp)
        self.mode = mode
        self.unity_slerp = slerp == "unity"
        self.dt = dt
        self.step = agv["speed"] * dt
        self.turns = mode == "kinematic"
        self.rot = agv["turn_speed"] * dt if self.turns else math.inf
        self.turn_thr = agv["path_turn_threshold"]
        self.align_thr = agv["alignment_threshold"]
        self.wp_dist = agv["waypoint_arrival_dist"]
        self.dock_dist = agv["dock_arrival_dist"]
        self.handshake_ticks = countdown_ticks(agv["handshake"], dt)

    def rotate(self, yaw, goal):
        """One tick of turning toward goal."""
        return rotate_toward(yaw, goal, self.rot, self.unity_slerp)

    def turn(self, yaw, goal):
        """Turning in place before driving: (ticks, yaw) once the heading is within path_turn_threshold."""
        if not self.turns:
            return 0, yaw
        n = 0
        while abs(angdiff(yaw, goal)) > self.turn_thr:
            yaw = self.rotate(yaw, goal)
            n += 1
        return n, yaw


@dataclass
class Leg:
    """One planned drive. Phases: ("turn", t0, t1, x, z, yaw0, goal) or ("drive", t0, t1, x0, z0, ux, uz, yaw, length);
    entries: (tick, zone, route_index_after) for each zone centre reached."""
    start_tick: int
    route: list
    start_pos: tuple
    start_yaw: float
    start_zone: int
    start_ri: int
    phases: list = field(default_factory=list)
    entries: list = field(default_factory=list)
    arrive_tick: int = 0
    end_pos: tuple = (0.0, 0.0)
    end_yaw: float = 0.0
    end_zone: int = -1
    path_length: float = 0.0

    def state_after(self, tick, kin):
        """(x, z, yaw, zone, route_index) at the end of agent tick `tick` (tick < start_tick: the start pose)."""
        x, z = self.start_pos
        yaw, zone, ri = self.start_yaw, self.start_zone, self.start_ri
        for t, zid, ri_after in self.entries:
            if t <= tick:
                zone, ri = zid, ri_after
            else:
                break
        for ph in self.phases:
            if ph[1] > tick:
                break
            n = min(tick, ph[2]) - ph[1] + 1
            if ph[0] == "turn":
                _, _, _, x, z, yaw, goal = ph
                for _ in range(n):
                    yaw = kin.rotate(yaw, goal)
            else:
                _, _, _, x0, z0, ux, uz, yaw, length = ph
                s = min(n * kin.step, length)
                x, z = x0 + ux * s, z0 + uz * s
        return x, z, yaw, zone, ri


def _drive_to(kin, phases, t, x, z, yaw, tx, tz, stop):
    """Turn then drive toward (tx, tz) until within `stop`. Returns (ticks, x, z, yaw, driven)."""
    d = math.hypot(tx - x, tz - z)
    if d <= stop:
        return 0, x, z, yaw, 0.0
    goal = heading(tx - x, tz - z)
    r, _ = kin.turn(yaw, goal)
    if r:
        phases.append(("turn", t, t + r - 1, x, z, yaw, goal))
    n = max(1, math.ceil((d - stop) / kin.step - EPS))
    length = min(n * kin.step, d)
    ux, uz = (tx - x) / d, (tz - z) / d
    phases.append(("drive", t + r, t + r + n - 1, x, z, ux, uz, goal, length))
    return r + n, x + ux * length, z + uz * length, goal, length


def _first_reach(ph, step, final, reach):
    """First move tick (1-based) of a drive phase after which the AGV is within reach of final, or None."""
    _, t0, t1, x0, z0, ux, uz, _, length = ph
    qx, qz = x0 - final[0], z0 - final[1]
    b = qx * ux + qz * uz
    disc = b * b - (qx * qx + qz * qz - reach * reach)
    if disc < 0:
        return None
    i = max(1, math.ceil((-b - math.sqrt(disc)) / step - EPS))
    if i > t1 - t0 + 1:
        return None
    s = min(i * step, length)
    return i if math.hypot(qx + ux * s, qz + uz * s) <= reach + EPS else None


def plan_leg(kin, floor, start_tick, pos, yaw, zone, route, final_pos, final_kind):
    """Plans a drive along `route` (zone ids; entries equal to the current zone are skipped, as in
    BeginNextWaypoint) and then to final_pos: a dock approach ("dock", arrival within dock_arrival_dist) or a
    parking bay ("park", ReachedParking within waypoint_arrival_dist). Returns a Leg whose arrive_tick is the
    agent tick the AGV reaches the dock / bay.

    ReachedParking is a distance test made after every move, so an AGV can park on the move that brings it within
    waypoint_arrival_dist of its bay, a tick before it would register entering the bay zone. ReachedDock has the
    same form but cannot fire early: dock approaches sit on zone centres, and a zone is entered while still more
    than dock_arrival_dist from its centre."""
    leg = Leg(start_tick, route, pos, yaw, zone, 0)
    x, z = pos
    ri = 0
    while ri < len(route) and route[ri] == zone:
        ri += 1
    leg.start_ri = ri
    t = start_tick
    reach = kin.dock_dist if final_kind == "dock" else kin.wp_dist
    while True:
        if ri < len(route):
            cx, cz = floor.zone_by_id[route[ri]].centre
            n, x, z, yaw, s = _drive_to(kin, leg.phases, t, x, z, yaw, cx, cz, kin.wp_dist)
            if final_kind == "park" and n and leg.phases[-1][0] == "drive":
                ph = leg.phases[-1]
                i = _first_reach(ph, kin.step, final_pos, reach)
                if i is not None:              # parks mid-drive, before reaching this zone
                    s = min(i * kin.step, ph[8])
                    leg.phases[-1] = ph[:2] + (ph[1] + i - 1,) + ph[3:8] + (s,)
                    leg.path_length += s
                    x, z = ph[3] + ph[5] * s, ph[4] + ph[6] * s
                    t = ph[1] + i - 1
                    break
            leg.path_length += s
            t += n                            # the tick that detects arrival (no movement in it)
            zone = route[ri]
            ri += 1
            while ri < len(route) and route[ri] == zone:
                ri += 1
            leg.entries.append((t, zone, ri))
            if ri >= len(route) and math.hypot(final_pos[0] - x, final_pos[1] - z) <= reach:
                break                         # dock / bay reached in the same tick
            t += 1
        else:
            d = math.hypot(final_pos[0] - x, final_pos[1] - z)
            if d > reach:
                n, x, z, yaw, s = _drive_to(kin, leg.phases, t, x, z, yaw, final_pos[0], final_pos[1], reach)
                leg.path_length += s
                t += n - 1                    # arrival is checked after the move, in the same tick
            break
    leg.arrive_tick = t
    leg.end_pos, leg.end_yaw, leg.end_zone = (x, z), yaw, zone
    return leg


def handshake_done(kin, yaw, goal, first_tick, align_first):
    """Tick the handshake completes at a dock reached (align_first) or re-armed at first_tick. The AGV turns
    toward the dock's facing direction and the timer only runs on ticks it is within alignment_threshold."""
    t = first_tick
    if align_first:
        yaw = kin.rotate(yaw, goal)
    count = 0
    while True:
        if abs(angdiff(yaw, goal)) <= kin.align_thr:
            count += 1
            if count >= kin.handshake_ticks:
                return t, yaw
        else:
            yaw = kin.rotate(yaw, goal)
        t += 1
