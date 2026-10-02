"""
@file test_des_twin.py
@brief Tests for env/des_twin, the event-based twin of the Unity floor.

The fixtures are two Unity runs of convoy_waves on layout D with "-destrace -baselinedrain" (SPT_ECT, 1 and 3
AGVs): des_floor.json, des_jobs.json, agv_events.csv and the run's results. With one AGV nothing can block it,
so the kinematic twin must reproduce the Unity run itself; with three, every leg that met no other vehicle must
match to the tick and the rest must be late by exactly their zone wait.

@par Usage
@code{.sh}
cd env && python -m pytest tests/test_des_twin.py -v
@endcode
"""

import csv
import json
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from des_twin import Floor, TwinConfig, run_twin
from des_twin.legs import compare_legs
from des_twin.motion import (Kinematics, angdiff, countdown_ticks, f32_countdown, handshake_done, heading,
                             plan_leg, rotate_toward)
from des_twin.rules import f32, parse_rule, rank_jobs, select_machine

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "des_twin")
DT = 0.0199999921            # Unity's Time.fixedDeltaTime, as exported
AGV = {"speed": 3.5, "turn_speed": 120.0, "path_turn_threshold": 5.0, "alignment_threshold": 3.0,
       "waypoint_arrival_dist": 0.4, "dock_arrival_dist": 0.3, "handshake": 1.5}


# ── Synthetic floor: a one-way 3 x 3 m loop with a parking-lane spur ─────────────────────────────────

def _zone(zid, x, z, down, up, docks=(), lane=False):
    return {"id": zid, "name": f"Z{zid}", "centre": [x, z], "size": [3.0, 3.0], "capacity": 1,
            "parking_lane": lane, "down": down, "up": up, "docks": list(docks)}


def _dock(key, x, z, fx, fz, pickup):
    return {"key": key, "approach": [x, z], "handshake": [x + 1.5 * fx, z + 1.5 * fz],
            "facing": [fx, fz], "pickup": pickup}


def synthetic_floor(agvs=1):
    # loop 0 -> 1 -> ... -> 7 -> 0 around a 9 x 9 square; 8 is a parking bay off zone 7 (in from 7, out to 0)
    zones = [
        _zone(0, 0, 0, [1], [7, 8]),
        _zone(1, 3, 0, [2], [0], [_dock(-1, 3, 0, 0, -1, True)]),                     # incoming belt
        _zone(2, 6, 0, [3], [1], [_dock(0, 6, 0, 0, -1, True)]),                      # machine 0
        _zone(3, 6, 3, [4], [2]),
        _zone(4, 6, 6, [5], [3]),
        _zone(5, 3, 6, [6], [4], [_dock(1, 3, 6, 0, 1, True)]),                       # machine 1
        _zone(6, 0, 6, [7], [5], [_dock(-2, 0, 6, 0, 1, False)]),                     # outgoing belt
        _zone(7, 0, 3, [0, 8], [6]),
        _zone(8, -3, 3, [0], [7], lane=True),
    ]
    machines = [
        {"id": 0, "type": "Mill", "tile": 0, "zones": [2], "pickup_zones": [2], "pickup_pos": [6, -1.5],
         "dropoff_pos": [6, -1.5], "in_belts": [{"input_pos": [6, -1.5], "capacity": 3}]},
        {"id": 1, "type": "Mill", "tile": 0, "zones": [5], "pickup_zones": [5], "pickup_pos": [3, 7.5],
         "dropoff_pos": [3, 7.5], "in_belts": [{"input_pos": [3, 7.5], "capacity": 3}]},
    ]
    return Floor({
        "schema": "des_floor/1", "layout": "T", "agv_count": agvs, "tile_count": 1, "agvs_pooled": False,
        "parking_method": "lane", "routing_trigger": "onTransport", "reservation_protocol": "releasePrevious",
        "fixed_dt": DT, "pre_dispatch_lead": 15, "parking_bayed": True, "agv": AGV, "zones": zones,
        "machines": machines,
        "tiles": [{"tile": 0, "incoming_key": -1, "outgoing_key": -2, "incoming_zone": 1, "outgoing_zone": 6,
                   "incoming_pos": [3, -1.5], "outgoing_pos": [0, 7.5]}],
        "agvs": [{"id": i, "tile": 0, "park_pos": [-3, 3], "park_zone": 8, "pos": [-3, 3], "yaw": 0.0, "zone": 8}
                 for i in range(agvs)],
    })


def reference_leg(kin, floor, pos, yaw, zone, route, final, kind, max_ticks=100000):
    """Literal per-tick port of AGVController (UpdateMovement + the state's arrival check), no reservations.
    Returns the 0-based tick of arrival and the pose there. In "geometric" mode the AGV never turns in place."""
    x, z = pos
    ri = 0
    while ri < len(route) and route[ri] == zone:
        ri += 1
    wp = floor.zone_by_id[route[ri]].centre if ri < len(route) else final
    reach = kin.dock_dist if kind == "dock" else kin.wp_dist
    for t in range(max_ticks):
        past = ri >= len(route)
        dist = math.hypot(wp[0] - x, wp[1] - z)
        if dist <= (kin.dock_dist if past else kin.wp_dist):
            if not past:
                zone = route[ri]
                ri += 1
                while ri < len(route) and route[ri] == zone:
                    ri += 1
                wp = floor.zone_by_id[route[ri]].centre if ri < len(route) else final
        else:
            dx, dz = wp[0] - x, wp[1] - z
            goal = heading(dx, dz)
            if kin.turns and abs(angdiff(yaw, goal)) > kin.turn_thr:
                yaw = kin.rotate(yaw, goal)
            else:
                yaw = goal
                s = min(kin.step, dist)
                x, z = x + dx / dist * s, z + dz / dist * s
        # ReachedDock / ReachedParking: distance only, checked every tick after the movement
        if math.hypot(final[0] - x, final[1] - z) <= reach and (kind == "park" or ri >= len(route)):
            return t, (x, z), yaw
    raise AssertionError("reference leg never arrived")


# ── Motion ──────────────────────────────────────────────────────────────────────────────────────────

def test_handshake_countdown_matches_unity_float32():
    """@brief A 1.5 s float32 timer at Unity's fixedDeltaTime needs 76 ticks (75 plus float residue), as logged."""
    assert countdown_ticks(1.5, DT) == 76


def test_f32_countdown_matches_sequential_float32():
    """@brief The jumping float32 timer equals a tick-by-tick float32 countdown (PhysicalMachine.remainingTime); a
    1500 s operation ends 43 ticks before 1500 / dt because each float32 subtraction removes slightly more."""
    import numpy as np

    def brute(x, until=None, steps=None):
        x, c, n = np.float32(x), np.float32(DT), 0
        while not ((until is not None and x <= until) or (steps is not None and n >= steps)):
            x, n = np.float32(x - c), n + 1
        return n, x

    rng = random.Random(5)
    for _ in range(12):
        d = np.float32(round(rng.uniform(0.01, 600), 2))
        assert f32_countdown(d, DT, until=0.0) == brute(d, until=0.0)
        assert f32_countdown(d, DT, until=15) == brute(d, until=15)
        k = rng.randrange(0, int(d / DT) + 3)
        assert f32_countdown(d, DT, steps=k) == brute(d, steps=k)
    assert f32_countdown(1500.0, DT, until=0.0)[0] == 74958


def test_rotate_toward_exact_and_unity_slerp():
    """@brief Above ~36 degrees a step is the full maxDegrees; below, Unity's lerp fallback turns slightly less."""
    assert rotate_toward(0.0, 90.0, 2.4, unity_slerp=True) == pytest.approx(2.4)
    assert rotate_toward(0.0, 90.0, 2.4, unity_slerp=False) == pytest.approx(2.4)
    near = rotate_toward(0.0, 20.0, 2.4, unity_slerp=True)
    assert 2.39 < near < 2.4
    assert rotate_toward(0.0, 20.0, 2.4, unity_slerp=False) == pytest.approx(2.4)
    assert rotate_toward(0.0, 2.0, 2.4) == pytest.approx(2.0)          # within one step: snaps to the goal
    assert rotate_toward(350.0, 10.0, 2.4, unity_slerp=False) == pytest.approx(352.4)   # shortest way round
    assert rotate_toward(10.0, 350.0, 2.4, unity_slerp=False) == pytest.approx(7.6)
    assert 352.39 < rotate_toward(350.0, 10.0, 2.4) < 352.4


@pytest.mark.parametrize("mode", ["kinematic", "geometric"])
def test_plan_leg_matches_per_tick_reference(mode):
    """@brief The analytic leg planner agrees tick-for-tick with a literal per-tick replay, from random poses."""
    floor = synthetic_floor()
    kin = Kinematics(AGV, DT, mode)
    rng = random.Random(3)
    cases = 0
    for _ in range(200):
        zone = rng.choice([0, 1, 2, 3, 4, 5, 6, 7])
        cx, cz = floor.zone_by_id[zone].centre
        pos = (cx + rng.uniform(-1.2, 1.2), cz + rng.uniform(-1.2, 1.2))
        yaw = rng.uniform(0, 360)
        if rng.random() < 0.5:
            target, kind = (8, (-3.0, 3.0)), "park"
        else:
            zid, dock = floor.machine_dock(rng.choice([0, 1]), (0, 0))
            target, kind = (zid, dock.approach), "dock"
        route = floor.route(zone, target[0])
        leg = plan_leg(kin, floor, 0, pos, yaw, zone, route, target[1], kind)
        t, end, end_yaw = reference_leg(kin, floor, pos, yaw, zone, route, target[1], kind)
        assert leg.arrive_tick == t
        assert leg.end_pos == pytest.approx(end, abs=1e-9)
        assert angdiff(leg.end_yaw, end_yaw) == pytest.approx(0.0, abs=1e-6)
        x, z, y, _, _ = leg.state_after(leg.arrive_tick, kin)
        assert (x, z) == pytest.approx(leg.end_pos, abs=1e-9)
        cases += 1
    assert cases == 200


def test_geometric_leg_is_length_over_speed():
    """@brief Without turning, a leg takes its path length / speed (to the tick)."""
    floor = synthetic_floor()
    kin = Kinematics(AGV, DT, "geometric")
    route = floor.route(0, 4)                       # 0 -> 1 -> 2 -> 3 -> 4: 12 m with one corner
    leg = plan_leg(kin, floor, 0, (0.0, 0.0), 0.0, 0, route, (6.0, 6.0), "dock")
    assert route == [0, 1, 2, 3, 4]
    assert leg.arrive_tick * kin.step == pytest.approx(leg.path_length, abs=5 * kin.step)


def test_handshake_waits_for_alignment():
    """@brief The handshake timer only runs once the AGV faces the dock; a 180-degree turn adds ~1.5 s."""
    kin = Kinematics(AGV, DT)
    aligned, _ = handshake_done(kin, 0.0, 0.0, 0, True)
    assert aligned == kin.handshake_ticks - 1
    turned, yaw = handshake_done(kin, 0.0, 180.0, 0, True)
    assert abs(angdiff(yaw, 180.0)) <= kin.align_thr
    assert turned - aligned == pytest.approx((180 - 3) / 2.4 - 1, abs=2)


# ── Floor queries ───────────────────────────────────────────────────────────────────────────────────

def test_route_keeps_out_of_parking_lane_unless_an_endpoint():
    """@brief The parking lane is no shortcut: 7 -> 0 goes direct, but trips to or from the bay may use it."""
    floor = synthetic_floor()
    assert floor.route(7, 0) == [7, 0]
    assert floor.route(6, 8) == [6, 7, 8]
    assert floor.route(8, 2) == [8, 0, 1, 2]
    hops = floor.hops_to([2])
    assert hops[2] == 0 and hops[1] == 1 and hops[0] == 2 and hops[8] == 3
    assert floor.estimate_path_length(-1, 0) == pytest.approx(3.0)      # incoming belt (zone 1) -> machine 0


def test_zone_at_and_docks():
    floor = synthetic_floor()
    assert floor.zone_at(6.2, 0.4) == 2
    assert floor.zone_at(50, 50) == -1
    assert floor.machine_dock(1, (3, 7.5))[0] == 5
    assert floor.special_dock(-2)[0] == 6


# ── Rules ───────────────────────────────────────────────────────────────────────────────────────────

def test_parse_rule():
    assert parse_rule("spt_ect") == ("SPT", "ECT")
    with pytest.raises(ValueError):
        parse_rule("Random")
    with pytest.raises(ValueError):
        parse_rule("SPT_XYZ")


def test_rank_and_select_tie_to_first_candidate():
    """@brief Equal scores go to the first candidate, as the C# ArgMin / ArgMinIdx."""
    times = {3: f32(5), 1: f32(5), 2: f32(7)}
    assert rank_jobs("SPT", [3, 1, 2], None, lambda j: times[j]) == 3
    assert rank_jobs("LPT", [3, 1, 2], None, lambda j: times[j]) == 2
    c, t, q = [4, 9], [f32(10), f32(10)], [f32(20), f32(20)]
    assert select_machine("ECT", c, t, q, [f32(0)] * 2, [f32(0)] * 2) == 4
    # TECT waits for the slower of travel and queue: machine 9 is nearer, so it wins despite equal queues
    assert select_machine("TECT", c, t, [f32(0), f32(0)], [f32(0)] * 2, [f32(30), f32(5)]) == 9


# ── Engine ──────────────────────────────────────────────────────────────────────────────────────────

def _jobs(*specs):
    return {"schema": "des_jobs/1", "jobs": [
        {"id": i, "arrival": a, "ops": [{"type": "Mill", "eligible": [[m, d] for m, d in op]} for op in ops]}
        for i, (a, ops) in enumerate(specs)]}


def test_instant_transport_is_a_plain_job_shop():
    """@brief DES-0: no vehicles, SPT runs the 5 s job first on the shared machine."""
    jobs = _jobs((0.0, [[(0, 10.0)]]), (0.0, [[(0, 5.0)]]))
    tw = run_twin(synthetic_floor(), jobs, TwinConfig("SPT_ECT", "instant"))
    rows = {r["job_id"]: r for r in tw.job_rows()}
    assert rows[1]["flow_time"] == pytest.approx(5.0, abs=0.1)
    assert rows[0]["flow_time"] == pytest.approx(15.0, abs=0.1)
    assert tw.summary()["agv_count"] == 0


@pytest.mark.parametrize("transport", ["geometric", "kinematic"])
def test_vehicle_twin_completes_and_parks(transport):
    """@brief Two-op jobs cross the floor and leave; the AGV ends parked; flow exceeds processing."""
    jobs = _jobs((0.0, [[(0, 20.0)], [(1, 10.0)]]), (40.0, [[(1, 15.0)]]))
    tw = run_twin(synthetic_floor(), jobs, TwinConfig("SRT_TECT", transport))
    s = tw.summary()
    assert s["jobs_completed"] == 2 and not s["timed_out"]
    rows = {r["job_id"]: r for r in tw.job_rows()}
    assert rows[0]["flow_time"] > 30.0 and rows[1]["flow_time"] > 15.0
    assert s["mean_transport_wait"] > 0
    assert tw._status(tw.agvs[0]) in ("idle", "returning")


def test_kinematic_slower_than_geometric_slower_than_instant():
    jobs = _jobs(*[(5.0 * i, [[(0, 12.0), (1, 14.0)], [(1, 9.0)]]) for i in range(6)])
    flows = {tr: run_twin(synthetic_floor(), jobs, TwinConfig("SPT_ECT", tr)).summary()["mean_flow_time"]
             for tr in ("instant", "geometric", "kinematic")}
    assert flows["instant"] < flows["geometric"] < flows["kinematic"]


# ── Docking against Unity (fixtures) ────────────────────────────────────────────────────────────────

def _fixture(name):
    d = os.path.join(FIX, name)
    floor = Floor.load(os.path.join(d, "des_floor.json"))
    with open(os.path.join(d, "des_jobs.json")) as f:
        jobs = json.load(f)
    with open(os.path.join(d, "results.csv")) as f:
        unity = next(csv.DictReader(f))
    return d, floor, jobs, unity


def test_single_agv_twin_reproduces_unity_run():
    """@brief With one AGV there is nothing to block it: the kinematic twin reproduces the Unity episode, every AGV
    milestone at the same tick and every job's exit time to the CSV's 0.1 s rounding."""
    d, floor, jobs, unity = _fixture("convoy_agv1")
    tw = run_twin(floor, jobs, TwinConfig(unity["rule"], "kinematic", agv_count=int(unity["agvCount"])))
    s = tw.summary()
    assert s["makespan"] == pytest.approx(float(unity["makespan"]), abs=0.011)
    assert s["mean_flow_time"] == pytest.approx(float(unity["mean_flow_time"]), abs=0.011)
    with open(os.path.join(d, "agv_events.csv")) as f:
        events = [(float(r["sim_time"]), r["event"].split("_")[0] if "dispatch" in r["event"] else r["event"],
                   int(r["job_id"])) for r in csv.DictReader(f) if r["event"] not in ("return", "park")]
    mine = sorted((t, e, j) for t, _, e, j in tw.trace if e != "park(planned)")
    assert len(events) == len(mine)
    for u, m in zip(events, mine):
        assert u[1:] == m[1:] and u[0] == pytest.approx(m[0], abs=0.011)
    with open(os.path.join(d, "job_completions.csv")) as f:
        exits = {int(r["job_id"]): float(r["exit_time"]) for r in csv.DictReader(f)}
    for r in tw.job_rows():
        assert r["exit_time"] == pytest.approx(exits[r["job_id"]], abs=0.051)


@pytest.mark.parametrize("name", ["convoy_agv1", "convoy_agv3"])
def test_uncontended_legs_match_unity_to_the_tick(name):
    """@brief Every Unity leg that waited for no zone takes exactly its free-flow time; a leg that did wait is late
    by its zone wait (to a tick per wait episode), so the gap to the twin is the reservation delay."""
    rows = compare_legs(os.path.join(FIX, name))
    free = [r for r in rows if r["zone_wait_s"] == 0]
    assert len(free) > 100
    assert all(abs(r["delay_s"]) < 0.011 for r in free)
    waited = [r for r in rows if r["zone_wait_s"] > 0]
    for r in waited:
        assert r["delay_s"] == pytest.approx(r["zone_wait_s"], abs=0.1)
    if name == "convoy_agv3":
        assert waited, "three AGVs on convoy_waves queue at the input belt"
