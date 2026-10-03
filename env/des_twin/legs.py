"""Leg-level comparison of Unity's AGV motion with the twin's free-flow model (agv_events.csv, "-destrace").

Every Unity leg (to a pickup, loaded to a dropoff, back to the bay) is replanned by motion.plan_leg from the
pose Unity logged when the leg started, with no other vehicles. For an uncontended leg (no zone wait) the two
should agree to the tick; for the rest, Unity's extra time over the free-flow plan is what the shared floor
(zone reservations, queues behind other AGVs) added. This is the per-trip counterpart of the episode-level gap.
"""
import csv
import math

from .floor import Floor
from .motion import Kinematics, plan_leg

START = {"dispatch_idle": "pickup", "dispatch_return": "pickup", "predispatch_idle": "prepickup",
         "predispatch_return": "prepickup", "pickup": "dropoff", "return": "park"}
END = {"pickup": "arrive_pickup", "prepickup": "arrive_prepickup", "dropoff": "arrive_dropoff", "park": "park"}
# Events that end a leg early (it is replanned or abandoned): the leg is dropped.
BREAK = {"dispatch_idle", "dispatch_return", "predispatch_idle", "predispatch_return", "cancel", "abort",
         "redirect", "orphan", "stall", "snap", "breakdown", "reset"}


def read_events(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def unity_legs(events, dt):
    """Yields dicts: agv, kind, start_tick (first moving tick), arrive_tick, start pose/zone, machine, end pos,
    wait (zone-blocked seconds during the leg), driven (path length during the leg)."""
    open_ = {}
    for r in events:
        a, ev = int(r["agv_id"]), r["event"]
        tick = round(float(r["sim_time"]) / dt)
        leg = open_.get(a)
        if leg is not None and ev == END[leg["kind"]]:
            leg.update(arrive_tick=tick, end=(float(r["x"]), float(r["z"])),
                       wait=float(r["cum_wait_route"]) - leg["wait0"],
                       driven=float(r["cum_path_length"]) - leg["dist0"])
            del open_[a]
            yield leg
            leg = None
        elif leg is not None and ev in BREAK:
            del open_[a]
        if ev in START:
            kind = START[ev]
            # Orchestrator decisions (dispatch) are logged before the AGV's own update in that tick, so the AGV
            # moves from the same tick; milestones the AGV logs itself (pickup, return) move from the next one.
            first = tick if ev.startswith(("dispatch", "predispatch")) else tick + 1
            machine = int(r["source_machine"]) if kind in ("pickup", "prepickup") else int(r["target_machine"])
            open_[a] = dict(agv=a, kind=kind, start_tick=first, pos=(float(r["x"]), float(r["z"])),
                            yaw=float(r["yaw"]), zone=int(r["zone_id"]), machine=machine,
                            wait0=float(r["cum_wait_route"]), dist0=float(r["cum_path_length"]))


def _target(floor, leg, park_by_agv):
    """(zone, final_pos, final_kind) of a Unity leg. A machine dock is the one nearest where Unity stopped."""
    if leg["kind"] == "park":
        zone, pos = park_by_agv[leg["agv"]]
        return zone, pos, "park"
    if leg["machine"] < 0:
        key = floor.incoming_key if leg["kind"] in ("pickup", "prepickup") else floor.outgoing_key
        zid, dock = floor.special_dock(key)
        return zid, dock.approach, "dock"
    best = None
    for zid in floor.machine_by_id[leg["machine"]].zones:
        dock = floor.zone_by_id[zid].docks.get(leg["machine"])
        if dock is None:
            continue
        d = math.dist(dock.approach, leg["end"])
        if best is None or d < best[0]:
            best = (d, zid, dock.approach)
    return best[1], best[2], "dock"


def compare_legs(run_dir, transport="kinematic"):
    """Rows: one per completed Unity leg with Unity and free-flow durations (seconds) and the zone wait."""
    floor = Floor.load(f"{run_dir}/des_floor.json")
    kin = Kinematics(floor.agv, floor.dt, transport)
    park = {a["id"]: (a["park_zone"], tuple(a["park_pos"])) for a in floor.agvs}
    rows = []
    for leg in unity_legs(read_events(f"{run_dir}/agv_events.csv"), floor.dt):
        zone_to, final, kind = _target(floor, leg, park)
        route = floor.route(leg["zone"], zone_to)
        if not route:
            continue
        plan = plan_leg(kin, floor, leg["start_tick"], leg["pos"], leg["yaw"], leg["zone"], route, final, kind)
        unity_s = (leg["arrive_tick"] - leg["start_tick"] + 1) * floor.dt
        free_s = (plan.arrive_tick - leg["start_tick"] + 1) * floor.dt
        rows.append(dict(agv=leg["agv"], kind=leg["kind"], start_time=round(leg["start_tick"] * floor.dt, 2),
                         unity_s=round(unity_s, 2), freeflow_s=round(free_s, 2),
                         delay_s=round(unity_s - free_s, 2), zone_wait_s=round(leg["wait"], 2),
                         unity_len=round(leg["driven"], 3), freeflow_len=round(plan.path_length, 3),
                         hops=len(route) - 1))
    return rows
