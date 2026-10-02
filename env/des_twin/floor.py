"""Zone graph exported by Unity (des_floor.json) and the routing queries the AGVs make on it.

Each method mirrors a TrafficZoneManager / AGVController / AGVPool query, including tie-breaking order, so
the twin picks the same routes, docks and vehicles as Unity whenever the floor state is the same.
"""
import json
import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Dock:
    key: int          # machine id, or a special key (incoming / outgoing belt of a tile)
    approach: tuple   # (x, z) the AGV drives to
    handshake: tuple  # (x, z) used to pick between a machine's docks
    facing: tuple     # (x, z) direction the AGV turns to before the handshake
    pickup: bool


@dataclass
class Zone:
    id: int
    name: str
    centre: tuple
    half: tuple
    capacity: int
    parking_lane: bool
    down: list
    up: list
    docks: dict       # key -> Dock

    def contains(self, x, z):
        """TrafficZone.Contains: inside the zone's box (edges included)."""
        return abs(x - self.centre[0]) <= self.half[0] and abs(z - self.centre[1]) <= self.half[1]


@dataclass
class Machine:
    id: int
    type: str
    tile: int
    zones: list
    pickup_zones: list
    pickup_pos: tuple
    dropoff_pos: tuple
    in_belts: list    # [(input_pos, capacity)] in PickIncomingBelt's preference order


class Floor:
    def __init__(self, data):
        if data.get("schema") != "des_floor/1":
            raise ValueError(f"unsupported floor schema {data.get('schema')!r}")
        self.data = data
        self.dt = float(data["fixed_dt"])
        self.agv = data["agv"]
        self.pre_dispatch_lead = float(data["pre_dispatch_lead"])
        self.routing_trigger = data["routing_trigger"]
        self.tile_count = int(data["tile_count"])
        if self.tile_count != 1:
            raise NotImplementedError("des_twin models single-tile floors only (tile_count must be 1)")
        self.zones = []
        for z in data["zones"]:
            docks = {}
            for d in z["docks"]:
                docks[d["key"]] = Dock(d["key"], tuple(d["approach"]), tuple(d["handshake"]),
                                       tuple(d["facing"]), bool(d["pickup"]))
            self.zones.append(Zone(z["id"], z["name"], tuple(z["centre"]),
                                   (z["size"][0] / 2.0, z["size"][1] / 2.0), z["capacity"],
                                   bool(z["parking_lane"]), list(z["down"]), list(z["up"]), docks))
        self.zone_by_id = {z.id: z for z in self.zones}
        self.machines = [Machine(m["id"], m["type"], m["tile"], list(m["zones"]), list(m["pickup_zones"]),
                                 tuple(m["pickup_pos"]), tuple(m["dropoff_pos"]),
                                 [(tuple(b["input_pos"]), int(b["capacity"])) for b in m["in_belts"]])
                         for m in data["machines"]]
        self.machine_by_id = {m.id: m for m in self.machines}
        t = data["tiles"][0]
        self.incoming_key, self.outgoing_key = t["incoming_key"], t["outgoing_key"]
        self.incoming_pos, self.outgoing_pos = tuple(t["incoming_pos"]), tuple(t["outgoing_pos"])
        self.agvs = data["agvs"]
        self._routes = {}
        self._hops = {}
        self._est = {}

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    # ── Routing (TrafficZoneManager.GetRoute / GetHopDistancesToNearest) ────────────────────────────

    def _lane(self, zid):
        z = self.zone_by_id.get(zid)
        return z is not None and z.parking_lane

    def route(self, a, b):
        """Fewest-hop zone route a -> b over the one-way links; the parking lane is only entered when the trip
        starts or ends in it. Empty list when unreachable."""
        key = (a, b)
        if key in self._routes:
            return self._routes[key]
        if a == b:
            res = [a]
        else:
            into_lane = self._lane(a) or self._lane(b)
            parent = {a: None}
            q = deque([a])
            res = []
            while q and not res:
                cur = q.popleft()
                for nxt in self.zone_by_id[cur].down:
                    if nxt in parent:
                        continue
                    if not into_lane and self._lane(nxt) and not self._lane(cur):
                        continue
                    parent[nxt] = cur
                    if nxt == b:
                        path = [b]
                        while path[-1] != a:
                            path.append(parent[path[-1]])
                        res = path[::-1]
                        break
                    q.append(nxt)
        self._routes[key] = res
        return res

    def hops_to(self, targets):
        """Zone -> hop count to the nearest of targets (reverse BFS over upstream links)."""
        key = tuple(targets)
        if key in self._hops:
            return self._hops[key]
        into_lane = any(self._lane(t) for t in targets)
        dist = {}
        q = deque()
        for t in targets:
            if t in self.zone_by_id and t not in dist:
                dist[t] = 0
                q.append(t)
        while q:
            cur = q.popleft()
            for pred in self.zone_by_id[cur].up:
                if pred in dist:
                    continue
                if not into_lane and self._lane(cur) and not self._lane(pred):
                    continue
                dist[pred] = dist[cur] + 1
                q.append(pred)
        self._hops[key] = dist
        return dist

    def zone_at(self, x, z):
        """TrafficZoneManager.GetZoneAtPosition: first zone (list order) whose box holds the point, or -1."""
        for zone in self.zones:
            if zone.contains(x, z):
                return zone.id
        return -1

    # ── Docks (AGVController.FindDockForMachine / FindSpecialDock) ──────────────────────────────────

    def machine_dock(self, machine_id, target_pos):
        """Among the machine's dock zones, the dock whose handshake point is nearest target_pos."""
        best = (-1, None)
        best_d = math.inf
        for zid in self.machine_by_id[machine_id].zones:
            dock = self.zone_by_id[zid].docks.get(machine_id)
            if dock is None:
                continue
            d = math.dist(dock.handshake, target_pos)
            if d < best_d:
                best_d, best = d, (zid, dock)
        return best

    def special_dock(self, key):
        for zone in self.zones:
            if key in zone.docks:
                return zone.id, zone.docks[key]
        return -1, None

    def incoming_zone(self):
        return self.special_dock(self.incoming_key)[0]

    # ── Travel estimate for TECT (FactoryOrchestrator.EstimateTravelSeconds) ─────────────────────────

    def estimate_path_length(self, from_machine, to_machine):
        """Zone-centre path length from the job's pickup dock zones to the target's dropoff dock zones."""
        key = (from_machine, to_machine)
        if key in self._est:
            return self._est[key]
        if from_machine >= 0:
            src = self.machine_by_id[from_machine].pickup_zones
        else:
            z = self.incoming_zone()
            src = [z] if z >= 0 else []
        tm = self.machine_by_id[to_machine]
        all_dst, pick_dst = tm.zones, tm.pickup_zones
        dst = [z for z in all_dst if not (pick_dst != all_dst and z in pick_dst)] or all_dst
        best = math.inf
        for s in src:
            for d in dst:
                r = self.route(s, d)
                if not r:
                    continue
                length = sum(math.dist(self.zone_by_id[r[i - 1]].centre, self.zone_by_id[r[i]].centre)
                             for i in range(1, len(r)))
                best = min(best, length)
        self._est[key] = best
        return best
