"""Unity side of the rules API (rules_api, 2026-10-11): the decision view parsed from the JSON a player launched with
-pyrules sends at each agent decision (C# FactoryOrchestrator.BuildPyRuleView, channel DecisionViewChannel).

Message (version 1): {"v", "kind", "decision", "t", "machine", "wip", "fleet", "nAgvs", "travelPrice", "pt" (routing:
{"alpha", "shape", "tau", "reroute", "avail"}; dispatch: null), "jobs": [{"id", "op", "ops", "arrival", "since", "due"
(null = none), "p", "rw", "winq", "m" (routing: [[machine id, p, load, utilization, travel], ...])}]}. Float32
quantities arrive with 9 significant digits and doubles with 17, so float32(value) / float(value) are the player's exact
values. Unity has no operation due dates: op_due = +inf (MOD then ranks every job equal and keeps the first).
"""
import json

import numpy as np

from .views import DecisionView, JobView, MachineView

f32 = np.float32
VIEW_VERSION = 1


def parse_view(raw):
    """DecisionView from the JSON text (or an already parsed dict)."""
    d = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    if d.get("v") != VIEW_VERSION:
        raise ValueError(f"decision view version {d.get('v')!r}, this code reads {VIEW_VERSION}: player and env/ differ")
    now = float(d["t"])
    routing = d["kind"] == "routing"
    jobs = []
    for j in d["jobs"]:
        machines = None
        if routing:
            machines = tuple(MachineView(int(m[0]), f32(m[1]), f32(m[2]), f32(m[3]), f32(m[4])) for m in j["m"])
        else:
            machines = ()
        jobs.append(JobView(j["id"], j["op"], j["ops"], f32(j["arrival"]), float(j["since"]), now,
                            None if j["due"] is None else float(f32(j["due"])), None, f32(j["p"]),
                            remaining_work=f32(j["rw"]), winq=f32(j["winq"]), machines=machines))
    pt = d.get("pt")
    if pt is not None:
        pt = {"alpha": float(pt["alpha"]), "shape": str(pt["shape"]), "tau": float(pt["tau"]),
              "reroute": float(pt["reroute"]), "avail": float(pt["avail"])}
    return DecisionView(d["kind"], now, d["decision"], jobs, d.get("machine", -1), d.get("wip", 0), d.get("fleet", 0),
                        d.get("nAgvs", 0), pt, float(f32(d.get("travelPrice", 0.0))))
