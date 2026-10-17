"""agv-ratio-twin-v5 (2026-10-14): ../agv-ratio-scale/twin_point.py with TWK due dates (c ~ U[1.75, 2.5], as B2) so the
v5 due-date rules (MDD / ATC x PECT / PTECT) apply, and total tardiness in the output. Otherwise unchanged: one twin
run of an rnd_load-family instance scaled to an M-machine floor, fixed rule, given fleet; prints summary + timing.

usage: twin_point.py FLOOR_JSON M AGVS [--seed S] [--horizon H] [--rule SRT_TECT] [--transport kinematic]
       [--util lo hi] [--cap SECONDS]
The instance is generated with agv_count = M (bays) and agv_max_utilization = 10 so the arrival stream is NOT
capped by the fleet (the randomized generator caps arrivals at 0.75 of an assumed 64 s/move fleet capacity, which
would tie load to fleet size); it is the same job list at every fleet size.
"""
import argparse, json, os, sys, time
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../env")]
from scenarios.randomized import RandomizedParams, randomized_scenario
from des_twin.floor import Floor
from des_twin.scenario import resolve_jobs
from des_twin.engine import TwinConfig, Twin

ap = argparse.ArgumentParser()
ap.add_argument("floor"); ap.add_argument("m", type=int); ap.add_argument("agvs", type=int)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--horizon", type=float, default=21600.0)
ap.add_argument("--rule", default="MDD_PTECT"); ap.add_argument("--transport", default="kinematic")
ap.add_argument("--util", type=float, nargs=2, default=(0.7, 1.4))
ap.add_argument("--opmean", type=float, nargs=2, default=(240.0, 480.0))
ap.add_argument("--cap", type=float, default=0.0, help="episode cap after arrivals start (0 = run to empty)")
ap.add_argument("--due", type=float, nargs=2, default=(1.75, 2.5), help="due-date allowance range c (TWK)")
ap.add_argument("--json", action="store_true")
a = ap.parse_args()

p = RandomizedParams(machines_per_type=a.m // 5, agv_count=a.m, agv_max_utilization=10.0, failure_probability=0.0,
                     horizon_seconds=a.horizon, utilization=tuple(a.util), op_mean_seconds=tuple(a.opmean),
                     due_date_allowance_range=tuple(a.due))
scen = randomized_scenario(a.seed, p)
floor = Floor.load(a.floor)
t0 = time.time()
jobs = resolve_jobs(scen, floor)
cfg = TwinConfig(rule=a.rule, transport=a.transport, agv_count=a.agvs, episode_duration_seconds=a.cap,
                 max_sim_seconds=a.horizon * 4)
tw = Twin(floor, jobs, cfg)
ticks = 0
gen = tw._events()
for _ in gen:
    raise RuntimeError
wall = time.time() - t0
s = tw.summary()
m = tw.metrics()
s.update(m=a.m, seed=a.seed, wall_s=round(wall, 2), sim_s=round(tw.now, 1), n_jobs_instance=len(scen["jobs"]),
         time_in_system_sum=m["time_in_system_sum"], wip_end=m["wip"], tardiness_sum=m["tardiness_sum"],
         jobs_late=m.get("jobs_late"),
         mean_tis=m["time_in_system_sum"] / max(1, m["jobs_total"]), truncated=tw.truncated,
         agv_trips_per_agv_h=s["agv_trips"] / max(1, a.agvs) / (tw.now / 3600.0),
         sim_per_wall=round(tw.now / max(wall, 1e-9), 1), decisions_per_s=round(tw.decisions / max(wall, 1e-9), 1))
if a.json:
    print(json.dumps(s))
else:
    for k in ("m", "agv_count", "rule", "n_jobs_instance", "jobs_completed", "makespan", "mean_flow_time", "mean_tis",
              "mean_transport_wait", "agv_busy_fraction", "machine_util_mean", "agv_path_length", "agv_trips",
              "decisions", "wall_s", "sim_per_wall", "decisions_per_s", "timed_out"):
        print(f"{k:22s} {s[k]}")
