"""Summarise run_experiment_queue.py sweeps (E1 load x fleet; hold vs release protocol).

Reads Results/<exp>/<scenario>/<layout>/agv<N>_<RULE>_s<seed>/ for every --exps tag and writes one row per run
(runs.csv) plus printed summaries. Safety first: a single AGV collision is disqualifying for a real floor, so the
collision table (events, overlap seconds, near misses, closest approach, parking overlaps) is printed before any
performance number.

Per run:
  safety      agv_collision_events, agv_collision_pair_seconds, agv_clearance_events (near misses),
              agv_min_centre_distance, agv_parking_overlap_events
  liveness    deadlock_detected, stalled (first_stall_sim_time >= 0), unfinished (jobs_censored > 0)
  performance mean / p95 flow, throughput (jobs per 1000 sim-s), transport share of flow
  utilisation AGV busy (travel + load + unload) / makespan, AGV route-wait / makespan, machine utilisation
  phases      compound scenarios: mean flow of jobs arriving in each phase kind (bottleneck, standoff, ...)

Usage (from repo root):
  python results/scripts/analyze_load_sweep.py --exps E1_hold E1_rel --out results/E1_runs.csv
"""
import argparse, glob, json, os, re
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "linux_server", "Results")
SCEN = os.path.join(os.path.dirname(__file__), "..", "..", "linux_server", "BatchConfigs", "Scenarios")


def phases_for(scenario):
    base = re.sub(r"_fail$", "", scenario)
    try:
        return json.load(open(os.path.join(SCEN, base + ".json"))).get("_phases")
    except FileNotFoundError:
        return None


def one_run(path, exp):
    rel = os.path.relpath(path, os.path.join(RESULTS, exp)).split(os.sep)
    scenario, layout, cell = rel[0], rel[1], rel[2]
    m = re.match(r"agv(\d+)_(.+)_s(\d+)$", cell)
    r = pd.read_csv(os.path.join(path, "results.csv")).iloc[0]
    row = dict(exp=exp, scenario=scenario, failures=scenario.endswith("_fail") or "shard" in scenario,
               layout=layout, agv=int(m[1]), rule=m[2], seed=int(m[3]), protocol=r.reservation_protocol,
               collisions=r.agv_collision_events, collision_s=r.agv_collision_pair_seconds,
               near_misses=r.agv_clearance_events, min_centre=r.agv_min_centre_distance,
               parking_overlaps=r.agv_parking_overlap_events,
               deadlock=r.deadlock_detected, stalled=r.first_stall_sim_time >= 0, unfinished=r.jobs_censored > 0,
               makespan=r.makespan, jobs=r.jobs, flow=r.mean_flow_time, p95=r.p95_flow_time,
               flow_pen=r.mean_flow_time_penalized, failures_n=r.episode_failures,
               throughput=1000.0 * r.jobs / r.makespan)
    a = pd.read_csv(os.path.join(path, "agv_performance.csv"))
    ms = a.makespan.iloc[0]
    row["agv_busy"] = ((a.time_traveling + a.time_loading + a.time_unloading) / ms).mean()
    row["agv_route_wait"] = (a.time_waiting_route / ms).mean()
    mu = pd.read_csv(os.path.join(path, "machine_utilization.csv"))
    row["machine_util"] = mu.utilization_rate.mean()
    j = pd.read_csv(os.path.join(path, "job_completions.csv"))
    done = j[j.completed == 1]
    row["transport_share"] = ((done.time_waiting_pickup + done.time_in_transit).sum() / max(done.flow_time.sum(), 1e-9))
    ph = phases_for(scenario)
    if ph:
        for p in ph:
            kind = re.sub(r"_\d+$", "", p["name"])
            sel = done[(done.arrival_time >= p["start"]) & (done.arrival_time < p["end"])]
            row.setdefault(f"flow_{kind}", []).extend(sel.flow_time.tolist())
        for k in [k for k in row if k.startswith("flow_") and isinstance(row[k], list)]:
            row[k] = sum(row[k]) / len(row[k]) if row[k] else float("nan")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--results-root", default=None, help="Results dir of another build, e.g. linux_server_e2/Results")
    a = ap.parse_args()
    global RESULTS
    if a.results_root: RESULTS = a.results_root
    rows = []
    for exp in a.exps:
        for res in glob.glob(os.path.join(RESULTS, exp, "*", "*", "*", "results.csv")):
            try: rows.append(one_run(os.path.dirname(res), exp))
            except Exception as e: print("skip", res, e)
    df = pd.DataFrame(rows)
    if df.empty: print("no runs yet"); return
    if a.out: df.to_csv(a.out, index=False); print("wrote", a.out, len(df), "runs")
    pd.set_option("display.width", 200)

    print("\n== SAFETY (sum over runs; any nonzero collision count is disqualifying) ==")
    g = df.groupby(["protocol", "layout"])
    print(g.agg(runs=("seed", "size"), collisions=("collisions", "sum"), collision_s=("collision_s", "sum"),
                runs_with_collision=("collisions", lambda s: (s > 0).sum()), near_misses=("near_misses", "sum"),
                min_centre=("min_centre", "min"), parking_overlaps=("parking_overlaps", "sum")).to_string())

    print("\n== LIVENESS (fraction of runs) ==")
    print(df.groupby(["protocol", "layout", "agv"])[["deadlock", "stalled", "unfinished"]].mean()
            .unstack("agv").round(2).to_string())

    print("\n== MEAN FLOW by scenario x AGVs (mean over layouts, rules, seeds) ==")
    print(df.pivot_table(index=["scenario", "protocol"], columns="agv", values="flow", aggfunc="mean").round(0).to_string())

    print("\n== AGV busy / route-wait fraction and machine utilisation (mean) ==")
    print(df.pivot_table(index=["scenario", "protocol"], columns="agv", values=["agv_busy", "agv_route_wait"],
                         aggfunc="mean").round(2).to_string())
    print(df.pivot_table(index=["scenario"], columns="agv", values="machine_util", aggfunc="mean").round(2).to_string())


if __name__ == "__main__":
    main()
