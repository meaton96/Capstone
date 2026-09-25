#!/usr/bin/env python3
"""
Best dispatching rule per seed and per arrival segment on the randomized training family.

Answers: does one rule win everywhere (then a policy can just learn it), or does the best rule change with the
segment kind / load / failures (the room a learned policy has to beat every single rule)?

Each job is assigned to the segment it ARRIVED in (the instance's _phases); its flow time is charged there.
Reads Results/<exp>/randomized_s<seed>/<layout>/agv<N>_<RULE>_s<seed>/job_completions.csv and the instance JSON
in BatchConfigs/Scenarios.

Usage:
    python3 results/scripts/gen_baseline_by_segment.py --results linux_server/Results/gen_baseline \
        --scenarios linux_server/BatchConfigs/Scenarios
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
from collections import Counter, defaultdict


def load(results, scen_dir, prefix="randomized"):
    """-> {seed: {rule: [(arrival, flow, completed)]}}, {seed: phases}, {seed: meta}"""
    runs, phases, meta = defaultdict(dict), {}, {}
    for path in glob.glob(os.path.join(results, f"{prefix}_s*", "*", "*", "job_completions.csv")):
        cell = os.path.basename(os.path.dirname(path))          # agv7_SPT_SMPT_s3
        rule = "_".join(cell.split("_")[1:-1])
        seed = int(cell.rsplit("_s", 1)[1])
        rows = list(csv.DictReader(open(path)))
        runs[seed][rule] = [(float(r["arrival_time"]), float(r["flow_time"]), r["completed"] == "1") for r in rows]
        if seed not in phases:
            sc = json.load(open(os.path.join(scen_dir, f"{prefix}_s{seed}.json")))
            phases[seed], meta[seed] = sc["_phases"], sc["_meta"]
    return runs, phases, meta


def segment_of(t, ph):
    for i, p in enumerate(ph):
        if p["start"] <= t < p["end"]:
            return i
    return len(ph) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--prefix", default="randomized", help="scenario name prefix (<prefix>_s<seed>)")
    ap.add_argument("--rules", nargs="*", help="only these rules (e.g. to compare sweeps over the same rule set)")
    ap.add_argument("--min-jobs", type=int, default=5, help="skip segments with fewer arrivals")
    a = ap.parse_args()
    runs, phases, meta = load(a.results, a.scenarios, a.prefix)
    if a.rules:
        runs = {seed: {r: v for r, v in rr.items() if r in a.rules} for seed, rr in runs.items()}

    print("== Mean flow per seed (s); * = best, gap = best rule vs 2nd best")
    wins = Counter()
    for seed in sorted(runs):
        m = {r: st.mean(f for _, f, c in v if c) for r, v in runs[seed].items()}
        order = sorted(m, key=m.get)
        wins[order[0]] += 1
        fail = "fail" if meta[seed]["failures_on"] else "nofail"
        print(f"s{seed} ({fail}, op {meta[seed]['op_mean_seconds']:.0f}s): "
              + "  ".join(f"{r}{'*' if r == order[0] else ''}={m[r]:.0f}" for r in order)
              + f"   gap {m[order[1]] / m[order[0]] - 1:+.1%}")
    print("seed-level wins:", dict(wins))

    print("\n== Per segment: best rule by mean flow of jobs arriving in the segment")
    seg_wins = defaultdict(Counter)
    gaps = defaultdict(list)                     # best single rule overall vs per-segment best
    for seed in sorted(runs):
        ph = phases[seed]
        per = defaultdict(lambda: defaultdict(list))   # seg -> rule -> flows
        for rule, jobs in runs[seed].items():
            for t, f, c in jobs:
                if c:
                    per[segment_of(t, ph)][rule].append(f)
        for i, p in enumerate(ph):
            if p["n_jobs"] < a.min_jobs or i not in per:
                continue
            m = {r: st.mean(v) for r, v in per[i].items() if v}
            best = min(m, key=m.get)
            seg_wins[p["kind"]][best] += 1
            seg_wins["ALL"][best] += 1
    for kind in ["ALL"] + sorted(k for k in seg_wins if k != "ALL"):
        print(f"{kind:9s} " + "  ".join(f"{r}:{n}" for r, n in seg_wins[kind].most_common()))

    # Oracle: pick the best rule per segment (charging each segment's jobs to that rule's run) vs best single rule.
    # Only an upper-bound hint: switching rules mid-episode changes later segments' state, which this ignores.
    print("\n== Per-seed oracle (best rule per segment) vs best single rule, total flow of completed jobs")
    for seed in sorted(runs):
        ph = phases[seed]
        rules = list(runs[seed])
        tot = {r: sum(f for _, f, c in runs[seed][r] if c) for r in rules}
        seg_tot = defaultdict(lambda: defaultdict(float))
        for r in rules:
            for t, f, c in runs[seed][r]:
                if c:
                    seg_tot[segment_of(t, ph)][r] += f
        oracle = sum(min(v.values()) for v in seg_tot.values())
        best = min(tot, key=tot.get)
        print(f"s{seed}: best single {best} {tot[best]:.0f}, segment oracle {oracle:.0f} ({oracle / tot[best] - 1:+.1%})")

    job_half_spread(runs)
    load_profile(a.results, a.prefix, a.rules)


def load_profile(results, prefix, rules=None):
    """Per rule, over seeds: machine utilization, AGV busy, peak WIP, dispatch decisions with <=1 candidate,
    routing decisions with >1 job in the pool, plus deadlocks / collisions / censored jobs."""
    agg = defaultdict(lambda: defaultdict(list))
    for d in glob.glob(os.path.join(results, f"{prefix}_s*", "*", "*")):
        if not os.path.exists(os.path.join(d, "results.csv")):
            continue
        cell = os.path.basename(d)
        rule = "_".join(cell.split("_")[1:-1])
        if rules and rule not in rules:
            continue
        r = next(csv.DictReader(open(os.path.join(d, "results.csv"))))
        ms = float(r["makespan"])
        mu = [float(x["utilization_rate"]) for x in csv.DictReader(open(os.path.join(d, "machine_utilization.csv")))]
        agv = list(csv.DictReader(open(os.path.join(d, "agv_performance.csv"))))
        wip = [int(x["work_in_progress"]) for x in csv.DictReader(open(os.path.join(d, "throughput.csv")))]
        dec = list(csv.DictReader(open(os.path.join(d, "decision_log.csv"))))
        disp = [x for x in dec if x["decision_type"] == "Dispatch"]
        rout = [x for x in dec if x["decision_type"] == "Routing"]
        a = agg[rule]
        a["util"].append(st.mean(mu)); a["util_max"].append(max(mu))
        a["agv"].append(1 - sum(float(x["time_idle"]) for x in agv) / (ms * len(agv)))
        a["wip_max"].append(max(wip))
        a["disp_deg"].append(sum(int(x["candidate_count"]) <= 1 for x in disp) / max(len(disp), 1))
        a["route_pool"].append(sum(int(x["job_candidate_count"]) > 1 for x in rout) / max(len(rout), 1))
        a["bad"].append(int(r["deadlock_detected"]) + int(r["agv_collision_events"]) + int(r["jobs_censored"]))
    print("\n== Load profile per rule (min-max over seeds)")
    for rule in sorted(agg):
        a = agg[rule]
        rng = lambda k, f="{:.2f}": f"{f.format(min(a[k]))}-{f.format(max(a[k]))}"
        print(f"{rule:10s} util {rng('util')} (max {rng('util_max')}) | AGV busy {rng('agv')} | WIP max "
              f"{rng('wip_max', '{:.0f}')} | dispatch<=1 {rng('disp_deg', '{:.0%}')} | routing pool>1 "
              f"{rng('route_pool', '{:.0%}')} | deadlock+collision+censored {sum(a['bad'])}")


def job_half_spread(runs):
    """Mean-flow spread across job-ordering rules with the machine half held fixed (per seed, then averaged)."""
    groups = defaultdict(list)
    for seed, rr in runs.items():
        by_machine = defaultdict(dict)
        for rule, jobs in rr.items():
            job_half, machine_half = rule.split("_")
            flows = [f for _, f, c in jobs if c]
            if flows:
                by_machine[machine_half][job_half] = st.mean(flows)
        for mh, m in by_machine.items():
            if len(m) > 1:
                groups[mh].append((max(m.values()) / min(m.values()) - 1, min(m, key=m.get)))
    print("\n== Job-ordering spread with the machine half fixed (worst/best job rule - 1, per seed)")
    for mh, v in sorted(groups.items()):
        spreads = [x for x, _ in v]
        print(f"{mh:5s} mean {st.mean(spreads):+.1%} (range {min(spreads):+.1%} to {max(spreads):+.1%}); "
              f"best job rule: {dict(Counter(b for _, b in v))}")


if __name__ == "__main__":
    main()
