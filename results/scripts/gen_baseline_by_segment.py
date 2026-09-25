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


if __name__ == "__main__":
    main()
