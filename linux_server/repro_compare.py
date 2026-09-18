#!/usr/bin/env python3
"""
repro_compare.py

Companion to repro_determinism_check.sh. Reads the results_<rule>_run<i>.csv
files produced by N repeated invocations of the same (config, rule, seed) and
checks whether the outcome is identical across runs.

Before the SimTime fix, this would fail badly on stress scenarios (e.g.
l008_mf_high showed jobs_censored swinging from 14 to 197 on an identical
seed). After the fix, every column below should match exactly (or, for the
float columns, within a tiny rounding tolerance) across all N runs.
"""
import argparse
import csv
import os
import sys

# Columns that should be a pure function of (config, rule, seed) and are
# therefore the ones that matter for this check. Excludes timestamp (wall
# clock, expected to differ) and metadata echoed straight from config.
CHECK_COLUMNS = [
    "jobs", "total_ops", "decisions", "total_reward",
    "mean_interarrival_realised", "last_arrival_sim_time",
    "episode_failures", "total_repair_time",
    "mean_flow_time", "p95_flow_time", "max_flow_time", "mean_transport_wait",
    "jobs_censored",
    "mean_flow_time_penalized", "p95_flow_time_penalized", "max_flow_time_penalized",
    "deadlock_detected", "deadlock_sim_time",
]

FLOAT_TOLERANCE = 1e-6


def read_single_row(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[-1]  # last row, in case the file was appended to


def values_match(a, b):
    if a is None or b is None:
        return a == b
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= FLOAT_TOLERANCE
    except ValueError:
        return a == b


def check_mode(results_root, mode, rules, runs):
    print("=" * 88)
    print(f"MODE: {mode}")
    print("=" * 88)
    any_fail = False
    for rule in rules:
        rows = []
        for i in range(1, runs + 1):
            path = os.path.join(results_root, mode, f"results_{rule}_run{i}.csv")
            row = read_single_row(path)
            rows.append(row)

        missing = [i + 1 for i, r in enumerate(rows) if r is None]
        if missing:
            print(f"[{rule:12s}] MISSING output for run(s) {missing} -- check worker logs")
            any_fail = True
            continue

        mismatches = []
        for col in CHECK_COLUMNS:
            if col not in rows[0]:
                continue
            base = rows[0][col]
            for i, row in enumerate(rows[1:], start=2):
                if not values_match(base, row.get(col)):
                    mismatches.append((col, 1, base, i, row.get(col)))

        if mismatches:
            any_fail = True
            print(f"[{rule:12s}] FAIL -- {len(mismatches)} column(s) differ across {runs} runs:")
            for col, run_a, val_a, run_b, val_b in mismatches:
                print(f"      {col:28s} run{run_a}={val_a!s:>12s}   run{run_b}={val_b!s:>12s}")
        else:
            comp = rows[0].get("jobs_censored", "?")
            mft = rows[0].get("mean_flow_time", "?")
            print(f"[{rule:12s}] PASS -- {runs}/{runs} runs identical "
                  f"(jobs_censored={comp}, mean_flow_time={mft})")

    print()
    return not any_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--rules", required=True, help="comma-separated rule names")
    ap.add_argument("--runs", type=int, required=True)
    ap.add_argument("--mode", default="both", choices=["sequential", "parallel", "both"])
    args = ap.parse_args()

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    modes = ["sequential", "parallel"] if args.mode == "both" else [args.mode]

    all_ok = True
    for mode in modes:
        ok = check_mode(args.results_root, mode, rules, args.runs)
        all_ok = all_ok and ok

    if all_ok:
        print("RESULT: all rules reproduced identically across all runs and modes.")
        sys.exit(0)
    else:
        print("RESULT: at least one rule did NOT reproduce identically -- see FAIL lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
