## @file compare_mirror.py
#  @brief Compares C# DES makespan results against the Python mirror DES implementation.
#
#  @details Both implementations share the same architecture — event-driven simulation
#  with per-machine queues and identical dispatching rules with identical tiebreakers —
#  so any delta in their output indicates a real bug in one of the two implementations,
#  not an expected divergence.
#
#  @par Prerequisites
#  -# Run the C# @c ValidationExporter to produce @c csharp_makespans.csv and
#     @c csharp_schedules.json.
#  -# Run @c mirror_des.py to produce @c mirror_des_makespans.csv and
#     @c mirror_des_schedules.json.
#  -# Run this script from the directory containing all four files:
#     @code
#     python compare_mirror.py
#     @endcode
#
#  @par Output
#  - A per-rule exact-match summary printed to stdout.
#  - An instance × rule comparison table showing @c OK or a signed integer delta.
#  - For any mismatches, the first divergent operation (by job/op key) is identified
#    by diffing @c csharp_schedules.json against @c mirror_des_schedules.json.
#  - @c mirror_comparison.csv written to the working directory containing the full
#    merged DataFrame with @c match and @c delta columns.
#
#  @par Exit codes
#  - @c 0 — all compared pairs matched exactly.
#  - @c 1 — one or more mismatches found, or required input files were missing.

import pandas as pd
import json
import os
import sys
from pathlib import Path


def main():
    ## @brief Entry point — loads, merges, and compares C# and mirror DES CSV results.
    #
    #  @details Executes the following steps:
    #  -# Validates that both input CSVs exist; exits with code @c 1 if either is missing.
    #  -# Loads @c csharp_makespans.csv and @c mirror_des_makespans.csv into DataFrames,
    #     deduplicating on @c (instance, rule) keeping the last occurrence.
    #  -# Merges the two DataFrames on @c (instance, rule) with suffixes @c _cs and @c _py.
    #     Exits with code @c 1 and diagnostic info if no common pairs are found.
    #  -# Computes a boolean @c match column and a signed integer @c delta column
    #     (@c makespan_cs - @c makespan_py).
    #  -# Prints an overall match summary, a per-rule breakdown, and an instance × rule table.
    #  -# If mismatches exist and both schedule JSON files are present, identifies the first
    #     divergent operation for each mismatched pair by comparing sorted @c (job, op_index)
    #     keys across @c csharp_schedules.json and @c mirror_des_schedules.json.
    #  -# Saves the full merged comparison to @c mirror_comparison.csv.
    #
    #  @returns @c 0 if all compared pairs matched exactly, @c 1 otherwise.

    data_dir = Path("../../../Resources/Validation")

    # Load both CSVs
    if not os.path.exists(data_dir / "csharp_makespans.csv"):
        print("ERROR: csharp_makespans.csv not found. Run C# ValidationExporter first.")
        return 1
    if not os.path.exists(data_dir / "mirror_des_makespans.csv"):
        print("ERROR: mirror_des_makespans.csv not found. Run mirror_des.py first.")
        return 1

    cs = pd.read_csv(data_dir / "csharp_makespans.csv")
    py = pd.read_csv(data_dir / "mirror_des_makespans.csv")

    # Deduplicate
    cs = cs.drop_duplicates(subset=["instance", "rule"], keep="last")
    py = py.drop_duplicates(subset=["instance", "rule"], keep="last")

    print(f"C# rows:     {len(cs)} ({cs['instance'].nunique()} instances)")
    print(f"Mirror rows: {len(py)} ({py['instance'].nunique()} instances)")

    # Merge
    merged = cs.merge(py, on=["instance", "rule"], suffixes=("_cs", "_py"))
    if merged.empty:
        print("\nERROR: No common (instance, rule) pairs found!")
        print(f"  C# instances:     {sorted(cs['instance'].unique())[:5]}")
        print(f"  Mirror instances: {sorted(py['instance'].unique())[:5]}")
        print(f"  C# rules:     {sorted(cs['rule'].unique())}")
        print(f"  Mirror rules: {sorted(py['rule'].unique())}")
        return 1

    merged["match"] = merged["makespan_cs"] == merged["makespan_py"]
    merged["delta"] = merged["makespan_cs"] - merged["makespan_py"]

    matches = merged["match"].sum()
    total = len(merged)
    mismatches = total - matches

    print(f"\n{'=' * 60}")
    print(f"C# vs MIRROR DES (same architecture)")
    print(f"{'=' * 60}")
    print(f"Compared:    {total}")
    print(f"EXACT MATCH: {matches}")
    print(f"Mismatch:    {mismatches}")

    if matches == total:
        print(f"\n*** PERFECT MATCH — your C# DES is correct! ***")
    print()

    # Per-rule breakdown
    print("PER-RULE:")
    for rule in sorted(merged["rule"].unique()):
        rd = merged[merged["rule"] == rule]
        rm = rd["match"].sum()
        print(f"  {rule:6s}  {rm}/{len(rd)} exact")
    print()

    # Instance x rule table: OK or signed delta
    instances = sorted(merged["instance"].unique())
    rules = sorted(merged["rule"].unique())
    header = f"{'Instance':>10s}  " + "  ".join(f"{r:>8s}" for r in rules)
    print(header)
    print("-" * len(header))

    for inst in instances:
        parts = [f"{inst:>10s}"]
        for rule in rules:
            row = merged[(merged["instance"] == inst) & (merged["rule"] == rule)]
            if row.empty:
                parts.append(f"{'---':>8s}")
            elif row.iloc[0]["match"]:
                parts.append(f"{'OK':>8s}")
            else:
                d = int(row.iloc[0]["delta"])
                parts.append(f"{d:>+8d}")
        print("  ".join(parts))

    # Diagnose mismatches by finding the first divergent operation
    if mismatches > 0:
        print(f"\n{'=' * 60}")
        print("MISMATCHES (bug in C# or Python mirror)")
        print(f"{'=' * 60}")

        for _, row in merged[~merged["match"]].iterrows():
            print(f"\n  {row['instance']} / {row['rule']}:")
            print(f"    C#:     {int(row['makespan_cs'])}")
            print(f"    Mirror: {int(row['makespan_py'])}")
            print(f"    Delta:  {int(row['delta'])}")

        cs_sched_path = "csharp_schedules.json"
        py_sched_path = "mirror_des_schedules.json"
        if os.path.exists(cs_sched_path) and os.path.exists(py_sched_path):
            with open(cs_sched_path) as f:
                cs_scheds = json.load(f)
            with open(py_sched_path) as f:
                py_scheds = json.load(f)

            for _, row in merged[~merged["match"]].iterrows():
                ## Schedule entries are keyed as "{instance}_{rule}" in both JSON files,
                #  matching the key format written by ValidationExporter.RunAndExport.
                key = f"{row['instance']}_{row['rule']}"
                cs_ops = cs_scheds.get(key, [])
                py_ops = py_scheds.get(key, [])
                if not cs_ops or not py_ops:
                    continue

                cs_by_key = {(op["job"], op["op_index"]): op for op in cs_ops}
                py_by_key = {(op["job"], op["op_index"]): op for op in py_ops}

                ## Walk operations in sorted (job, op_index) order and report the first
                #  pair where start time or machine assignment diverges between the two
                #  implementations. Earlier divergences are more likely to be root causes.
                first_div = None
                for k in sorted(cs_by_key.keys()):
                    c = cs_by_key.get(k)
                    p = py_by_key.get(k)
                    if (
                        c
                        and p
                        and (c["start"] != p["start"] or c["machine"] != p["machine"])
                    ):
                        first_div = (k, c, p)
                        break

                if first_div:
                    k, c, p = first_div
                    print(f"\n  {key} first divergence: Job {k[0]}, Op {k[1]}")
                    print(f"    C#:     start={c['start']}, machine={c['machine']}")
                    print(f"    Mirror: start={p['start']}, machine={p['machine']}")

    merged.to_csv(data_dir / "mirror_comparison.csv", index=False)
    print(f"\nSaved to mirror_comparison.csv")

    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
