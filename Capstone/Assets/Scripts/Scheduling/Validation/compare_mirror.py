"""
Compare C# DES against Mirror DES (Python).
=============================================

Both implement the same architecture:
  - Event-driven, per-machine queues
  - Same dispatching rules with same tiebreakers

Any delta here means a real bug in one of the two implementations.

Usage:
    1. Run C# ValidationExporter  → csharp_makespans.csv
    2. Run mirror_des.py          → mirror_des_makespans.csv
    3. Run this script            → comparison report

    python compare_mirror.py
"""

import pandas as pd
import json
import os
import sys


def main():
    # Load both CSVs
    if not os.path.exists("csharp_makespans.csv"):
        print("ERROR: csharp_makespans.csv not found. Run C# ValidationExporter first.")
        return 1
    if not os.path.exists("mirror_des_makespans.csv"):
        print("ERROR: mirror_des_makespans.csv not found. Run mirror_des.py first.")
        return 1

    cs = pd.read_csv("csharp_makespans.csv")
    py = pd.read_csv("mirror_des_makespans.csv")

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

    # Per-rule
    print("PER-RULE:")
    for rule in sorted(merged["rule"].unique()):
        rd = merged[merged["rule"] == rule]
        rm = rd["match"].sum()
        print(f"  {rule:6s}  {rm}/{len(rd)} exact")
    print()

    # Show table
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

    # Diagnose mismatches
    if mismatches > 0:
        print(f"\n{'=' * 60}")
        print("MISMATCHES (bug in C# or Python mirror)")
        print(f"{'=' * 60}")

        for _, row in merged[~merged["match"]].iterrows():
            print(f"\n  {row['instance']} / {row['rule']}:")
            print(f"    C#:     {int(row['makespan_cs'])}")
            print(f"    Mirror: {int(row['makespan_py'])}")
            print(f"    Delta:  {int(row['delta'])}")

        # Try to find first divergent operation
        cs_sched_path = "csharp_schedules.json"
        py_sched_path = "mirror_des_schedules.json"
        if os.path.exists(cs_sched_path) and os.path.exists(py_sched_path):
            with open(cs_sched_path) as f:
                cs_scheds = json.load(f)
            with open(py_sched_path) as f:
                py_scheds = json.load(f)

            for _, row in merged[~merged["match"]].iterrows():
                key = f"{row['instance']}_{row['rule']}"
                cs_ops = cs_scheds.get(key, [])
                py_ops = py_scheds.get(key, [])
                if not cs_ops or not py_ops:
                    continue

                cs_by_key = {(op["job"], op["op_index"]): op for op in cs_ops}
                py_by_key = {(op["job"], op["op_index"]): op for op in py_ops}

                first_div = None
                for k in sorted(cs_by_key.keys()):
                    c = cs_by_key.get(k)
                    p = py_by_key.get(k)
                    if c and p and (c["start"] != p["start"] or c["machine"] != p["machine"]):
                        first_div = (k, c, p)
                        break

                if first_div:
                    k, c, p = first_div
                    print(f"\n  {key} first divergence: Job {k[0]}, Op {k[1]}")
                    print(f"    C#:     start={c['start']}, machine={c['machine']}")
                    print(f"    Mirror: start={p['start']}, machine={p['machine']}")

    merged.to_csv("mirror_comparison.csv", index=False)
    print(f"\nSaved to mirror_comparison.csv")

    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())