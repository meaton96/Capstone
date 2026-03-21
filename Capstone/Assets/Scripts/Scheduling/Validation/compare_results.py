"""
Cross-Validation Comparator (v2)
=================================
Compares C# DES output against job_shop_lib reference.

Handles:
  - Duplicate rows (from re-running exporters)
  - Missing rules on either side
  - Large-scale mismatch diagnosis

Usage:
    python compare_results.py
"""

import json
import pandas as pd
import sys
import os


def load_data():
    ref = pd.read_csv("reference_makespans.csv")
    csharp = pd.read_csv("csharp_makespans.csv")

    # Deduplicate: keep last occurrence (most recent run)
    ref = ref.drop_duplicates(subset=["instance", "rule"], keep="last")
    csharp = csharp.drop_duplicates(subset=["instance", "rule"], keep="last")

    ref_schedules, cs_schedules = {}, {}
    if os.path.exists("reference_schedules.json"):
        with open("reference_schedules.json") as f:
            ref_schedules = json.load(f)
    if os.path.exists("csharp_schedules.json"):
        with open("csharp_schedules.json") as f:
            cs_schedules = json.load(f)

    return ref, csharp, ref_schedules, cs_schedules


def find_first_divergence(ref_ops, cs_ops):
    ref_by_key = {(op["job"], op["op_index"]): op for op in ref_ops}
    cs_by_key  = {(op["job"], op["op_index"]): op for op in cs_ops}
    all_keys = sorted(set(ref_by_key.keys()) | set(cs_by_key.keys()))

    divergences = []
    for key in all_keys:
        r = ref_by_key.get(key)
        c = cs_by_key.get(key)
        if r is None or c is None:
            divergences.append({
                "job": key[0], "op_index": key[1],
                "issue": "missing in " + ("C#" if c is None else "reference"),
            })
            continue
        if r["start"] != c["start"] or r["machine"] != c["machine"]:
            divergences.append({
                "job": key[0], "op_index": key[1],
                "ref_start": r["start"], "cs_start": c["start"],
                "ref_machine": r["machine"], "cs_machine": c["machine"],
                "ref_end": r["end"], "cs_end": c["end"],
                "duration": r["duration"],
                "issue": "timing/assignment differs",
            })
    divergences.sort(key=lambda d: d.get("ref_start", 0))
    return divergences


def main():
    ref, csharp, ref_schedules, cs_schedules = load_data()

    print(f"Reference rows:  {len(ref)} ({ref['instance'].nunique()} instances, {ref['rule'].nunique()} rules)")
    print(f"C# rows:         {len(csharp)} ({csharp['instance'].nunique()} instances, {csharp['rule'].nunique()} rules)")
    print()

    # ── Find common (instance, rule) pairs ────────────────────────────────
    ref_keys = set(zip(ref["instance"], ref["rule"]))
    cs_keys  = set(zip(csharp["instance"], csharp["rule"]))
    common   = ref_keys & cs_keys
    only_ref = ref_keys - cs_keys
    only_cs  = cs_keys - ref_keys

    if only_ref:
        print(f"Only in reference (not in C#): {len(only_ref)} pairs")
        for inst, rule in sorted(only_ref)[:10]:
            print(f"  {inst} / {rule}")
        if len(only_ref) > 10:
            print(f"  ... and {len(only_ref) - 10} more")
        print()

    if only_cs:
        print(f"Only in C# (not in reference): {len(only_cs)} pairs")
        for inst, rule in sorted(only_cs)[:10]:
            print(f"  {inst} / {rule}")
        if len(only_cs) > 10:
            print(f"  ... and {len(only_cs) - 10} more")
        print()

    if not common:
        print("ERROR: No common (instance, rule) pairs to compare!")
        print("Check that both CSVs use the same instance names and rule keys.")
        print(f"  Reference rules: {sorted(ref['rule'].unique())}")
        print(f"  C# rules:        {sorted(csharp['rule'].unique())}")
        print(f"  Reference instances (first 5): {sorted(ref['instance'].unique())[:5]}")
        print(f"  C# instances (first 5):        {sorted(csharp['instance'].unique())[:5]}")
        return 1

    # ── Merge and compare ─────────────────────────────────────────────────
    merged = ref.merge(csharp, on=["instance", "rule"], suffixes=("_ref", "_cs"))
    merged["match"] = merged["makespan_ref"] == merged["makespan_cs"]
    merged["delta"] = merged["makespan_cs"] - merged["makespan_ref"]
    merged["delta_pct"] = (merged["delta"] / merged["makespan_ref"] * 100).round(2)

    matches = merged["match"].sum()
    mismatches = len(merged) - matches

    print("=" * 70)
    print("CROSS-VALIDATION REPORT")
    print("=" * 70)
    print(f"Comparable pairs:   {len(merged)}")
    print(f"Exact matches:      {matches}  ✓")
    print(f"Mismatches:         {mismatches}  ✗")
    print()

    # ── Per-rule summary ──────────────────────────────────────────────────
    print("PER-RULE SUMMARY")
    print("-" * 70)
    for rule in sorted(merged["rule"].unique()):
        rule_data = merged[merged["rule"] == rule]
        rule_matches = rule_data["match"].sum()
        rule_total = len(rule_data)
        avg_delta = rule_data["delta_pct"].abs().mean()
        print(f"  {rule:6s}  {rule_matches}/{rule_total} exact  "
              f"avg |delta|={avg_delta:.1f}%  "
              f"max delta={rule_data['delta_pct'].max():+.1f}%")
    print()

    # ── Instance-by-instance table ────────────────────────────────────────
    print("INSTANCE x RULE COMPARISON (showing delta)")
    print("-" * 70)

    instances = sorted(merged["instance"].unique())
    rules = sorted(merged["rule"].unique())

    # Header
    header = f"{'Instance':>10s}  " + "  ".join(f"{r:>8s}" for r in rules)
    print(header)
    print("-" * len(header))

    for inst in instances:
        row_parts = [f"{inst:>10s}"]
        for rule in rules:
            mask = (merged["instance"] == inst) & (merged["rule"] == rule)
            subset = merged[mask]
            if subset.empty:
                row_parts.append(f"{'---':>8s}")
            elif subset.iloc[0]["match"]:
                row_parts.append(f"{'OK':>8s}")
            else:
                delta = int(subset.iloc[0]["delta"])
                row_parts.append(f"{delta:>+8d}")
        print("  ".join(row_parts))
    print()

    # ── Diagnose worst mismatches ─────────────────────────────────────────
    if mismatches > 0:
        print("=" * 70)
        print("TOP 10 WORST MISMATCHES (by absolute delta)")
        print("=" * 70)

        worst = merged[~merged["match"]].sort_values("delta", key=abs, ascending=False).head(10)
        for _, row in worst.iterrows():
            inst = row["instance"]
            rule = row["rule"]
            print(f"\n  {inst} / {rule}:")
            print(f"    Reference: {int(row['makespan_ref'])}    C#: {int(row['makespan_cs'])}    "
                  f"Delta: {int(row['delta']):+d} ({row['delta_pct']:+.1f}%)")

            sched_key = f"{inst}_{rule}"
            ref_ops = ref_schedules.get(sched_key, [])
            cs_ops = cs_schedules.get(sched_key, [])

            if ref_ops and cs_ops:
                divergences = find_first_divergence(ref_ops, cs_ops)
                if divergences:
                    d = divergences[0]
                    print(f"    First divergence: Job {d['job']}, Op {d['op_index']}")
                    if "ref_start" in d:
                        print(f"      Ref:  start={d['ref_start']}, machine={d['ref_machine']}, end={d['ref_end']}")
                        print(f"      C#:   start={d['cs_start']}, machine={d['cs_machine']}, end={d['cs_end']}")
                    print(f"    Total divergent ops: {len(divergences)} / {len(ref_ops)}")

        # ── Quick diagnosis hint ──────────────────────────────────────────
        print("\n" + "=" * 70)
        print("DIAGNOSIS HINTS")
        print("=" * 70)

        deltas = merged[~merged["match"]]["delta"]
        all_positive = (deltas > 0).all()
        all_negative = (deltas < 0).all()

        if all_positive:
            print("  C# makespan is ALWAYS higher than reference.")
            print("  Likely cause: C# dispatching is less aggressive or has")
            print("  different queue ordering than job_shop_lib.")
        elif all_negative:
            print("  C# makespan is ALWAYS lower than reference.")
            print("  Likely cause: C# may be dispatching more aggressively,")
            print("  or there's a precedence constraint being skipped.")
        else:
            print("  Deltas go both directions (some +, some -).")
            print("  This is typical of tiebreaking differences.")

        fcfs_data = merged[merged["rule"] == "FCFS"] if "FCFS" in merged["rule"].values else pd.DataFrame()
        if not fcfs_data.empty:
            fcfs_matches = fcfs_data["match"].sum()
            print(f"\n  FCFS matches: {fcfs_matches}/{len(fcfs_data)}")
            if fcfs_matches == len(fcfs_data):
                print("  FCFS is exact -> base DES event logic is correct.")
                print("  Mismatches in other rules -> dispatching rule implementation differs.")
            else:
                print("  FCFS has mismatches -> base DES event ordering may differ.")
                print("  Fix FCFS first before debugging other rules.")

    # Save
    merged.to_csv("comparison_results.csv", index=False)
    print(f"\nFull comparison saved to comparison_results.csv")

    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())