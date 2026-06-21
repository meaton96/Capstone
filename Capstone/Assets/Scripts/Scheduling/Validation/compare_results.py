## @file compare_results.py
#  @brief Cross-validates C# DES output against the job_shop_lib Python reference (v2).
#
#  @details Compares makespan CSVs and full operation schedules produced by the C#
#  @c ValidationExporter against the job_shop_lib reference generator. Designed to be
#  robust against common data quality issues: duplicate rows from re-runs, missing rules
#  on either side, and large-scale mismatch sets requiring triage.
#
#  @par Prerequisites
#  -# Run the C# @c ValidationExporter to produce @c csharp_makespans.csv and
#     @c csharp_schedules.json.
#  -# Run the Python reference generator to produce @c reference_makespans.csv and
#     @c reference_schedules.json.
#  -# Run this script from the directory containing all four files:
#     @code
#     python compare_results.py
#     @endcode
#
#  @par Output
#  - Coverage report: rows and instances found on each side, and any asymmetric pairs.
#  - Overall exact-match / mismatch counts.
#  - Per-rule summary showing match rate, average absolute delta, and maximum delta.
#  - Instance × rule comparison table showing @c OK or a signed integer delta.
#  - Top-10 worst mismatches by absolute delta, each with the first divergent operation.
#  - Diagnosis hints inferring the likely bug class from the sign distribution of deltas
#    and FCFS match rate.
#  - Full merged DataFrame saved to @c comparison_results.csv.
#
#  @par Exit codes
#  - @c 0 — all comparable pairs matched exactly.
#  - @c 1 — one or more mismatches found, or no common pairs could be compared.

import json
import pandas as pd
import sys
import os
from pathlib import Path

data_dir = Path("../../../Resources/Validation")


def load_data():
    ## @brief Loads and deduplicates both makespan CSVs and, if present, both schedule JSON files.
    #
    #  @details Reads @c reference_makespans.csv and @c csharp_makespans.csv into DataFrames,
    #  deduplicating each on @c (instance, rule) keeping the last occurrence to handle
    #  rows accumulated from multiple exporter runs. Schedule JSON files are loaded only
    #  if they exist on disk; missing files result in empty dicts, which downstream
    #  divergence analysis silently skips.
    #
    #  @returns A four-tuple @c (ref, csharp, ref_schedules, cs_schedules) where:
    #  - @c ref — deduplicated reference makespan DataFrame.
    #  - @c csharp — deduplicated C# makespan DataFrame.
    #  - @c ref_schedules — dict keyed by @c "{instance}_{rule}" from @c reference_schedules.json,
    #    or an empty dict if the file is absent.
    #  - @c cs_schedules — dict keyed by @c "{instance}_{rule}" from @c csharp_schedules.json,
    #    or an empty dict if the file is absent.

    ref = pd.read_csv(data_dir / "reference_makespans.csv")
    csharp = pd.read_csv(data_dir / "csharp_makespans.csv")

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
    ## @brief Identifies all operations where the reference and C# schedules disagree.
    #
    #  @details Indexes both operation lists by @c (job, op_index) and walks the union
    #  of keys in sorted order. An operation is flagged as divergent if it is absent
    #  from one side, or if its @c start time or @c machine assignment differs between
    #  the two schedules. Results are sorted by @c ref_start ascending so that the
    #  earliest divergence — most likely the root cause — appears first.
    #
    #  @param ref_ops List of operation dicts from @c reference_schedules.json for one
    #  @c "{instance}_{rule}" key. Each dict must contain @c job, @c op_index, @c start,
    #  @c machine, @c end, and @c duration fields.
    #  @param cs_ops List of operation dicts from @c csharp_schedules.json for the same
    #  @c "{instance}_{rule}" key, with identical field structure.
    #
    #  @returns A list of divergence dicts sorted by @c ref_start ascending. Each dict
    #  contains @c job, @c op_index, and @c issue. Timing/assignment divergences also
    #  include @c ref_start, @c cs_start, @c ref_machine, @c cs_machine, @c ref_end,
    #  @c cs_end, and @c duration. Missing-operation entries include only @c issue.

    ref_by_key = {(op["job"], op["op_index"]): op for op in ref_ops}
    cs_by_key = {(op["job"], op["op_index"]): op for op in cs_ops}
    all_keys = sorted(set(ref_by_key.keys()) | set(cs_by_key.keys()))

    divergences = []
    for key in all_keys:
        r = ref_by_key.get(key)
        c = cs_by_key.get(key)
        if r is None or c is None:
            divergences.append(
                {
                    "job": key[0],
                    "op_index": key[1],
                    "issue": "missing in " + ("C#" if c is None else "reference"),
                }
            )
            continue
        if r["start"] != c["start"] or r["machine"] != c["machine"]:
            divergences.append(
                {
                    "job": key[0],
                    "op_index": key[1],
                    "ref_start": r["start"],
                    "cs_start": c["start"],
                    "ref_machine": r["machine"],
                    "cs_machine": c["machine"],
                    "ref_end": r["end"],
                    "cs_end": c["end"],
                    "duration": r["duration"],
                    "issue": "timing/assignment differs",
                }
            )
    divergences.sort(key=lambda d: d.get("ref_start", 0))
    return divergences


def main():
    ## @brief Entry point — runs the full cross-validation pipeline and prints a report.
    #
    #  @details Executes the following steps in order:
    #  -# Loads and deduplicates both CSVs and both schedule JSON files via @ref load_data.
    #  -# Computes coverage: common @c (instance, rule) pairs, pairs only in reference,
    #     and pairs only in C#. Prints up to 10 examples of asymmetric pairs.
    #     Exits with code @c 1 if no common pairs exist, with diagnostic rule/instance lists.
    #  -# Merges on @c (instance, rule) with suffixes @c _ref and @c _cs, then adds
    #     boolean @c match, signed integer @c delta, and percentage @c delta_pct columns.
    #  -# Prints overall match/mismatch counts, a per-rule summary (match rate, average
    #     and maximum absolute delta), and an instance × rule table.
    #  -# For the top 10 worst mismatches by absolute delta, calls @ref find_first_divergence
    #     and reports the earliest divergent operation along with the total divergent op count.
    #  -# Prints diagnosis hints by analysing the sign distribution of deltas and the FCFS
    #     match rate. A perfect FCFS match with other-rule failures isolates the bug to
    #     dispatching rule logic; FCFS failures point to base event-ordering bugs.
    #  -# Saves the full merged DataFrame to @c comparison_results.csv.
    #
    #  @returns @c 0 if all comparable pairs matched exactly, @c 1 otherwise.

    ref, csharp, ref_schedules, cs_schedules = load_data()

    print(
        f"Reference rows:  {len(ref)} ({ref['instance'].nunique()} instances, {ref['rule'].nunique()} rules)"
    )
    print(
        f"C# rows:         {len(csharp)} ({csharp['instance'].nunique()} instances, {csharp['rule'].nunique()} rules)"
    )
    print()

    # Find common (instance, rule) pairs and diagnose any asymmetry
    ref_keys = set(zip(ref["instance"], ref["rule"]))
    cs_keys = set(zip(csharp["instance"], csharp["rule"]))
    common = ref_keys & cs_keys
    only_ref = ref_keys - cs_keys
    only_cs = cs_keys - ref_keys

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
        print(
            f"  Reference instances (first 5): {sorted(ref['instance'].unique())[:5]}"
        )
        print(
            f"  C# instances (first 5):        {sorted(csharp['instance'].unique())[:5]}"
        )
        return 1

    # Merge and compute match/delta columns
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

    # Per-rule summary
    print("PER-RULE SUMMARY")
    print("-" * 70)
    for rule in sorted(merged["rule"].unique()):
        rule_data = merged[merged["rule"] == rule]
        rule_matches = rule_data["match"].sum()
        rule_total = len(rule_data)
        avg_delta = rule_data["delta_pct"].abs().mean()
        print(
            f"  {rule:6s}  {rule_matches}/{rule_total} exact  "
            f"avg |delta|={avg_delta:.1f}%  "
            f"max delta={rule_data['delta_pct'].max():+.1f}%"
        )
    print()

    # Instance x rule table
    print("INSTANCE x RULE COMPARISON (showing delta)")
    print("-" * 70)

    instances = sorted(merged["instance"].unique())
    rules = sorted(merged["rule"].unique())

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

    # Top-10 worst mismatches with first divergent operation
    if mismatches > 0:
        print("=" * 70)
        print("TOP 10 WORST MISMATCHES (by absolute delta)")
        print("=" * 70)

        worst = (
            merged[~merged["match"]]
            .sort_values("delta", key=abs, ascending=False)
            .head(10)
        )
        for _, row in worst.iterrows():
            inst = row["instance"]
            rule = row["rule"]
            print(f"\n  {inst} / {rule}:")
            print(
                f"    Reference: {int(row['makespan_ref'])}    C#: {int(row['makespan_cs'])}    "
                f"Delta: {int(row['delta']):+d} ({row['delta_pct']:+.1f}%)"
            )

            ## Schedule dicts are keyed as "{instance}_{rule}", matching the format
            #  written by ValidationExporter.RunAndExport and mirror_des.py.
            sched_key = f"{inst}_{rule}"
            ref_ops = ref_schedules.get(sched_key, [])
            cs_ops = cs_schedules.get(sched_key, [])

            if ref_ops and cs_ops:
                divergences = find_first_divergence(ref_ops, cs_ops)
                if divergences:
                    d = divergences[0]
                    print(f"    First divergence: Job {d['job']}, Op {d['op_index']}")
                    if "ref_start" in d:
                        print(
                            f"      Ref:  start={d['ref_start']}, machine={d['ref_machine']}, end={d['ref_end']}"
                        )
                        print(
                            f"      C#:   start={d['cs_start']}, machine={d['cs_machine']}, end={d['cs_end']}"
                        )
                    print(
                        f"    Total divergent ops: {len(divergences)} / {len(ref_ops)}"
                    )

        # Diagnosis hints based on delta sign distribution and FCFS match rate
        print("\n" + "=" * 70)
        print("DIAGNOSIS HINTS")
        print("=" * 70)

        deltas = merged[~merged["match"]]["delta"]
        all_positive = (deltas > 0).all()
        all_negative = (deltas < 0).all()

        ## Delta sign distribution narrows the bug class:
        #  - All positive  → C# is always slower; likely a dispatching ordering difference.
        #  - All negative  → C# is always faster; possible precedence constraint being skipped.
        #  - Mixed signs   → Typical of tiebreaking differences rather than a systematic bug.
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

        ## FCFS match rate is the most reliable base-layer diagnostic:
        #  FCFS applies no sorting, so its correctness depends only on event-ordering
        #  logic. A perfect FCFS match alongside other-rule failures isolates the bug
        #  to dispatching rule implementation. Any FCFS failure points to the core
        #  DES event queue or machine state logic and should be fixed first.
        fcfs_data = (
            merged[merged["rule"] == "FCFS"]
            if "FCFS" in merged["rule"].values
            else pd.DataFrame()
        )
        if not fcfs_data.empty:
            fcfs_matches = fcfs_data["match"].sum()
            print(f"\n  FCFS matches: {fcfs_matches}/{len(fcfs_data)}")
            if fcfs_matches == len(fcfs_data):
                print("  FCFS is exact -> base DES event logic is correct.")
                print(
                    "  Mismatches in other rules -> dispatching rule implementation differs."
                )
            else:
                print("  FCFS has mismatches -> base DES event ordering may differ.")
                print("  Fix FCFS first before debugging other rules.")

    merged.to_csv(data_dir / "comparison_results.csv", index=False)
    print(f"\nFull comparison saved to comparison_results.csv")

    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
