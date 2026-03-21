## @file generate_reference.py
#  @brief Generates ground-truth makespans and operation schedules from job_shop_lib
#  for cross-validation against the C# DES simulator.
#
#  @details Runs every rule in @ref RULES against every instance in @ref INSTANCES
#  using job_shop_lib's @c DispatchingRuleSolver, then writes two output files whose
#  structure is kept deliberately in sync with the C# @c ValidationExporter outputs:
#  - @c reference_makespans.csv — one row per instance/rule combination, with columns
#    matching @c csharp_makespans.csv for direct DataFrame merging in @c compare_results.py.
#  - @c reference_schedules.json — full operation-level schedule per instance/rule,
#    keyed as @c "{instance}_{rule_key}", for divergence analysis in @c compare_results.py
#    and @c compare_mirror.py.
#
#  @par Prerequisites
#  @code
#  pip install job-shop-lib pandas
#  @endcode
#
#  @par Usage
#  @code
#  python generate_reference.py
#  @endcode
#
#  @par Output
#  - @c reference_makespans.csv — ground-truth makespans for all instance/rule pairs.
#  - @c reference_schedules.json — per-instance/rule operation lists sorted by start time,
#    then machine ID, then job ID.
#  - A pivot table printed to stdout summarising makespan by instance × rule.
#
#  @note The C# simulator should produce identical makespans for all mapped rules.
#  If not, the schedule JSON lets @c compare_results.py identify exactly which operation
#  first diverges between the two implementations.

import json
import time
import pandas as pd
from job_shop_lib.benchmarking import load_benchmark_instance
from job_shop_lib.dispatching.rules import DispatchingRuleSolver

## @brief Taillard benchmark instance names to process.
#
#  @details Kept small by default for fast iteration during debugging.
#  Expand to the full Taillard set (e.g. @c ta01 – @c ta80) once
#  the C# and Python outputs are confirmed to match on this subset.
INSTANCES = ["ta01", "ta11", "ta21", "ta31"]

## @brief Fully qualified job_shop_lib rule names to simulate.
#
#  @details Only rules that have a confirmed 1:1 mapping to a C# @c DispatchingRule
#  are included. The inline comments document the corresponding C# enum value for
#  each rule. Rules without a C# equivalent (e.g. @c most_operations_remaining)
#  can be included here for reference but will produce unmatched rows in the comparison.
RULES = [
    "shortest_processing_time",  ##< Maps to C# DispatchingRule.SPT_SMPT.
    "largest_processing_time",  ##< Maps to C# DispatchingRule.LPT_MMUR / LPT_SMPT.
    "most_work_remaining",  ##< Maps to C# DispatchingRule.LRT_MMUR.
    "first_come_first_served",  ##< Maps to C# DispatchingRule.SDT_SRWT (FIFO fallback).
    "most_operations_remaining",  ##< Maps to C# DispatchingRule.MOR.
]

## @brief Maps job_shop_lib full rule names to the short keys used in both CSV outputs.
#
#  @details Short keys must match the @c key field in @c ValidationExporter.RuleMap
#  exactly so that @c compare_results.py can merge the two DataFrames on the @c rule
#  column without any name translation. Adding a rule here without a corresponding
#  entry in @c ValidationExporter.RuleMap will produce unmatched rows in the comparison.
RULE_MAP = {
    "shortest_processing_time": "SPT",
    "largest_processing_time": "LPT",
    "most_work_remaining": "MWR",
    "first_come_first_served": "FCFS",
    "most_operations_remaining": "MOR",
}


## @brief Accumulates one record dict per instance/rule combination for CSV output.
#  @details Populated during the main simulation loop and converted to a DataFrame
#  at the end. Each entry matches the column schema of @c csharp_makespans.csv.
records = []

## @brief Accumulates full operation schedules keyed by @c "{instance}_{rule_key}".
#  @details Populated during the main simulation loop and serialized to
#  @c reference_schedules.json. Key format must match @c ValidationExporter.RunAndExport
#  and @c mirror_des.py so that @c compare_results.py can look up schedules by the
#  same key used in the merged DataFrame.
schedules = {}

for instance_name in INSTANCES:
    print(f"\n{'='*60}")
    print(f"Instance: {instance_name}")
    print(f"{'='*60}")

    instance = load_benchmark_instance(instance_name)

    ## @brief Extract the optimum from instance metadata if available.
    #  @details job_shop_lib exposes metadata as a dict on some instance types.
    #  The @c hasattr guard handles instances that do not carry metadata, in which
    #  case @c optimum is @c None and gap percentage is skipped for that instance.
    metadata = instance.metadata if hasattr(instance, "metadata") else {}
    optimum = metadata.get("optimum")

    for rule_name in RULES:
        solver = DispatchingRuleSolver(rule_name)
        start = time.perf_counter()
        schedule = solver(instance)
        elapsed = time.perf_counter() - start
        makespan = schedule.makespan()

        gap = ((makespan - optimum) / optimum * 100) if optimum else None
        rule_key = RULE_MAP[rule_name]

        print(
            f"  {rule_key:6s} → makespan={makespan:6d}  "
            f"optimum={optimum}  gap={gap:.1f}%  ({elapsed:.4f}s)"
        )

        records.append(
            {
                "instance": instance_name,
                "rule": rule_key,
                "rule_full": rule_name,
                "makespan": makespan,
                "optimum": optimum,
                "gap_pct": round(gap, 4) if gap else None,
                "num_jobs": instance.num_jobs,
                "num_machines": instance.num_machines,
            }
        )

        ## @brief Extract the full operation-level schedule for this instance/rule pair.
        #
        #  @details @c schedule.schedule is a list of lists indexed by machine ID.
        #  Each inner list contains @c ScheduledOperation objects whose @c .operation
        #  field exposes @c position_in_job and @c duration. The extracted dicts are
        #  sorted by start time, then machine ID, then job ID — matching the sort order
        #  used by @c ValidationExporter.RunAndExport — so that @c compare_results.py
        #  can diff the two schedule JSONs entry-by-entry without re-sorting.
        instance_schedule_key = f"{instance_name}_{rule_key}"
        op_list = []

        for machine_id, machine_ops in enumerate(schedule.schedule):
            for sched_op in machine_ops:
                op = sched_op.operation
                job_id = sched_op.job_id
                op_list.append(
                    {
                        "job": job_id,
                        "op_index": op.position_in_job,
                        "machine": machine_id,
                        "start": sched_op.start_time,
                        "end": sched_op.end_time,
                        "duration": op.duration,
                    }
                )

        op_list.sort(key=lambda x: (x["start"], x["machine"], x["job"]))
        schedules[instance_schedule_key] = op_list


# Save CSV
df = pd.DataFrame(records)
df.to_csv("reference_makespans.csv", index=False)
print(f"\nSaved {len(df)} rows to reference_makespans.csv")

# Save detailed schedules
with open("reference_schedules.json", "w") as f:
    json.dump(schedules, f, indent=2)
print(f"Saved {len(schedules)} schedules to reference_schedules.json")

# Print pivot summary
print(f"\n{'='*60}")
print("REFERENCE SUMMARY")
print(f"{'='*60}")
print(df.pivot(index="instance", columns="rule", values="makespan").to_string())
