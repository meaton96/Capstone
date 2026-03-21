"""
Reference Makespan Generator
=============================
Generates ground-truth makespans from job_shop_lib for cross-validation
against the C# DES simulator.

Outputs two files:
  1. reference_makespans.csv — (instance, rule, makespan) for comparison
  2. reference_schedules.json — full operation-level schedule for debugging
     mismatches (which op started when on which machine)

Usage:
    pip install job-shop-lib pandas
    python generate_reference.py

The C# simulator should produce identical makespans for matching rules.
If not, the schedule JSON lets you find exactly where the two diverge.
"""

import json
import time
import pandas as pd
from job_shop_lib.benchmarking import load_benchmark_instance
from job_shop_lib.dispatching.rules import DispatchingRuleSolver

# ── Config ────────────────────────────────────────────────────────────────────

# Start small for debugging, then expand
INSTANCES = ["ta01", "ta11", "ta21", "ta31"]

# These are the rules we can map 1:1 between job_shop_lib and C#
RULES = [
    "shortest_processing_time",    # → C# SPT_SMPT
    "largest_processing_time",     # → C# LPT_MMUR / LPT_SMPT
    "most_work_remaining",         # → C# LRT_MMUR
    "first_come_first_served",     # → C# SDT_SRWT (FIFO fallback)
    "most_operations_remaining",   # → no C# equivalent yet
]

# Rule name mapping for the CSV (so both sides use the same key)
RULE_MAP = {
    "shortest_processing_time":    "SPT",
    "largest_processing_time":     "LPT",
    "most_work_remaining":         "MWR",
    "first_come_first_served":     "FCFS",
    "most_operations_remaining":   "MOR",
}


# ── Generate ──────────────────────────────────────────────────────────────────

records = []
schedules = {}

for instance_name in INSTANCES:
    print(f"\n{'='*60}")
    print(f"Instance: {instance_name}")
    print(f"{'='*60}")

    instance = load_benchmark_instance(instance_name)
    metadata = instance.metadata if hasattr(instance, "metadata") else {}
    optimum = metadata.get("optimum")

    for rule_name in RULES:
        solver = DispatchingRuleSolver(rule_name, ready_operations_filter=None)
        start = time.perf_counter()
        schedule = solver(instance)
        elapsed = time.perf_counter() - start
        makespan = schedule.makespan()

        gap = ((makespan - optimum) / optimum * 100) if optimum else None
        rule_key = RULE_MAP[rule_name]

        print(f"  {rule_key:6s} → makespan={makespan:6d}  "
              f"optimum={optimum}  gap={gap:.1f}%  ({elapsed:.4f}s)")

        records.append({
            "instance":   instance_name,
            "rule":       rule_key,
            "rule_full":  rule_name,
            "makespan":   makespan,
            "optimum":    optimum,
            "gap_pct":    round(gap, 4) if gap else None,
            "num_jobs":   instance.num_jobs,
            "num_machines": instance.num_machines,
        })

        # ── Extract full operation-level schedule for debugging ───────────
        # schedule.schedule is a list of lists: schedule[machine_id] = 
        #   list of ScheduledOperation
        instance_schedule_key = f"{instance_name}_{rule_key}"
        op_list = []

        for machine_id, machine_ops in enumerate(schedule.schedule):
            for sched_op in machine_ops:
                op = sched_op.operation
                job_id = sched_op.job_id
                op_list.append({
                    "job":       job_id,
                    "op_index":  op.position_in_job,
                    "machine":   machine_id,
                    "start":     sched_op.start_time,
                    "end":       sched_op.end_time,
                    "duration":  op.duration,
                })

        # Sort by start time for easy comparison
        op_list.sort(key=lambda x: (x["start"], x["machine"], x["job"]))
        schedules[instance_schedule_key] = op_list


# ── Save CSV ──────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv("reference_makespans.csv", index=False)
print(f"\nSaved {len(df)} rows to reference_makespans.csv")

# ── Save detailed schedules ──────────────────────────────────────────────────
with open("reference_schedules.json", "w") as f:
    json.dump(schedules, f, indent=2)
print(f"Saved {len(schedules)} schedules to reference_schedules.json")

# ── Quick summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("REFERENCE SUMMARY")
print(f"{'='*60}")
print(df.pivot(index="instance", columns="rule", values="makespan").to_string())
