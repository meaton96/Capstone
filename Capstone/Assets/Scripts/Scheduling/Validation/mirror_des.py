"""
Mirror DES — Python implementation matching the C# DESSimulator exactly.
========================================================================

Architecture (matches C# DESSimulator.cs):
  - Event-driven with a priority queue sorted by (time, sequence_number)
  - Per-machine waiting queues
  - When a job arrives at a machine:
      - If machine is idle → start immediately
      - If machine is busy → add to machine's waiting queue
  - When a machine finishes an operation:
      - Advance the job to its next operation (dispatch to next machine)
      - Pick next op from this machine's waiting queue via dispatching rule
  - Tiebreaker: lowest job_id (matching C# .ThenBy(op => op.JobId))

This should produce IDENTICAL makespans to the C# DES.
Any delta means a real bug in one of the two implementations.

Usage:
    python mirror_des.py                          # runs all instances in current dir
    python mirror_des.py --instance ta01.json     # runs a single instance
"""

import json
import heapq
import os
import argparse
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable


# ── Data Structures ───────────────────────────────────────────────────────────

class EventType(Enum):
    JOB_ARRIVED = auto()
    OPERATION_COMPLETE = auto()


@dataclass(order=True)
class SimEvent:
    time: float
    sequence: int
    event_type: EventType = field(compare=False)
    job_id: int = field(compare=False)
    machine_id: int = field(compare=False)


@dataclass
class Operation:
    job_id: int
    op_index: int
    machine_id: int
    duration: int
    start_time: float = -1.0
    end_time: float = -1.0


@dataclass
class Job:
    job_id: int
    operations: list[Operation]
    next_op_index: int = 0
    completion_time: float = -1.0

    @property
    def is_complete(self) -> bool:
        return self.next_op_index >= len(self.operations)

    @property
    def current_operation(self) -> Optional[Operation]:
        if self.is_complete:
            return None
        return self.operations[self.next_op_index]


class MachineState(Enum):
    IDLE = auto()
    BUSY = auto()


@dataclass
class Machine:
    machine_id: int
    state: MachineState = MachineState.IDLE
    current_op: Optional[Operation] = None
    waiting_queue: list[Operation] = field(default_factory=list)

    def start_processing(self, op: Operation, current_time: float):
        self.state = MachineState.BUSY
        self.current_op = op
        op.start_time = current_time
        op.end_time = current_time + op.duration

    def finish_processing(self):
        self.state = MachineState.IDLE
        self.current_op = None


# ── Dispatching Rules ─────────────────────────────────────────────────────────
# Each rule: (waiting_ops, all_jobs) → selected Operation
# Matches C# DispatchingRules.cs with .ThenBy(op => op.JobId) tiebreaker

def remaining_work(op: Operation, jobs: list[Job]) -> int:
    job = jobs[op.job_id]
    return sum(o.duration for o in job.operations[op.op_index:])


def remaining_ops(op: Operation, jobs: list[Job]) -> int:
    job = jobs[op.job_id]
    return len(job.operations) - op.op_index


def rule_spt(waiting: list[Operation], jobs: list[Job]) -> Operation:
    """Shortest Processing Time (tiebreak: lowest job_id)"""
    return min(waiting, key=lambda op: (op.duration, op.job_id))


def rule_lpt(waiting: list[Operation], jobs: list[Job]) -> Operation:
    """Longest Processing Time (tiebreak: lowest job_id)"""
    return min(waiting, key=lambda op: (-op.duration, op.job_id))


def rule_mwr(waiting: list[Operation], jobs: list[Job]) -> Operation:
    """Most Work Remaining = Longest Remaining Time (tiebreak: lowest job_id)"""
    return min(waiting, key=lambda op: (-remaining_work(op, jobs), op.job_id))


def rule_mor(waiting: list[Operation], jobs: list[Job]) -> Operation:
    """Most Operations Remaining (tiebreak: lowest job_id)"""
    return min(waiting, key=lambda op: (-remaining_ops(op, jobs), op.job_id))


def rule_fcfs(waiting: list[Operation], jobs: list[Job]) -> Operation:
    """First Come First Served — return first in queue (insertion order)"""
    return waiting[0]


RULES: dict[str, Callable] = {
    "SPT": rule_spt,
    "LPT": rule_lpt,
    "MWR": rule_mwr,
    "MOR": rule_mor,
    "FCFS": rule_fcfs,
}

RULE_FULL_NAMES = {
    "SPT": "shortest_processing_time",
    "LPT": "largest_processing_time",
    "MWR": "most_work_remaining",
    "MOR": "most_operations_remaining",
    "FCFS": "first_come_first_served",
}


# ── DES Simulator ─────────────────────────────────────────────────────────────

class DESSimulator:
    """
    Event-driven DES with per-machine queues.
    Mirrors C# DESSimulator exactly.
    """

    def __init__(self):
        self.jobs: list[Job] = []
        self.machines: list[Machine] = []
        self.event_queue: list[SimEvent] = []  # min-heap
        self.sequence: int = 0
        self.current_time: float = 0.0
        self.rule_fn: Callable = rule_spt

    def load_instance(self, data: dict):
        """Load from the Taillard JSON format."""
        duration_matrix = data["duration_matrix"]
        machines_matrix = data["machines_matrix"]
        num_jobs = len(duration_matrix)
        num_machines = len(duration_matrix[0])

        self.machines = [Machine(m) for m in range(num_machines)]
        self.jobs = []

        for j in range(num_jobs):
            ops = []
            for o in range(len(duration_matrix[j])):
                ops.append(Operation(
                    job_id=j,
                    op_index=o,
                    machine_id=machines_matrix[j][o],
                    duration=duration_matrix[j][o],
                ))
            self.jobs.append(Job(job_id=j, operations=ops))

    def reset(self):
        """Reset state and enqueue all job arrivals at t=0."""
        self.current_time = 0.0
        self.event_queue = []
        self.sequence = 0

        for m in self.machines:
            m.state = MachineState.IDLE
            m.current_op = None
            m.waiting_queue = []

        for job in self.jobs:
            job.next_op_index = 0
            job.completion_time = -1.0
            for op in job.operations:
                op.start_time = -1.0
                op.end_time = -1.0

        # Enqueue all job arrivals at t=0
        for j in range(len(self.jobs)):
            self._enqueue(0.0, EventType.JOB_ARRIVED, job_id=j, machine_id=-1)

    def _enqueue(self, time: float, event_type: EventType, job_id: int, machine_id: int):
        evt = SimEvent(time, self.sequence, event_type, job_id, machine_id)
        self.sequence += 1
        heapq.heappush(self.event_queue, evt)

    def run(self, rule_fn: Callable) -> float:
        """Run simulation to completion, return makespan."""
        self.rule_fn = rule_fn
        self.reset()

        while self.event_queue:
            evt = heapq.heappop(self.event_queue)
            self.current_time = evt.time

            if evt.event_type == EventType.JOB_ARRIVED:
                self._handle_job_arrived(evt)
            elif evt.event_type == EventType.OPERATION_COMPLETE:
                self._handle_operation_complete(evt)

        makespan = max(j.completion_time for j in self.jobs)
        return makespan

    def _handle_job_arrived(self, evt: SimEvent):
        job = self.jobs[evt.job_id]
        if job.is_complete:
            return
        self._try_dispatch_job(job)

    def _handle_operation_complete(self, evt: SimEvent):
        machine = self.machines[evt.machine_id]
        job = self.jobs[evt.job_id]

        # Finish processing
        machine.finish_processing()

        # Advance job
        job.next_op_index += 1

        if job.is_complete:
            job.completion_time = self.current_time
        else:
            # Send to next machine (zero transit time)
            self._try_dispatch_job(job)

        # Check waiting queue on this now-free machine
        self._try_start_next_on_machine(machine)

    def _try_dispatch_job(self, job: Job):
        if job.is_complete:
            return

        op = job.current_operation
        machine = self.machines[op.machine_id]

        if machine.state == MachineState.IDLE:
            machine.start_processing(op, self.current_time)
            self._enqueue(op.end_time, EventType.OPERATION_COMPLETE,
                          job_id=job.job_id, machine_id=machine.machine_id)
        else:
            machine.waiting_queue.append(op)

    def _try_start_next_on_machine(self, machine: Machine):
        if not machine.waiting_queue:
            return
        if machine.state != MachineState.IDLE:
            return

        next_op = self.rule_fn(machine.waiting_queue, self.jobs)
        machine.waiting_queue.remove(next_op)

        machine.start_processing(next_op, self.current_time)
        self._enqueue(next_op.end_time, EventType.OPERATION_COMPLETE,
                      job_id=next_op.job_id, machine_id=machine.machine_id)

    def get_schedule(self) -> list[dict]:
        """Extract full schedule for comparison."""
        ops = []
        for job in self.jobs:
            for op in job.operations:
                ops.append({
                    "job": op.job_id,
                    "op_index": op.op_index,
                    "machine": op.machine_id,
                    "start": op.start_time,
                    "end": op.end_time,
                    "duration": op.duration,
                })
        ops.sort(key=lambda x: (x["start"], x["machine"], x["job"]))
        return ops


# ── Main ──────────────────────────────────────────────────────────────────────

def find_json_files(directory: str) -> list[str]:
    files = []
    for f in sorted(os.listdir(directory)):
        if f.endswith(".json") and f.startswith("ta"):
            files.append(os.path.join(directory, f))
    return files


def main():
    parser = argparse.ArgumentParser(description="Mirror DES validator")
    parser.add_argument("--instance", type=str, default=None,
                        help="Path to a single JSON instance")
    parser.add_argument("--dir", type=str, default=".",
                        help="Directory containing JSON instances")
    parser.add_argument("--rules", type=str, default="SPT,LPT,MWR,MOR,FCFS",
                        help="Comma-separated rule keys")
    parser.add_argument("--output-csv", type=str, default="mirror_des_makespans.csv",
                        help="Output CSV path")
    parser.add_argument("--output-json", type=str, default="mirror_des_schedules.json",
                        help="Output schedules JSON path")
    args = parser.parse_args()

    rule_keys = [r.strip() for r in args.rules.split(",")]

    if args.instance:
        json_files = [args.instance]
    else:
        json_files = find_json_files(args.dir)

    if not json_files:
        print(f"No JSON files found in {args.dir}")
        return

    sim = DESSimulator()
    csv_lines = ["instance,rule,rule_full,makespan,optimum,gap_pct,num_jobs,num_machines"]
    all_schedules = {}

    for filepath in json_files:
        with open(filepath) as f:
            data = json.load(f)

        name = data.get("name", os.path.splitext(os.path.basename(filepath))[0])
        metadata = data.get("metadata", {})
        optimum = metadata.get("optimum")

        sim.load_instance(data)

        print(f"\n{'=' * 60}")
        print(f"Instance: {name} ({sim.jobs.__len__()} jobs x {sim.machines.__len__()} machines)")
        print(f"{'=' * 60}")

        for rule_key in rule_keys:
            if rule_key not in RULES:
                print(f"  [SKIP] Unknown rule: {rule_key}")
                continue

            rule_fn = RULES[rule_key]
            makespan = sim.run(rule_fn)

            gap = None
            if optimum and optimum > 0:
                gap = round((makespan - optimum) / optimum * 100, 4)

            print(f"  {rule_key:6s} -> makespan={makespan:8.0f}  "
                  f"optimum={optimum}  gap={gap}%")

            csv_lines.append(",".join(str(x) for x in [
                name, rule_key, RULE_FULL_NAMES.get(rule_key, rule_key),
                int(makespan),
                optimum if optimum else "",
                gap if gap is not None else "",
                len(sim.jobs), len(sim.machines),
            ]))

            sched_key = f"{name}_{rule_key}"
            all_schedules[sched_key] = sim.get_schedule()

    # Save CSV
    with open(args.output_csv, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"\nSaved {len(csv_lines) - 1} rows to {args.output_csv}")

    # Save schedules
    with open(args.output_json, "w") as f:
        json.dump(all_schedules, f, indent=2)
    print(f"Saved {len(all_schedules)} schedules to {args.output_json}")


if __name__ == "__main__":
    main()