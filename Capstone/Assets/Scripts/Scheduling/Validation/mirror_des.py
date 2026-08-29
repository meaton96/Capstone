## @file mirror_des.py
#  @brief Python DES implementation mirroring the C# @c DESSimulator exactly.
#
#  @details Implements the same event-driven architecture as @c DESSimulator.cs so that
#  any makespan delta between the two is a definitive indicator of a real bug rather than
#  an expected implementation divergence:
#  - Priority queue sorted by @c (time, sequence_number), matching @ref SimEvent.CompareTo.
#  - Per-machine waiting queues with the same idle/busy state machine as @ref Machine.
#  - Identical dispatch logic: immediate start if idle, queue append if busy.
#  - Tiebreaker on all rules: lowest @c job_id, matching C# @c .ThenBy(op => op.JobId).
#
#  @par Usage
#  @code
#  python mirror_des.py                        # runs all ta*.json files in current dir
#  python mirror_des.py --instance ta01.json   # runs a single instance
#  python mirror_des.py --rules SPT,FCFS       # runs a subset of rules
#  @endcode
#
#  @par Output
#  - @c mirror_des_makespans.csv — one row per instance/rule, column schema matching
#    @c csharp_makespans.csv and @c reference_makespans.csv for direct merging.
#  - @c mirror_des_schedules.json — full operation schedules keyed as
#    @c "{instance}_{rule_key}", matching the format written by @c ValidationExporter.RunAndExport.
#
#  @par Exit codes
#  This script does not currently return a non-zero exit code on mismatch.
#  Use @c compare_mirror.py to diff its output against the C# results.

import json
import heapq
import os
import argparse
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

# ── Data Structures ───────────────────────────────────────────────────────────


## @brief Discriminated union of simulation event types, mirroring C# @c EventType.
class EventType(Enum):
    JOB_ARRIVED = auto()  ##< A job has arrived at its next required machine.
    OPERATION_COMPLETE = auto()  ##< An operation running on a machine has finished.


## @brief Immutable simulation event with (time, sequence) ordering.
#
#  @details @c @dataclass(order=True) generates @c __lt__ from the field declaration
#  order, so @c time is the primary sort key and @c sequence is the FIFO tiebreaker —
#  matching the @c CompareTo logic in the C# @ref SimEvent class. Fields after
#  @c sequence are marked @c field(compare=False) to exclude them from ordering.
@dataclass(order=True)
class SimEvent:
    ## @brief Simulation time at which this event fires.
    time: float
    ## @brief Monotonically increasing counter used as a FIFO tiebreaker for equal times.
    sequence: int
    ## @brief Category of this event; excluded from ordering comparisons.
    event_type: EventType = field(compare=False)
    ## @brief ID of the associated job, or @c -1 if not applicable.
    job_id: int = field(compare=False)
    ## @brief ID of the associated machine, or @c -1 if not applicable.
    machine_id: int = field(compare=False)


## @brief Single operation within a job, mirroring the C# @ref Operation class.
#
#  @details @c start_time and @c end_time are initialised to @c -1.0 and stamped
#  by @ref Machine.start_processing during the simulation run, matching the C#
#  behaviour where @c -1 sentinel values indicate an unscheduled operation.
@dataclass
class Operation:
    ## @brief ID of the owning job.
    job_id: int
    ## @brief Zero-based index of this operation within its job's sequence.
    op_index: int
    ## @brief ID of the machine that must process this operation.
    machine_id: int
    ## @brief Processing time in simulation time units.
    duration: int
    ## @brief Simulation time at which processing began. @c -1.0 until scheduled.
    start_time: float = -1.0
    ## @brief Simulation time at which processing finished. @c -1.0 until scheduled.
    end_time: float = -1.0


## @brief Ordered sequence of operations representing a single job, mirroring C# @ref Job.
#
#  @details Tracks execution progress via @c next_op_index, advanced by
#  @ref DESSimulator._handle_operation_complete after each operation completes.
#  @c is_complete and @c current_operation are derived properties with no side effects.
@dataclass
class Job:
    ## @brief Unique job identifier matching its index in @ref DESSimulator.jobs.
    job_id: int
    ## @brief Ordered list of operations to execute sequentially.
    operations: list[Operation]
    ## @brief Index of the next operation awaiting dispatch. Starts at @c 0.
    next_op_index: int = 0
    ## @brief Simulation time at which the final operation completed. @c -1.0 until done.
    completion_time: float = -1.0

    @property
    def is_complete(self) -> bool:
        ## @brief @c True when all operations have been processed.
        return self.next_op_index >= len(self.operations)

    @property
    def current_operation(self) -> Optional[Operation]:
        ## @brief The next operation awaiting dispatch, or @c None if the job is complete.
        if self.is_complete:
            return None
        return self.operations[self.next_op_index]


## @brief Operational states for a machine, mirroring the C# @ref MachineState enum.
class MachineState(Enum):
    IDLE = auto()  ##< Machine is free and ready to accept a new operation.
    BUSY = auto()  ##< Machine is currently processing an operation.


## @brief Single shop-floor machine with a waiting queue, mirroring C# @ref Machine.
#
#  @details State transitions are driven exclusively by @ref DESSimulator via
#  @ref start_processing and @ref finish_processing, matching the C# design.
@dataclass
class Machine:
    ## @brief Unique machine identifier matching its index in @ref DESSimulator.machines.
    machine_id: int
    ## @brief Current operational state. Starts @c IDLE.
    state: MachineState = MachineState.IDLE
    ## @brief The operation currently being processed, or @c None when idle.
    current_op: Optional[Operation] = None
    ## @brief Operations waiting to be processed, in arrival order.
    waiting_queue: list[Operation] = field(default_factory=list)

    def start_processing(self, op: Operation, current_time: float):
        ## @brief Transitions to @c BUSY and stamps @c start_time / @c end_time on @p op.
        #
        #  @param op The operation to begin processing.
        #  @param current_time The simulation time at which processing starts.
        self.state = MachineState.BUSY
        self.current_op = op
        op.start_time = current_time
        op.end_time = current_time + op.duration

    def finish_processing(self):
        ## @brief Transitions to @c IDLE and clears @c current_op.
        #
        #  @details Does not select the next queued operation — that is handled by
        #  @ref DESSimulator._try_start_next_on_machine, matching the C# separation
        #  of concerns in @c Machine.FinishProcessing.
        self.state = MachineState.IDLE
        self.current_op = None


# ── Dispatching Rules ─────────────────────────────────────────────────────────


## @brief Computes total remaining processing time for a job from @p op onward.
#
#  @details Sums durations from @c op.op_index to the end of the job's operation list.
#  Used as the sort key for @ref rule_mwr. Mirrors C# @c DispatchingRules.RemainingWork.
#
#  @param op The reference operation whose index marks the start of summation.
#  @param jobs The full job list used to retrieve the owning job's operations.
#  @returns Total remaining duration including @p op itself.
def remaining_work(op: Operation, jobs: list[Job]) -> int:
    job = jobs[op.job_id]
    return sum(o.duration for o in job.operations[op.op_index :])


## @brief Computes the number of unprocessed operations remaining for a job from @p op onward.
#
#  @details Used as the sort key for @ref rule_mor. Mirrors the inline lambda in the
#  C# @c DispatchingRule.MOR switch arm.
#
#  @param op The reference operation whose index marks the start of counting.
#  @param jobs The full job list used to retrieve the owning job's operations.
#  @returns Count of remaining operations including @p op itself.
def remaining_ops(op: Operation, jobs: list[Job]) -> int:
    job = jobs[op.job_id]
    return len(job.operations) - op.op_index


## @brief Shortest Processing Time — selects the operation with the smallest duration.
#  @details Tiebreaker: lowest @c job_id. Mirrors C# @c DispatchingRule.SPT_SMPT.
#  @param waiting Non-empty list of queued operations.
#  @param jobs Full job list (unused by this rule).
#  @returns The operation with the minimum @c (duration, job_id).
def rule_spt(waiting: list[Operation], jobs: list[Job]) -> Operation:
    return min(waiting, key=lambda op: (op.duration, op.job_id))


## @brief Longest Processing Time — selects the operation with the largest duration.
#  @details Tiebreaker: lowest @c job_id. Mirrors C# @c DispatchingRule.LPT_MMUR.
#  @param waiting Non-empty list of queued operations.
#  @param jobs Full job list (unused by this rule).
#  @returns The operation with the minimum @c (-duration, job_id).
def rule_lpt(waiting: list[Operation], jobs: list[Job]) -> Operation:
    return min(waiting, key=lambda op: (-op.duration, op.job_id))


## @brief Most Work Remaining — selects the operation whose job has the most remaining work.
#  @details Equivalent to Longest Remaining Time. Tiebreaker: lowest @c job_id.
#  Mirrors C# @c DispatchingRule.LRT_MMUR.
#  @param waiting Non-empty list of queued operations.
#  @param jobs Full job list used to compute @ref remaining_work.
#  @returns The operation with the minimum @c (-remaining_work, job_id).
def rule_mwr(waiting: list[Operation], jobs: list[Job]) -> Operation:
    return min(waiting, key=lambda op: (-remaining_work(op, jobs), op.job_id))


## @brief Most Operations Remaining — selects the operation whose job has the most ops left.
#  @details Tiebreaker: lowest @c job_id. Mirrors C# @c DispatchingRule.MOR.
#  @param waiting Non-empty list of queued operations.
#  @param jobs Full job list used to compute @ref remaining_ops.
#  @returns The operation with the minimum @c (-remaining_ops, job_id).
def rule_mor(waiting: list[Operation], jobs: list[Job]) -> Operation:
    return min(waiting, key=lambda op: (-remaining_ops(op, jobs), op.job_id))


## @brief First Come First Served — returns the first operation in the queue.
#  @details No sorting is applied; insertion order is preserved, matching
#  C# @c DispatchingRule.SDT_SRWT which returns @c waitingOps[0].
#  @param waiting Non-empty list of queued operations.
#  @param jobs Full job list (unused by this rule).
#  @returns The first element of @p waiting.
def rule_fcfs(waiting: list[Operation], jobs: list[Job]) -> Operation:
    return waiting[0]


## @brief Registry mapping short rule keys to their dispatching functions.
#
#  @details Keys must match the @c rule column values in @c mirror_des_makespans.csv
#  and correspond to the same short keys used in @c ValidationExporter.RuleMap and
#  @c RULE_MAP in @c generate_reference.py, ensuring all three CSVs can be merged
#  on the @c rule column without translation.
RULES: dict[str, Callable] = {
    "SPT": rule_spt,
    "LPT": rule_lpt,
    "MWR": rule_mwr,
    "MOR": rule_mor,
    "FCFS": rule_fcfs,
}

## @brief Maps short rule keys to job_shop_lib full rule name strings.
#
#  @details Written to the @c rule_full column of @c mirror_des_makespans.csv.
#  Values must match the @c rule_full column in @c reference_makespans.csv so that
#  consumers can join on either @c rule or @c rule_full.
RULE_FULL_NAMES = {
    "SPT": "shortest_processing_time",
    "LPT": "largest_processing_time",
    "MWR": "most_work_remaining",
    "MOR": "most_operations_remaining",
    "FCFS": "first_come_first_served",
}


# ── DES Simulator ─────────────────────────────────────────────────────────────


class DESSimulator:
    ## @brief Event-driven DES with per-machine queues, mirroring C# @c DESSimulator exactly.
    #
    #  @details All event handling, dispatch logic, and queue management are intentionally
    #  structured to match the C# implementation method-for-method. Any behavioural
    #  divergence from the C# @c DESSimulator is a bug.

    def __init__(self):
        ## @brief All jobs in the loaded instance.
        self.jobs: list[Job] = []
        ## @brief All machines in the loaded instance.
        self.machines: list[Machine] = []
        ## @brief Min-heap event queue, ordered by @c (time, sequence).
        self.event_queue: list[SimEvent] = []
        ## @brief Monotonically increasing counter stamped onto each enqueued event.
        self.sequence: int = 0
        ## @brief Current simulation time, advanced on each event dequeue.
        self.current_time: float = 0.0
        ## @brief Active dispatching function, set at the start of each @ref run call.
        self.rule_fn: Callable = rule_spt

    def load_instance(self, data: dict):
        ## @brief Loads a Taillard instance from its deserialized JSON dict.
        #
        #  @details Constructs @ref Machine and @ref Job lists from @c duration_matrix
        #  and @c machines_matrix, matching the layout expected by
        #  @c DESSimulator.LoadInstance in C#. Does not call @ref reset; that is
        #  deferred to @ref run so that the same loaded instance can be re-run under
        #  multiple rules without reloading.
        #
        #  @param data Dict deserialized from a Taillard JSON file. Must contain
        #  @c duration_matrix and @c machines_matrix as lists of equal-length lists.
        duration_matrix = data["duration_matrix"]
        machines_matrix = data["machines_matrix"]
        num_jobs = len(duration_matrix)
        num_machines = len(duration_matrix[0])

        self.machines = [Machine(m) for m in range(num_machines)]
        self.jobs = []

        for j in range(num_jobs):
            ops = []
            for o in range(len(duration_matrix[j])):
                ops.append(
                    Operation(
                        job_id=j,
                        op_index=o,
                        machine_id=machines_matrix[j][o],
                        duration=duration_matrix[j][o],
                    )
                )
            self.jobs.append(Job(job_id=j, operations=ops))

    def reset(self):
        ## @brief Resets all simulator state and re-enqueues job arrivals at t=0.
        #
        #  @details Clears the event heap, resets @c current_time and @c sequence to
        #  zero, restores all machines to @c IDLE with empty waiting queues, resets
        #  all job and operation timestamps to @c -1.0, and enqueues a
        #  @c JOB_ARRIVED event for every job at @c t=0. Mirrors C# @c DESSimulator.Reset.
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

        for j in range(len(self.jobs)):
            self._enqueue(0.0, EventType.JOB_ARRIVED, job_id=j, machine_id=-1)

    def _enqueue(
        self, time: float, event_type: EventType, job_id: int, machine_id: int
    ):
        ## @brief Creates a @ref SimEvent and pushes it onto the min-heap.
        #
        #  @details Stamps the event with the next @c sequence number before pushing,
        #  ensuring FIFO ordering for events at equal times. Mirrors
        #  C# @c EventQueue.Enqueue.
        #
        #  @param time Simulation time at which the event should fire.
        #  @param event_type The @ref EventType classifying this event.
        #  @param job_id ID of the associated job, or @c -1 if not applicable.
        #  @param machine_id ID of the associated machine, or @c -1 if not applicable.
        evt = SimEvent(time, self.sequence, event_type, job_id, machine_id)
        self.sequence += 1
        heapq.heappush(self.event_queue, evt)

    def run(self, rule_fn: Callable) -> float:
        ## @brief Runs the simulation to completion and returns the makespan.
        #
        #  @details Sets @p rule_fn as the active dispatching function, calls @ref reset,
        #  then processes events from the min-heap in chronological order until the queue
        #  is empty. Mirrors C# @c DESSimulator.Run.
        #
        #  @param rule_fn A dispatching function from @ref RULES to use for queue selection.
        #  @returns The makespan, defined as the maximum @c completion_time across all jobs.
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
        ## @brief Handles a @c JOB_ARRIVED event. Mirrors C# @c HandleJobArrived.
        #
        #  @details Retrieves the arriving job and calls @ref _try_dispatch_job if it
        #  is not already complete. Early-exits silently if the job has no remaining ops.
        #
        #  @param evt The event carrying the arriving job's ID.
        job = self.jobs[evt.job_id]
        if job.is_complete:
            return
        self._try_dispatch_job(job)

    def _handle_operation_complete(self, evt: SimEvent):
        ## @brief Handles an @c OPERATION_COMPLETE event. Mirrors C# @c HandleOperationComplete.
        #
        #  @details Calls @ref Machine.finish_processing, advances @c next_op_index,
        #  records @c completion_time if the job is done, otherwise immediately dispatches
        #  the job to its next machine (zero transit time). Then calls
        #  @ref _try_start_next_on_machine on the now-free machine.
        #
        #  @param evt The event carrying the completing job ID and machine ID.
        machine = self.machines[evt.machine_id]
        job = self.jobs[evt.job_id]

        machine.finish_processing()
        job.next_op_index += 1

        if job.is_complete:
            job.completion_time = self.current_time
        else:
            self._try_dispatch_job(job)

        self._try_start_next_on_machine(machine)

    def _try_dispatch_job(self, job: Job):
        ## @brief Assigns a job's current operation to its required machine, or queues it.
        #
        #  @details If the target machine is @c IDLE, starts processing immediately and
        #  enqueues an @c OPERATION_COMPLETE event. If busy, appends the operation to the
        #  machine's @c waiting_queue. Mirrors C# @c TryDispatchJob.
        #
        #  @param job The job whose @c current_operation should be dispatched.
        if job.is_complete:
            return

        op = job.current_operation
        machine = self.machines[op.machine_id]

        if machine.state == MachineState.IDLE:
            machine.start_processing(op, self.current_time)
            self._enqueue(
                op.end_time,
                EventType.OPERATION_COMPLETE,
                job_id=job.job_id,
                machine_id=machine.machine_id,
            )
        else:
            machine.waiting_queue.append(op)

    def _try_start_next_on_machine(self, machine: Machine):
        ## @brief Selects and starts the next operation from a machine's waiting queue.
        #
        #  @details No-ops if the queue is empty or the machine is not yet idle.
        #  Otherwise calls @c rule_fn to select the next operation, removes it from
        #  the queue, starts processing, and enqueues the @c OPERATION_COMPLETE event.
        #  Mirrors C# @c TryStartNextOnMachine.
        #
        #  @param machine The machine that has just become free.
        if not machine.waiting_queue:
            return
        if machine.state != MachineState.IDLE:
            return

        next_op = self.rule_fn(machine.waiting_queue, self.jobs)
        machine.waiting_queue.remove(next_op)

        machine.start_processing(next_op, self.current_time)
        self._enqueue(
            next_op.end_time,
            EventType.OPERATION_COMPLETE,
            job_id=next_op.job_id,
            machine_id=machine.machine_id,
        )

    def get_schedule(self) -> list[dict]:
        ## @brief Extracts the full post-simulation schedule as a sorted list of operation dicts.
        #
        #  @details Collects all operations from all jobs and sorts by
        #  @c (start_time, machine_id, job_id) — matching the sort order used by
        #  @c ValidationExporter.RunAndExport and @c generate_reference.py — so that
        #  @c compare_mirror.py can diff the output entry-by-entry without re-sorting.
        #
        #  @returns A list of dicts with keys @c job, @c op_index, @c machine,
        #  @c start, @c end, @c duration, one per operation across all jobs.
        ops = []
        for job in self.jobs:
            for op in job.operations:
                ops.append(
                    {
                        "job": op.job_id,
                        "op_index": op.op_index,
                        "machine": op.machine_id,
                        "start": op.start_time,
                        "end": op.end_time,
                        "duration": op.duration,
                    }
                )
        ops.sort(key=lambda x: (x["start"], x["machine"], x["job"]))
        return ops


# ── Main ──────────────────────────────────────────────────────────────────────


def find_json_files(directory: str) -> list[str]:
    ## @brief Returns sorted paths of all @c ta*.json files in @p directory.
    #
    #  @details The @c ta prefix filter avoids accidentally processing unrelated JSON
    #  files (e.g. config or output files) that may reside in the same directory.
    #
    #  @param directory Filesystem path to search.
    #  @returns Sorted list of absolute file paths matching @c ta*.json.
    files = []
    for f in sorted(os.listdir(directory)):
        if f.endswith(".json") and f.startswith("ta"):
            files.append(os.path.join(directory, f))
    return files


def main():
    ## @brief Entry point — parses arguments, runs all instance/rule combinations, and writes outputs.
    #
    #  @details Executes the following steps:
    #  -# Parses CLI arguments for instance path, directory, rule subset, and output file paths.
    #  -# Resolves the list of JSON files: single file if @c --instance is set,
    #     otherwise all @c ta*.json files found by @ref find_json_files in @c --dir.
    #  -# For each file, deserializes the instance, then runs @ref DESSimulator.run for
    #     each requested rule key. Skips unrecognised rule keys with a console warning.
    #  -# Computes the optimality gap if @c metadata.optimum is non-zero; leaves the
    #     @c gap_pct column empty otherwise.
    #  -# Appends one CSV row per instance/rule and stores the full schedule via
    #     @ref DESSimulator.get_schedule under the key @c "{name}_{rule_key}".
    #  -# Writes @c mirror_des_makespans.csv and @c mirror_des_schedules.json.

    parser = argparse.ArgumentParser(description="Mirror DES validator")
    parser.add_argument(
        "--instance", type=str, default=None, help="Path to a single JSON instance"
    )
    parser.add_argument(
        "--dir", type=str, default=".", help="Directory containing JSON instances"
    )
    parser.add_argument(
        "--rules",
        type=str,
        default="SPT,LPT,MWR,MOR,FCFS",
        help="Comma-separated rule keys",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="mirror_des_makespans.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="mirror_des_schedules.json",
        help="Output schedules JSON path",
    )
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
    csv_lines = [
        "instance,rule,rule_full,makespan,optimum,gap_pct,num_jobs,num_machines"
    ]
    all_schedules = {}

    for filepath in json_files:
        with open(filepath) as f:
            data = json.load(f)

        name = data.get("name", os.path.splitext(os.path.basename(filepath))[0])
        metadata = data.get("metadata", {})
        optimum = metadata.get("optimum")

        sim.load_instance(data)

        print(f"\n{'=' * 60}")
        print(
            f"Instance: {name} ({sim.jobs.__len__()} jobs x {sim.machines.__len__()} machines)"
        )
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

            print(
                f"  {rule_key:6s} -> makespan={makespan:8.0f}  "
                f"optimum={optimum}  gap={gap}%"
            )

            csv_lines.append(
                ",".join(
                    str(x)
                    for x in [
                        name,
                        rule_key,
                        RULE_FULL_NAMES.get(rule_key, rule_key),
                        int(makespan),
                        optimum if optimum else "",
                        gap if gap is not None else "",
                        len(sim.jobs),
                        len(sim.machines),
                    ]
                )
            )

            ## Schedule key format matches @c ValidationExporter.RunAndExport and
            #  @c generate_reference.py so that @c compare_mirror.py can look up
            #  schedules from all three sources using the same key.
            sched_key = f"{name}_{rule_key}"
            all_schedules[sched_key] = sim.get_schedule()

    with open(args.output_csv, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"\nSaved {len(csv_lines) - 1} rows to {args.output_csv}")

    with open(args.output_json, "w") as f:
        json.dump(all_schedules, f, indent=2)
    print(f"Saved {len(all_schedules)} schedules to {args.output_json}")


if __name__ == "__main__":
    main()
