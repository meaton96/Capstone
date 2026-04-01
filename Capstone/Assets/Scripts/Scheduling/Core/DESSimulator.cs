using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Assets.Scripts.Scheduling.Data;

namespace Assets.Scripts.Scheduling.Core
{
    /// @brief Result of a single event-processing step.
    ///
    /// @details Returned by @ref DESSimulator.ProcessNextEvent to tell the
    /// caller whether the simulation needs an external dispatch decision,
    /// can continue autonomously, or has finished.
    public enum SimStepResult
    {
        /// @brief Event was processed and more events remain. No decision needed.
        Continue,

        /// @brief A machine has become free with a non-empty queue.
        /// The caller must supply a @ref DispatchingRule via @ref DESSimulator.ApplyDecision
        /// before calling @ref ProcessNextEvent again.
        DecisionRequired,

        /// @brief The event queue is empty. All jobs are complete.
        Done,
    }

    /// @brief Pure discrete-event simulator for Job Shop Problem (JSP) validation.
    ///
    /// @details No Unity dependencies — can run standalone or inside Unity.
    /// Transit times are zero for benchmark validation.
    /// The RL layer will later wrap this and inject non-zero transit via AGVs.
    public class DESSimulator
    {
        public Job[] Jobs { get; private set; }
        public Machine[] Machines { get; private set; }
        public EventQueue EventQueue { get; private set; }
        public double CurrentTime { get; private set; }
        public double Makespan { get; private set; }
        public DispatchingRule ActiveRule { get; set; }

        /// @brief Transit time delegate injected by the spatial layer.
        /// @details Defaults to zero for benchmark validation. Parameters represent
        /// the source machine ID and destination machine ID respectively.
        public Func<int, int, double> GetTransitTime { get; set; } = (from, to) => 0.0;

        /// @brief Callback for RL integration, invoked when a machine requires a dispatch decision.
        /// @details The integer parameter passed to the action represents the machine ID
        /// that is awaiting a dispatch assignment.
        public Action<int> OnDispatchRequired { get; set; }

        public int TotalOperationsCompleted { get; private set; }
        public int TotalJobsCompleted { get; private set; }

        /// @brief True when the simulator is paused waiting for an external dispatch decision.
        ///
        /// @details Set to true inside @ref ProcessNextEvent when @ref TryStartNextOnMachineStepped
        /// encounters a non-empty waiting queue. Cleared by @ref ApplyDecision.
        /// While true, @ref ProcessNextEvent will immediately return
        /// @ref SimStepResult.DecisionRequired without processing further events.
        public bool WaitingForDecision { get; private set; }

        /// @brief The machine ID that requires a dispatch decision.
        ///
        /// @details Only valid when @ref WaitingForDecision is true. The caller
        /// should inspect @c Machines[PendingDecisionMachineId].WaitingQueue
        /// to see the candidate operations, then call @ref ApplyDecision.
        public int PendingDecisionMachineId { get; private set; } = -1;

        public DESSimulator()
        {
            EventQueue = new EventQueue();
            ActiveRule = DispatchingRule.SPT_SMPT;
        }

        /// @brief Processes a single event from the queue (stepped execution).
        ///
        /// @details This is the stepped equivalent of the @ref Run loop. Call it
        /// repeatedly from the bridge/agent until it returns @ref SimStepResult.Done.
        ///
        /// When the return value is @ref SimStepResult.DecisionRequired, the caller
        /// MUST call @ref ApplyDecision before calling this method again.
        ///
        /// @par Event flow
        /// 1. If already waiting for a decision → return DecisionRequired immediately.
        /// 2. If queue is empty → compute makespan, return Done.
        /// 3. Dequeue the next event, advance @ref CurrentTime, and process it.
        ///    - @ref EventType.JobArrived calls the existing @ref HandleJobArrived
        ///      (which may auto-start on an idle machine — no decision needed).
        ///    - @ref EventType.OperationComplete calls @ref HandleOperationCompleteStepped
        ///      which uses @ref TryStartNextOnMachineStepped instead of the auto-dispatch version.
        /// 4. If the stepped handler flagged a decision → return DecisionRequired.
        /// 5. Otherwise → return Continue or Done depending on remaining events.
        ///
        /// @returns A @ref SimStepResult indicating what the caller should do next.
        public SimStepResult ProcessNextEvent()
        {
            // Still waiting for a decision from the last step.
            if (WaitingForDecision)
                return SimStepResult.DecisionRequired;

            if (!EventQueue.HasEvents)
            {
                Makespan = Jobs.Max(j => j.CompletionTime);
                return SimStepResult.Done;
            }

            var evt = EventQueue.Dequeue();
            CurrentTime = evt.Time;

            switch (evt.Type)
            {
                case EventType.JobArrived:
                    HandleJobArrived(evt);
                    break;

                case EventType.OperationComplete:
                    HandleOperationCompleteStepped(evt);
                    break;
            }

            if (WaitingForDecision)
                return SimStepResult.DecisionRequired;

            if (!EventQueue.HasEvents)
            {
                Makespan = Jobs.Max(j => j.CompletionTime);
                return SimStepResult.Done;
            }

            return SimStepResult.Continue;
        }

        /// @brief Stepped variant of @ref HandleOperationComplete.
        ///
        /// @details Identical to the original except it calls
        /// @ref TryStartNextOnMachineStepped, which pauses for an external
        /// decision instead of auto-dispatching.
        ///
        /// @param evt The OperationComplete event.
        private void HandleOperationCompleteStepped(SimEvent evt)
        {
            var machine = Machines[evt.MachineId];
            var job = Jobs[evt.JobId];

            machine.FinishProcessing();
            TotalOperationsCompleted++;

            job.NextOperationIndex++;

            if (job.IsComplete)
            {
                job.CompletionTime = CurrentTime;
                TotalJobsCompleted++;
            }
            else
            {
                var nextOp = job.CurrentOperation;
                double transitTime = GetTransitTime(machine.Id, nextOp.MachineId);
                double arrivalTime = CurrentTime + transitTime;

                if (transitTime <= 0)
                {
                    TryDispatchJob(job);
                }
                else
                {
                    EventQueue.Enqueue(arrivalTime, EventType.JobArrived, jobId: job.Id);
                }
            }

            // This is the only line that differs from the original:
            TryStartNextOnMachineStepped(machine);
        }

        /// @brief Stepped variant of @ref TryStartNextOnMachine.
        ///
        /// @details Instead of auto-selecting the next operation via
        /// @ref DispatchingRules.SelectNext, this method sets
        /// @ref WaitingForDecision and @ref PendingDecisionMachineId, then
        /// returns without dispatching. The external caller must inspect
        /// the queue and call @ref ApplyDecision with a chosen rule.
        ///
        /// If the queue is empty or the machine is not idle, behaves
        /// identically to the original (no-op).
        ///
        /// @param machine The machine that just became free.
        private void TryStartNextOnMachineStepped(Machine machine)
        {
            if (machine.WaitingQueue.Count == 0) return;
            if (machine.State != MachineState.Idle) return;

            // Pause: the external caller must decide which job to dispatch.
            WaitingForDecision = true;
            PendingDecisionMachineId = machine.Id;
        }

        /// @brief Applies an external dispatch decision and resumes the simulation.
        ///
        /// @details Uses the supplied @p rule to select the next operation from the
        /// pending machine's waiting queue, starts processing it, and enqueues the
        /// corresponding @ref EventType.OperationComplete event. Clears the
        /// @ref WaitingForDecision flag so @ref ProcessNextEvent can continue.
        ///
        /// @param rule The @ref DispatchingRule the agent/bridge chose (one of the
        /// 8 composite PDRs, or any single rule for validation).
        ///
        /// @pre @ref WaitingForDecision must be true.
        /// @post @ref WaitingForDecision is false, the chosen operation is running
        /// on the machine, and a completion event has been enqueued.
        public void ApplyDecision(DispatchingRule rule)
        {
            if (!WaitingForDecision)
            {
                throw new InvalidOperationException(
                    "ApplyDecision called but no decision is pending.");
            }

            var machine = Machines[PendingDecisionMachineId];
            var nextOp = DispatchingRules.SelectNext(machine.WaitingQueue, Jobs, rule);
            machine.WaitingQueue.Remove(nextOp);

            machine.StartProcessing(nextOp, CurrentTime);
            EventQueue.Enqueue(nextOp.EndTime, EventType.OperationComplete,
                jobId: nextOp.JobId, machineId: machine.Id);

            WaitingForDecision = false;
            PendingDecisionMachineId = -1;
        }

        /// @brief Resets the stepped-execution state.
        ///
        /// @details Call this at the END of your existing @ref Reset method body,
        /// or anywhere after the base reset logic clears the event queue and
        /// re-enqueues job arrivals.
        private void ResetSteppedState()
        {
            WaitingForDecision = false;
            PendingDecisionMachineId = -1;
        }

        /// @brief Loads a Taillard instance and initializes all jobs and machines.
        ///
        /// @details Constructs @ref Machine and @ref Job arrays from the provided instance,
        /// mapping raw operation data into typed @ref Operation objects, then calls
        /// @ref Reset() to prepare the simulator for execution.
        ///
        /// @param instance The Taillard benchmark instance to load, containing job count,
        /// machine count, and per-job operation sequences with durations.
        public void LoadInstance(TaillardInstance instance)
        {

            int jobCount = instance.JobCount;
            int machineCount = instance.MachineCount;

            Machines = new Machine[machineCount];
            for (int m = 0; m < machineCount; m++)
                Machines[m] = new Machine(m);

            Jobs = new Job[jobCount];
            for (int j = 0; j < jobCount; j++)
            {
                var rawOps = instance.GetJobOperations(j);
                var operations = new Operation[rawOps.Length];
                for (int o = 0; o < rawOps.Length; o++)
                {
                    operations[o] = new Operation(j, o, rawOps[o].machine, rawOps[o].duration);
                }
                Jobs[j] = new Job(j, operations, arrivalTime: 0);
            }
            Reset();
        }

        /// @brief Resets all simulator state without reloading the instance.
        ///
        /// @details Clears the event queue, resets time and statistics counters, restores
        /// all machines to @ref MachineState.Idle, clears all machine waiting queues,
        /// and re-enqueues a @ref EventType.JobArrived event at t=0 for every job.
        /// Call this to re-run the same instance under a different dispatching rule.
        public void Reset()
        {
            CurrentTime = 0;
            Makespan = 0;
            TotalOperationsCompleted = 0;
            TotalJobsCompleted = 0;
            EventQueue.Clear();

            foreach (var machine in Machines)
            {
                machine.State = MachineState.Idle;
                machine.CurrentOperation = null;
                machine.BusyUntil = 0;
                machine.WaitingQueue.Clear();
            }

            foreach (var job in Jobs)
            {
                job.NextOperationIndex = 0;
                job.CompletionTime = -1;
                foreach (var op in job.Operations)
                {
                    op.StartTime = -1;
                    op.EndTime = -1;
                }
            }

            for (int j = 0; j < Jobs.Length; j++)
            {
                EventQueue.Enqueue(0, EventType.JobArrived, jobId: j);
            }
            ResetSteppedState();
        }

        /// @brief Runs the full simulation to completion and returns the makespan.
        ///
        /// @details Processes all events in the @ref EventQueue in chronological order,
        /// advancing @ref CurrentTime to each event's timestamp and dispatching to either
        /// @ref HandleJobArrived or @ref HandleOperationComplete as appropriate.
        /// Simulation ends when the queue is empty.
        ///
        /// @returns The makespan, defined as the latest completion time across all jobs.
        public double Run()
        {
            while (EventQueue.HasEvents)
            {
                var evt = EventQueue.Dequeue();
                CurrentTime = evt.Time;

                switch (evt.Type)
                {
                    case EventType.JobArrived:
                        HandleJobArrived(evt);
                        break;
                    case EventType.OperationComplete:
                        HandleOperationComplete(evt);
                        break;
                }
            }

            Makespan = Jobs.Max(j => j.CompletionTime);
            return Makespan;
        }

        /// @brief Handles a @ref EventType.JobArrived event.
        ///
        /// @details Retrieves the arriving job and calls @ref TryDispatchJob if it is
        /// not already complete. Early-exits silently if the job has no remaining operations.
        ///
        /// @param evt The simulation event containing the ID of the arriving job.
        private void HandleJobArrived(SimEvent evt)
        {
            var job = Jobs[evt.JobId];
            if (job.IsComplete) return;

            TryDispatchJob(job);
        }

        /// @brief Handles a @ref EventType.OperationComplete event.
        ///
        /// @details Marks the current operation finished on the machine, increments
        /// @ref TotalOperationsCompleted, and advances the job's operation index.
        /// If the job is now fully complete its @ref Job.CompletionTime is recorded and
        /// @ref TotalJobsCompleted is incremented. Otherwise, the transit time to the next
        /// machine is queried via @ref GetTransitTime: if zero the job is dispatched
        /// immediately, otherwise a future @ref EventType.JobArrived event is enqueued.
        /// Finally, @ref TryStartNextOnMachine is called to assign a waiting job to the
        /// newly freed machine.
        ///
        /// @param evt The simulation event containing the completing job ID and machine ID.
        private void HandleOperationComplete(SimEvent evt)
        {
            var machine = Machines[evt.MachineId];
            var job = Jobs[evt.JobId];

            machine.FinishProcessing();
            TotalOperationsCompleted++;

            job.NextOperationIndex++;

            if (job.IsComplete)
            {
                job.CompletionTime = CurrentTime;
                TotalJobsCompleted++;
            }
            else
            {
                var nextOp = job.CurrentOperation;
                double transitTime = GetTransitTime(machine.Id, nextOp.MachineId);
                double arrivalTime = CurrentTime + transitTime;

                if (transitTime <= 0)
                {
                    TryDispatchJob(job);
                }
                else
                {
                    EventQueue.Enqueue(arrivalTime, EventType.JobArrived, jobId: job.Id);
                }
            }

            TryStartNextOnMachine(machine);
        }

        /// @brief Attempts to assign a job's current operation to its required machine.
        ///
        /// @details If the target machine is @ref MachineState.Idle the operation starts
        /// immediately and a corresponding @ref EventType.OperationComplete event is
        /// enqueued. If the machine is busy the operation is appended to the machine's
        /// @ref Machine.WaitingQueue to be picked up when the machine next becomes free.
        ///
        /// @param job The job whose @ref Job.CurrentOperation should be dispatched.
        private void TryDispatchJob(Job job)
        {
            if (job.IsComplete) return;

            var op = job.CurrentOperation;
            var machine = Machines[op.MachineId];

            if (machine.State == MachineState.Idle)
            {
                machine.StartProcessing(op, CurrentTime);
                EventQueue.Enqueue(op.EndTime, EventType.OperationComplete,
                    jobId: job.Id, machineId: machine.Id);
            }
            else
            {
                machine.WaitingQueue.Add(op);
            }
        }

        /// @brief Selects and starts the next operation from a machine's waiting queue.
        ///
        /// @details No-ops if the queue is empty or the machine is not yet idle.
        /// Otherwise delegates selection to @ref DispatchingRules.SelectNext using
        /// @ref ActiveRule, removes the chosen operation from the queue, starts
        /// processing, and enqueues the resulting @ref EventType.OperationComplete event.
        ///
        /// @param machine The machine that has just become free and requires a new assignment.
        private void TryStartNextOnMachine(Machine machine)
        {
            if (machine.WaitingQueue.Count == 0) return;
            if (machine.State != MachineState.Idle) return;

            var nextOp = DispatchingRules.SelectNext(machine.WaitingQueue, Jobs, ActiveRule);
            machine.WaitingQueue.Remove(nextOp);

            machine.StartProcessing(nextOp, CurrentTime);
            EventQueue.Enqueue(nextOp.EndTime, EventType.OperationComplete,
                jobId: nextOp.JobId, machineId: machine.Id);
        }

        /// @brief Prints a Gantt-style schedule summary to stdout for debugging.
        ///
        /// @details Outputs the makespan, job and machine counts, total operations
        /// completed, and a per-machine timeline showing each operation as
        /// @c [J{jobId} {startTime}-{endTime}], sorted by start time.
        public void PrintSchedule()
        {
            Console.WriteLine($"=== Schedule Results ===");
            Console.WriteLine($"Makespan: {Makespan}");
            Console.WriteLine($"Jobs: {Jobs.Length}, Machines: {Machines.Length}");
            Console.WriteLine($"Operations completed: {TotalOperationsCompleted}");
            Console.WriteLine();

            for (int m = 0; m < Machines.Length; m++)
            {
                Console.Write($"Machine {m,2}: ");
                var opsOnMachine = Jobs
                    .SelectMany(j => j.Operations)
                    .Where(op => op.MachineId == m)
                    .OrderBy(op => op.StartTime);

                foreach (var op in opsOnMachine)
                {
                    Console.Write($"[J{op.JobId} {op.StartTime}-{op.EndTime}] ");
                }
                Console.WriteLine();
            }
        }
    }
}