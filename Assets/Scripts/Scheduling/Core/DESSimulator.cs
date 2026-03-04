using System;
using System.Collections.Generic;
using System.Linq;
using Scheduling.Data;

namespace Scheduling.Core
{
    /// <summary>
    /// Pure discrete-event simulator for JSP validation.
    /// No Unity dependencies — can run standalone or inside Unity.
    /// 
    /// Transit times are zero for benchmark validation.
    /// The RL layer will later wrap this and inject non-zero transit via AGVs.
    /// </summary>
    public class DESSimulator
    {
        public Job[] Jobs { get; private set; }
        public Machine[] Machines { get; private set; }
        public EventQueue EventQueue { get; private set; }
        public double CurrentTime { get; private set; }
        public double Makespan { get; private set; }
        public DispatchingRule ActiveRule { get; set; }

        // Transit time injected by the spatial layer (zero for validation)
        public Func<int, int, double> GetTransitTime { get; set; } = (from, to) => 0.0;

        // Callback for RL integration: called when a machine needs a dispatch decision
        public Action<int> OnDispatchRequired { get; set; }

        // Statistics
        public int TotalOperationsCompleted { get; private set; }
        public int TotalJobsCompleted { get; private set; }

        public DESSimulator()
        {
            EventQueue = new EventQueue();
            ActiveRule = DispatchingRule.SPT_SMPT;
        }

        /// <summary>
        /// Load a Taillard instance and initialize all jobs and machines.
        /// </summary>
        public void LoadInstance(TaillardInstance instance)
        {
            int jobCount = instance.JobCount;
            int machineCount = instance.MachineCount;

            // Create machines
            Machines = new Machine[machineCount];
            for (int m = 0; m < machineCount; m++)
                Machines[m] = new Machine(m);

            // Create jobs from the Taillard data
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

        /// <summary>
        /// Reset simulator state without reloading the instance.
        /// </summary>
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

            // Enqueue all job arrivals at t=0
            for (int j = 0; j < Jobs.Length; j++)
            {
                EventQueue.Enqueue(0, EventType.JobArrived, jobId: j);
            }
        }

        /// <summary>
        /// Run the full simulation to completion.
        /// Returns the makespan.
        /// </summary>
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

            // Makespan = latest job completion time
            Makespan = Jobs.Max(j => j.CompletionTime);
            return Makespan;
        }

        private void HandleJobArrived(SimEvent evt)
        {
            var job = Jobs[evt.JobId];
            if (job.IsComplete) return;

            TryDispatchJob(job);
        }

        private void HandleOperationComplete(SimEvent evt)
        {
            var machine = Machines[evt.MachineId];
            var job = Jobs[evt.JobId];

            // Finish processing on this machine
            machine.FinishProcessing();
            TotalOperationsCompleted++;

            // Advance job to next operation
            job.NextOperationIndex++;

            if (job.IsComplete)
            {
                // Job is done
                job.CompletionTime = CurrentTime;
                TotalJobsCompleted++;
            }
            else
            {
                // Send job to its next machine (with transit time)
                var nextOp = job.CurrentOperation;
                double transitTime = GetTransitTime(machine.Id, nextOp.MachineId);
                double arrivalTime = CurrentTime + transitTime;

                if (transitTime <= 0)
                {
                    // Zero transit: immediately try to dispatch
                    TryDispatchJob(job);
                }
                else
                {
                    // Schedule arrival at next machine after transit
                    EventQueue.Enqueue(arrivalTime, EventType.JobArrived, jobId: job.Id);
                }
            }

            // Check if any waiting jobs can start on the now-free machine
            TryStartNextOnMachine(machine);
        }

        /// <summary>
        /// Try to assign a job's current operation to its required machine.
        /// If the machine is busy, add to waiting queue.
        /// </summary>
        private void TryDispatchJob(Job job)
        {
            if (job.IsComplete) return;

            var op = job.CurrentOperation;
            var machine = Machines[op.MachineId];

            if (machine.State == MachineState.Idle)
            {
                // Machine is free — start immediately
                machine.StartProcessing(op, CurrentTime);
                EventQueue.Enqueue(op.EndTime, EventType.OperationComplete,
                    jobId: job.Id, machineId: machine.Id);
            }
            else
            {
                // Machine is busy — add to waiting queue
                machine.WaitingQueue.Add(op);
            }
        }

        /// <summary>
        /// When a machine becomes free, pick the next job from its queue.
        /// Uses the active dispatching rule.
        /// </summary>
        private void TryStartNextOnMachine(Machine machine)
        {
            if (machine.WaitingQueue.Count == 0) return;
            if (machine.State != MachineState.Idle) return;

            // Select next operation via dispatching rule
            var nextOp = DispatchingRules.SelectNext(machine.WaitingQueue, Jobs, ActiveRule);
            machine.WaitingQueue.Remove(nextOp);

            machine.StartProcessing(nextOp, CurrentTime);
            EventQueue.Enqueue(nextOp.EndTime, EventType.OperationComplete,
                jobId: nextOp.JobId, machineId: machine.Id);
        }

        /// <summary>
        /// Print a Gantt-style summary to the console for debugging.
        /// </summary>
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
