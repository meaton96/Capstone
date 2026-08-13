using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;
using UnityEngine;

namespace Assets.Scripts.Simulation.Jobs
{
    /// <summary>
    /// Defines the distinct lifecycle stages of a job within the factory.
    /// </summary>
    /// <remarks>
    /// A job typically flows from <c>NeedsRouting</c> (spawned) through <c>WaitingForPickup</c>,
    /// <c>InTransit</c>, <c>Queued</c>, and <c>Processing</c>. This cycle repeats for each
    /// operation until the job finally reaches the <c>Exited</c> state.
    /// </remarks>
    public enum JobState
    {
        /// Agent must select the next machine for processing.
        NeedsRouting,

        /// Target machine assigned; job is awaiting AGV pickup at its current location.
        WaitingForPickup,

        /// Job is currently being transported by an AGV.
        InTransit,

        /// Job has arrived at the target machine and is waiting in the machine's buffer.
        Queued,

        /// Machine is actively performing the scheduled operation on this job.
        Processing,

        /// All scheduled operations are complete and the job has left the simulation.
        Exited
    }

    /// <summary>
    /// Encapsulates all persistent data, state, and tracking metrics for a single job.
    /// </summary>
    /// <remarks>
    /// This is a pure data container. Logic for state transitions, AGV assignments,
    /// and routing is handled exclusively by the central orchestrator. It maintains
    /// the operation sequence, machine eligibility, and timing stats for reward calculation.
    /// </remarks>
    public class JobData
    {
        /// <summary>
        /// AGV transit duration (sim-seconds) for each operation, indexed by operation position.
        /// Stamped in FlagHarvester when DeliveredFlag is processed:
        ///   job.OperationTravelTimes[job.CurrentOpIndex] = agv.LastTripDuration;
        /// Zero for operations not yet delivered or where transit was not recorded.
        /// </summary>
        public float[] OperationTravelTimes;

        /// <summary>
        /// Sim-time each operation's job entered Queued state at its target machine, indexed by
        /// operation position. -1 if the operation has not yet reached that point.
        /// </summary>
        public float[] OpQueueEntryTimes;

        /// <summary>
        /// Sim-time each operation began Processing, indexed by operation position.
        /// -1 if not yet started. Stamped in ExecuteDispatchDecision.
        /// </summary>
        public float[] OpProcStartTimes;

        /// <summary>
        /// Sim-time each operation finished Processing, indexed by operation position.
        /// -1 if not yet completed. Stamped in FlagHarvester.HarvestMachineFlags, before
        /// CurrentOpIndex advances, so the index always refers to the op that just finished.
        /// </summary>
        public float[] OpProcEndTimes;

        /// <summary>Unique identifier for this job.</summary>
        public int JobId;

        /// <summary>Time at which the job becomes available in the system.</summary>
        public float ArrivalTime;

        /// <summary>Sim-time the job reached JobState.Exited. -1 if still in the system.</summary>
        public float ExitTime = -1f;

        [Header("State Control")]
        /// <summary>Current lifecycle state of the job.</summary>
        public JobState State;

        [Header("Location Context")]
        /// <summary>
        /// Current machine ID where the job is located. -1 represents the factory entry or exit zones.
        /// </summary>
        public int LocationMachineId;

        /// <summary>
        /// Designated destination machine ID for the job. -1 indicates the factory exit.
        /// </summary>
        public int TargetMachineId;

        /// <summary>
        /// The ID of the AGV currently handling or assigned to this job. -1 if none.
        /// </summary>
        public int AssignedAgvId;

        /// <summary>
        /// The ID of an AGV dispatched to the pickup point before processing completes.
        /// </summary>
        public int PreDispatchedAgvId = -1;

        [Header("Operation Tracking")]
        /// <summary>Machine types required for each operation in the job sequence.</summary>
        public MachineType[] OperationTypes;

        /// <summary>
        /// Per-operation mapping of eligible machine IDs to processing times.
        /// </summary>
        public Dictionary<int, float>[] EligibleMachinesPerOp;

        /// <summary>Total number of operations scheduled for this job.</summary>
        public int TotalOperations;

        /// <summary>Index of the current operation being processed (0-based).</summary>
        public int CurrentOpIndex;

        /// <summary>Number of operations completed so far.</summary>
        public int CompletedOps;

        [Header("Performance Metrics")]
        /// <summary>Simulation time when the current state was entered.</summary>
        public double StateEntryTime;

        // ── Per-state cumulative time budget (sim-seconds), updated by TransitionTo ────
        /// <summary>Cumulative time spent waiting for a routing decision.</summary>
        public double TimeNeedsRouting;
        /// <summary>Cumulative time spent waiting for AGV pickup after routing/processing.</summary>
        public double TimeWaitingPickup;
        /// <summary>Cumulative time spent being transported by an AGV.</summary>
        public double TimeInTransit;
        /// <summary>Cumulative time spent queued at a machine awaiting dispatch.</summary>
        public double TimeQueued;
        /// <summary>Cumulative time spent actively being processed by a machine.</summary>
        public double TimeProcessing;

        [Header("Visuals")]
        /// <summary>Visual representation of this job on the factory floor.</summary>
        public JobVisual Visual;

        /// <summary>
        /// Indicates whether the job has finished its final scheduled operation.
        /// </summary>
        public bool IsLastOperation => CompletedOps >= TotalOperations;

        /// <summary>
        /// Returns the <see cref="MachineType"/> required for the current operation index.
        /// </summary>
        public MachineType NextRequiredType =>
            CurrentOpIndex < TotalOperations
                ? OperationTypes[CurrentOpIndex]
                : default;

        /// <summary>
        /// Retrieves the processing time for a specific machine on the current operation.
        /// </summary>
        /// <param name="machineId">The ID of the machine to query.</param>
        /// <returns>The processing time in simulation seconds; returns 0 if the machine is ineligible.</returns>
        public float GetProcessingTime(int machineId)
        {
            if (CurrentOpIndex < 0 || CurrentOpIndex >= EligibleMachinesPerOp.Length)
                return 0f;

            return EligibleMachinesPerOp[CurrentOpIndex].TryGetValue(machineId, out float t) ? t : 0f;
        }

        /// <summary>
        /// Moves the job to <paramref name="next"/>, first folding the time spent in the
        /// current state into that state's cumulative bucket. This is the single source of
        /// truth for State/StateEntryTime mutation — every state-changing call site should
        /// use this instead of assigning State/StateEntryTime directly, so the time buckets
        /// stay consistent with the realized state machine (including failure-driven reroutes).
        /// </summary>
        /// <param name="next">The state being entered.</param>
        /// <param name="simTime">Current simulation time.</param>
        public void TransitionTo(JobState next, double simTime)
        {
            double dt = simTime - StateEntryTime;
            if (dt > 0)
            {
                switch (State)
                {
                    case JobState.NeedsRouting: TimeNeedsRouting += dt; break;
                    case JobState.WaitingForPickup: TimeWaitingPickup += dt; break;
                    case JobState.InTransit: TimeInTransit += dt; break;
                    case JobState.Queued: TimeQueued += dt; break;
                    case JobState.Processing: TimeProcessing += dt; break;
                }
            }
            State = next;
            StateEntryTime = simTime;
        }
    }
}