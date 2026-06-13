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

        /// <summary>Unique identifier for this job.</summary>
        public int JobId;

        /// <summary>Time at which the job becomes available in the system.</summary>
        public float ArrivalTime;

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

        /// <summary>Total accumulated waiting time across all operations.</summary>
        public double TotalWaitTime;

        /// <summary>Total accumulated transit time across all operations.</summary>
        public double TotalTransitTime;

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
    }
}