using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;
using UnityEngine;

namespace Assets.Scripts.Simulation.Jobs
{
    /// @brief Defines the distinct lifecycle stages of a job within the factory.
    ///
    /// @details A job typically flows from @c NeedsRouting (spawned) through 
    /// @c WaitingForPickup, @c InTransit, @c Queued, and @c Processing. This 
    /// cycle repeats for each operation until the job finally reaches the 
    /// @c Exited state.
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

    /// @brief Encapsulates all persistent data, state, and tracking metrics for a single job.
    ///
    /// @details This is a pure data container. Logic for state transitions, AGV 
    /// assignments, and routing is handled exclusively by the central orchestrator. 
    /// It maintains the operation sequence, machine eligibility, and timing stats 
    /// for reward calculation.
    public class JobData
    {
        [Header("Identity")]
        public int JobId;
        public float ArrivalTime;

        [Header("State Control")]
        public JobState State;

        [Header("Location Context")]
        /// Current machine ID location. -1 represents the factory entry or exit zones.
        public int LocationMachineId;

        /// Designated destination machine ID. -1 indicates the factory exit.
        public int TargetMachineId;

        /// The ID of the AGV currently handling or assigned to this job. -1 if none.
        public int AssignedAgvId;

        /// The ID of an AGV dispatched to the pickup point before processing completes.
        public int PreDispatchedAgvId = -1;

        [Header("Operation Tracking")]
        public MachineType[] OperationTypes;
        public Dictionary<int, float>[] EligibleMachinesPerOp;
        public int TotalOperations;
        public int CurrentOpIndex;
        public int CompletedOps;

        [Header("Performance Metrics")]
        public double StateEntryTime;
        public double TotalWaitTime;
        public double TotalTransitTime;

        [Header("Visuals")]
        public JobVisual Visual;

        /// @brief Indicates whether the job has finished its final scheduled operation.
        public bool IsLastOperation => CompletedOps >= TotalOperations;

        /// @brief Returns the @c MachineType required for the current operation index.
        public MachineType NextRequiredType =>
            CurrentOpIndex < TotalOperations
                ? OperationTypes[CurrentOpIndex]
                : default;

        /// @brief Retrieves the processing time for a specific machine on the current operation.
        ///
        /// @param machineId The ID of the machine to query.
        /// @return The processing time in simulation seconds; returns 0 if the machine is ineligible.
        public float GetProcessingTime(int machineId)
        {
            if (CurrentOpIndex < 0 || CurrentOpIndex >= EligibleMachinesPerOp.Length)
                return 0f;

            return EligibleMachinesPerOp[CurrentOpIndex].TryGetValue(machineId, out float t) ? t : 0f;
        }
    }
}