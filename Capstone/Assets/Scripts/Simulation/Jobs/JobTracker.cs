using UnityEngine;
using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation.Jobs
{
    /// @brief Where a job physically exists in the factory.
    /// @details This is the SINGLE SOURCE OF TRUTH for job location.
    ///          All queries about "which jobs are at machine X" go through
    ///          JobManager using this field — never through ConveyorBelt.
    public enum JobLocation
    {
        PendingEntry,       // created but not yet on factory floor
        OnFactoryBelt,      // on the factory incoming conveyor, visible
        AwaitingPickup,     // AGV assigned, waiting to be picked up
        InTransit,          // being carried by an AGV
        InMachineQueue,     // on a machine's incoming belt, waiting to process
        Processing,         // inside a machine, being worked on
        AwaitingTransport,  // on a machine's outgoing belt, needs routing or exit
        InTransitToExit,    // AGV carrying to factory exit
        OnExitBelt,         // on the exit conveyor
        Exited              // deactivated, done
    }

    /// @brief Runtime tracking data for a single job across all of its operations.
    /// @details JobTracker is the authoritative record for every job. Its Location
    ///          and LocationMachineId fields are the single source of truth —
    ///          ConveyorBelt entries and AGVPool requests are visual/transport
    ///          layers only.
    public class JobTracker
    {
        public int JobId;
        public int TotalOperations;
        // public JobLifecycleState State;
        public int CurrentOperationIndex;
        public int CompletedOperations;
        public Vector3 WorldPosition;
        public int CurrentMachineId;
        public int NextMachineId;           // -1 until routing decision is made
        public MachineType NextMachineType; // what type the next op needs
        public double StateEntryTime;
        public double TotalWaitTime;
        public double TotalTransitTime;
        public float OperationProgress;
        public float ArrivalTime;

        // ── Single source of truth ──────────────────────────────────
        /// @brief Where this job physically is right now.
        public JobLocation Location;
        /// @brief Which machine this job is at (-1 = factory belt / exit / in transit).
        public int LocationMachineId = -1;
        /// @brief Which AGV is carrying this job (-1 = none).
        public int AssignedAGVId = -1;

        // FJSSP fields
        public MachineType[] OperationTypes;
        public Dictionary<int, float>[] EligibleMachinesPerOp;
        public float[] OperationStatuses;

        public bool PhysicallyAtMachine;
        public int IncomingQueueSlot;
        public JobVisual Visual;
        public int TimeInCurrentState;
    }
}