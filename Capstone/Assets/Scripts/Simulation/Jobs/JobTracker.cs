using UnityEngine;
using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation.Jobs
{
    /// @brief Runtime tracking data for a single job across all of its operations.
    public class JobTracker
    {
        public int JobId;
        public int TotalOperations;
        public JobLifecycleState State;
        public int CurrentOperationIndex;
        public int CompletedOperations;
        public Vector3 WorldPosition;
        public int CurrentMachineId;
        public int NextMachineId;       // -1 until routing decision is made
        public MachineType NextMachineType; // what type the next op needs
        public double StateEntryTime;
        public double TotalWaitTime;
        public double TotalTransitTime;
        public float OperationProgress;
        public float ArrivalTime;

        // FJSSP fields — replace OperationMachineIds + OperationDurations
        public MachineType[] OperationTypes;
        public Dictionary<int, float>[] EligibleMachinesPerOp; // [opIndex] → {machineId: procTime}
        public float[] OperationStatuses;

        public bool PhysicallyAtMachine;
        public int IncomingQueueSlot;
        public JobVisual Visual;
        public int TimeInCurrentState;
    }
}
