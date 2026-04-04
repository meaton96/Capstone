using UnityEngine;

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
        public int NextMachineId;
        public double StateEntryTime;
        public double TotalWaitTime;
        public double TotalTransitTime;
        public float OperationProgress;
        public float[] OperationStatuses;
        public int[] OperationMachineIds;
        public float[] OperationDurations;
        public bool PhysicallyAtMachine;
        public int IncomingQueueSlot;
        public JobVisual Visual;
        public int TimeInCurrentState;
    }
}
