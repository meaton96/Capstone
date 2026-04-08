using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation.Jobs
{
    // ═══════════════════════════════════════════════════════════════
    //  6 states. No overlaps. No ambiguity.
    //
    //  Spawned ──► NeedsRouting ──► WaitingForPickup ──► InTransit ──► Queued ──► Processing ─┐
    //                   ▲                                                                     │
    //                   └──────────────────── (more ops remain) ──────────────────────────────┘
    //                                                                                         │
    //              (last op done) ──► WaitingForPickup(exit) ──► InTransit(exit) ──► Exited ◄──┘
    //
    // ═══════════════════════════════════════════════════════════════

    public enum JobState
    {
        /// Agent must pick which machine handles the next operation.
        NeedsRouting,

        /// Target machine chosen. Sitting at source location, needs an AGV.
        WaitingForPickup,

        /// AGV is carrying this job.
        InTransit,

        /// Physically at the target machine. Waiting for the machine to become idle
        /// and for the agent to pick it for processing.
        Queued,

        /// Machine is actively working on this job.
        Processing,

        /// All operations complete and job has left the factory.
        Exited
    }

    /// <summary>
    /// All data for one job. Pure data — no MonoBehaviour, no callbacks, no references
    /// to SimulationBridge or any manager. The orchestrator reads/writes these fields.
    /// </summary>
    public class JobData
    {
        // ── Identity ──────────────────────────────────────────────
        public int JobId;
        public float ArrivalTime;

        // ── Current state (written ONLY by orchestrator) ─────────
        public JobState State;

        // ── Location context ─────────────────────────────────────
        /// Where the job physically is right now.
        /// -1 = factory entry area or exit area, ≥0 = machine ID.
        public int LocationMachineId;

        /// Where the job is headed. -1 = exit, ≥0 = machine ID.
        /// Set by routing decision, read by AGV dispatch.
        public int TargetMachineId;

        /// Which AGV is assigned to this job. -1 = none.
        public int AssignedAgvId;

        // ── Operation tracking ───────────────────────────────────
        public MachineType[] OperationTypes;
        public Dictionary<int, float>[] EligibleMachinesPerOp;
        public int TotalOperations;
        public int CurrentOpIndex;     // next op to execute (0-based)
        public int CompletedOps;

        // ── Timing (for reward/stats) ────────────────────────────
        public double StateEntryTime;
        public double TotalWaitTime;
        public double TotalTransitTime;

        // ── Visual (optional, orchestrator hands it off) ─────────
        public JobVisual Visual;

        // ── Helpers ──────────────────────────────────────────────

        public bool IsLastOperation => CompletedOps >= TotalOperations;

        public MachineType NextRequiredType =>
            CurrentOpIndex < TotalOperations
                ? OperationTypes[CurrentOpIndex]
                : default;

        public float GetProcessingTime(int machineId)
        {
            if (CurrentOpIndex < 0 || CurrentOpIndex >= EligibleMachinesPerOp.Length)
                return 0f;
            return EligibleMachinesPerOp[CurrentOpIndex].TryGetValue(machineId, out float t) ? t : 0f;
        }
    }
}