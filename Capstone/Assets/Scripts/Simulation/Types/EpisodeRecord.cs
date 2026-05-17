using System.Collections.Generic;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation.Types
{
    /// <summary>
    /// Immutable snapshot of a completed episode's statistics.
    /// Built by EpisodeTracker.Build() at episode end.
    ///
    /// This is the single object that travels from SimulationBridge through
    /// HeadlessBatchRunner to ResultsLogger. Adding a new stochastic event
    /// only requires:
    ///   1. A new field here
    ///   2. A new RecordX() method in EpisodeTracker
    ///   3. A new column in ResultsLogger.LogEpisode(record)
    ///
    /// Nothing else changes — no parameter lists to update, no event
    /// signatures to modify.
    /// </summary>
    public class EpisodeRecord
    {
        // ── Identity ─────────────────────────────────────────────────────────

        public string InstanceName;
        public string RuleName;
        public int Seed;

        // ── Episode-level metrics ─────────────────────────────────────────────

        public double Makespan;
        public double OptimalMakespan;     // 0 if unknown
        public double TotalReward;
        public int JobCount;
        public int MachineCount;
        public int AGVCount;
        public int TotalOperations;
        public int CompletedJobs;
        public int DecisionPoints;
        public float AverageTimeScale;

        public double OptimalityGap => OptimalMakespan > 0
            ? (Makespan - OptimalMakespan) / OptimalMakespan * 100.0
            : 0;

        // ── Stochastic config snapshot ────────────────────────────────────────
        // Null for deterministic runs. Columns default to 0/"none" in logger.

        public StochasticConfig Stochastic;
        public string StochasticTag => Stochastic?.Tag ?? "none";

        // ── Episode-level disruption totals ───────────────────────────────────

        public int MachineFailureCount;
        public float MachineRepairTime;     // total cumulative repair seconds

        // Phase 3: AGV failure totals — add fields here when implemented
        // public int   AGVFailureCount;
        // public float AGVRepairTime;

        // Phase 4: dynamic arrival totals — add here when implemented
        // public int   DynamicArrivals;

        // ── Per-machine statistics ────────────────────────────────────────────
        // Keyed by machine ID. Populated by EpisodeTracker.Build().

        public List<MachineRecord> MachineRecords = new List<MachineRecord>();
    }

    /// <summary>
    /// Per-machine statistics for one episode. One row in machine_utilization.csv.
    /// </summary>
    public class MachineRecord
    {
        public int MachineId;
        public string MachineType;

        public int OpsCompleted;
        public double TimeProcessing;
        public double TimeOperational;   // Makespan - total downtime

        // Derived (computed by ResultsLogger or EpisodeTracker)
        public double UtilizationRate => TimeOperational > 0 ? TimeProcessing / TimeOperational : 0;
        public double IdleTime => TimeOperational - TimeProcessing;
        public double IdleRate => TimeOperational > 0 ? IdleTime / TimeOperational : 0;
        public double AvailabilityRate;  // TimeOperational / Makespan

        // Failure tracking per machine
        public int FailureCount;
        public float RepairTime;

        // Phase 3: AGV-stranded time, etc. — add fields here
    }
}