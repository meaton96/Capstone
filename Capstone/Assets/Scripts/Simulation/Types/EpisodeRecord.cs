using System.Collections.Generic;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation.Types
{
    /// @brief Immutable snapshot of a completed episode's statistics.
    ///
    /// @details Built by EpisodeTracker.Build() at episode end. This class serves as the single
    ///          source of truth for all post-episode analytics and CSV export.
    ///
    /// Adding a new logging domain (e.g. Phase 3 AGV failures):
    ///   1. Add fields here
    ///   2. Add a RecordX() method in EpisodeTracker
    ///   3. Add columns in ResultsLogger
    ///   Nothing else changes.
    ///
    /// @note This class is mutable in its current implementation; fields are assigned by EpisodeTracker.
    /// @see EpisodeTracker.Build
    /// @see ResultsLogger
    public class EpisodeRecord
    {
        // ── Identity ─────────────────────────────────────────────────────────
        public string InstanceName;
        public string RuleName;
        public int Seed;

        // ── Episode-level metrics ─────────────────────────────────────────────
        public double Makespan;
        public double OptimalMakespan;      // 0 if unknown
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
        public StochasticConfig Stochastic;
        public string StochasticTag => Stochastic?.Tag ?? "none";

        // ── Episode-level disruption totals ───────────────────────────────────
        public int MachineFailureCount;
        public float MachineRepairTime;

        // Phase 3: AGV failure totals
        // public int   AGVFailureCount;
        // public float AGVRepairTime;

        // Phase 4: dynamic arrival totals
        public int DynamicArrivals;           // total jobs injected by the Poisson clock this episode
        public float LastDynamicArrivalTime = -1f; // sim-time of final Poisson arrival; -1 if none fired

        // ── Dynamic arrival derived metrics ───────────────────────────────────
        /// @brief Configured Poisson arrival rate (jobs/sim-second). 0 if arrivals disabled.
        public float ArrivalLambda => Stochastic is { DynamicArrivalsEnabled: true }
            ? Stochastic.ArrivalLambda : 0f;

        /// @brief Theoretical mean time between arrivals (sim-seconds). 0 if arrivals disabled.
        public float MeanInterarrivalTime => ArrivalLambda > 0f ? 1f / ArrivalLambda : 0f;

        /// @brief Realised mean interarrival time this episode (sim-seconds).
        /// @details LastDynamicArrivalTime / DynamicArrivals; 0 if no arrivals fired.
        public float RealisedMeanInterarrival => DynamicArrivals > 0 && LastDynamicArrivalTime > 0f
            ? LastDynamicArrivalTime / DynamicArrivals : 0f;

        // ── Per-machine statistics ────────────────────────────────────────────
        public List<MachineRecord> MachineRecords = new List<MachineRecord>();

        // ── Per-AGV statistics  (Phase 2 addition) ────────────────────────────
        public List<AGVRecord> AGVRecords = new List<AGVRecord>();

        // ── Per-segment congestion (Phase 2 addition) ─────────────────────────
        public List<SegmentRecord> SegmentRecords = new List<SegmentRecord>();

        // ── Per-job operation log ─────────────────────────────────────────────
        // Populated by EpisodeTracker.RecordJobOperations(jobStore.AllJobs, dynamicJobIds)
        // at episode end. One JobOperationRecord per (job × operation).
        public List<JobOperationRecord> JobOperationRecords = new List<JobOperationRecord>();
    }

    // ── Per-machine statistics ──

    /// @brief Per-machine statistics for one episode. One row in machine_utilization.csv.
    ///
    /// @details Tracks processing time, idle time, availability, and failure metrics
    ///          for each machine. Utilization and idle rates are computed on access.
    public class MachineRecord
    {
        public int MachineId;
        public string MachineType;
        public int OpsCompleted;
        public double TimeProcessing;
        public double TimeOperational;      // Makespan − total downtime

        // Derived — computed on access, not stored
        public double UtilizationRate => TimeOperational > 0 ? TimeProcessing / TimeOperational : 0;
        public double IdleTime => TimeOperational - TimeProcessing;
        public double IdleRate => TimeOperational > 0 ? IdleTime / TimeOperational : 0;
        public double AvailabilityRate;     // TimeOperational / Makespan

        // Failure tracking
        public int FailureCount;
        public float RepairTime;

        // Phase 3: AGV-stranded time per machine, etc.
    }

    // ── Per-AGV statistics (Phase 2) ──

    /// @brief Per-AGV time budget and throughput for one episode. One row per AGV in agv_performance.csv.
    ///
    /// @details Tracks the AGV's time budget across five mutually exclusive states:
    ///          - TimeIdle: parked with no assignment (demand-side slack)
    ///          - TimeWaitingRoute: assigned but blocked by zone clearance (congestion)
    ///          - TimeTraveling: actively following NavMesh path (productive movement)
    ///          - TimeLoading: handshake at pickup dock
    ///          - TimeUnloading: handshake at dropoff dock
    ///
    /// Diagnostic guidance:
    ///          - If TimeWaitingRoute >> TimeTraveling, the floor is congestion-limited.
    ///          - If TimeIdle >> all other states, the floor is AGV-over-provisioned.
    public class AGVRecord
    {
        public int AgvId;
        public int TotalTrips;           // complete pickup-to-dropoff cycles
        public double MeanTripDuration;     // avg sim-seconds from dispatch to delivery

        public double TimeIdle;             // parked, no assignment
        public double TimeWaitingRoute;     // blocked waiting for zone clearance
        public double TimeTraveling;        // actively following NavMesh path
        public double TimeLoading;          // at pickup dock, handshake timer
        public double TimeUnloading;        // at dropoff dock, handshake timer

        public double TotalPathLength;      // cumulative NavMesh distance (sim-units)
        public int RerouteCount;         // RedirectDropoff calls (machine-failure reroutes)

        // Derived
        public double ProductiveTime => TimeTraveling + TimeLoading + TimeUnloading;
        public double TotalAccountedTime =>
            TimeIdle + TimeWaitingRoute + TimeTraveling + TimeLoading + TimeUnloading;
        public double CongestionFraction =>
            TotalAccountedTime > 0 ? TimeWaitingRoute / TotalAccountedTime : 0;
    }

    // ── Per-segment congestion (Phase 2) ──

    /// @brief Per-zone congestion metrics for one episode. One row per TrafficZone in segment_congestion.csv.
    ///
    /// @details Tracks traversal counts, block events, and cumulative block time per zone.
    ///          High BlockEvents / TraversalCount indicates a chronic bottleneck.
    ///          High MeanBlockTime indicates prolonged blockages when they occur.
    ///          Combine with ZoneName (which encodes topology, e.g. "RowAisle1_Seg2")
    ///          to locate physical hotspots on the floor.
    public class SegmentRecord
    {
        public int ZoneId;
        public string ZoneName;             // e.g. "RowAisle1_Seg2" — encodes topology
        public string AisleType;            // RowAisle / SpineAisle / VerticalAisle
        public string FlowDirection;        // East / West / North / South

        public int TraversalCount;       // successful zone entries by any AGV
        public int BlockEvents;          // times TryReserve returned false
        public float TotalBlockTime;       // cumulative sim-seconds AGVs waited here

        // Derived
        public float MeanBlockTime =>
            BlockEvents > 0 ? TotalBlockTime / BlockEvents : 0f;
        public float BlockRate =>
            (TraversalCount + BlockEvents) > 0
                ? (float)BlockEvents / (TraversalCount + BlockEvents)
                : 0f;
    }

    // ── Per-job operation records ──

    /// @brief One record per (job × operation). Logged to job_operations.csv.
    ///
    /// @details Captures the full operation plan as generated — eligible machines,
    ///          processing time spread, and whether the job arrived dynamically.
    ///          Use this to verify proc-time distributions match config, diagnose
    ///          flexibility (eligible_count), and compare static vs Poisson jobs.
    ///
    /// Populate via EpisodeTracker.RecordJobOperations(jobStore.AllJobs, dynamicJobIds).
    ///
    /// Key diagnostic queries:
    ///   - GROUP BY machine_type → mean_proc_time: verify mu/sigma are landing correctly
    ///   - GROUP BY is_dynamic: do Poisson jobs have the same op-load distribution?
    ///   - WHERE eligible_count = 1: fully-typed ops (routing has no choice)
    public class JobOperationRecord
    {
        // ── Job identity ──
        public int JobId;
        public bool IsDynamic;         // true if injected by the Poisson clock
        public float ArrivalTime;       // sim-time the job entered the system

        // ── Operation identity ──
        public int OpIndex;           // 0-based position in operation sequence
        public string MachineTypeRequired; // e.g. "Mill", "Weld"

        // ── Eligible machine pool for this op ──
        public int EligibleMachineCount;  // 1 = fully typed, >1 = flexible routing choice

        // ── Processing time spread across eligible machines ──
        /// @details Spread comes from per-machine normal sampling in FJSSPJobGenerator.
        ///          Large (max - min) relative to mean indicates high variance config.
        public float MinProcTime;
        public float MaxProcTime;
        public float MeanProcTime;

        // Derived
        public float ProcTimeSpread => MaxProcTime - MinProcTime;
    }
}