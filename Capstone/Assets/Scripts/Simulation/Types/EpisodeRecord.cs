using System.Collections.Generic;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation.Types
{
    /// <summary>
    /// Immutable snapshot of a completed episode's statistics.
    /// Built by EpisodeTracker.Build() at episode end.
    ///
    /// Adding a new logging domain (e.g. Phase 3 AGV failures):
    ///   1. Add fields here
    ///   2. Add a RecordX() method in EpisodeTracker
    ///   3. Add columns in ResultsLogger
    ///   Nothing else changes.
    /// </summary>
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
        // public int   DynamicArrivals;

        // ── Per-machine statistics ────────────────────────────────────────────
        public List<MachineRecord> MachineRecords = new List<MachineRecord>();

        // ── Per-AGV statistics  (Phase 2 addition) ────────────────────────────
        public List<AGVRecord> AGVRecords = new List<AGVRecord>();

        // ── Per-segment congestion (Phase 2 addition) ─────────────────────────
        public List<SegmentRecord> SegmentRecords = new List<SegmentRecord>();
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Per-machine
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Per-machine statistics for one episode. One row in machine_utilization.csv.
    /// </summary>
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

    // ─────────────────────────────────────────────────────────────────────────
    //  Per-AGV  (new — Phase 2)
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Per-AGV time budget and throughput for one episode.
    /// One row per AGV in agv_performance.csv.
    ///
    /// Key diagnostic split:
    ///   time_idle          — parked with no assignment (demand-side slack)
    ///   time_waiting_route — assigned but blocked by quadrant reservation (congestion)
    ///   time_traveling     — actively moving along NavMesh path (productive)
    ///   time_loading       — handshake at pickup dock
    ///   time_unloading     — handshake at dropoff dock
    ///
    /// If time_waiting_route >> time_traveling, the floor is congestion-limited.
    /// If time_idle >> everything else, the floor is AGV-over-provisioned.
    /// </summary>
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

    // ─────────────────────────────────────────────────────────────────────────
    //  Per-segment congestion  (new — Phase 2)
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Per-zone congestion metrics for one episode.
    /// One row per TrafficZone in segment_congestion.csv.
    ///
    /// High block_events / traversal_count  → this zone is a chronic bottleneck.
    /// High mean_block_time                  → blockages are long when they occur.
    /// Combine with ZoneName (encodes topology) to locate physical hotspots.
    /// </summary>
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
}