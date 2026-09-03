using System.Collections.Generic;
using System.Linq;
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

        // ── Configuration snapshot ────────────────────────────────────────────
        public string ParkingMethod;
        public string PreDispatchingMethod;

        // ── Deadlock watchdog outcome ──────────────────────────────────────────
        // Set by FactoryOrchestrator.CheckForDeadlock. True means the episode was terminated
        // early because no zone anywhere in the AGV traffic network admitted an entry for
        // DEADLOCK_STALL_SECONDS — a circular-wait deadlock, not a genuine long-running episode.
        // Makespan for a deadlocked episode is the sim-time deadlock was DETECTED at (roughly
        // stall-onset + DEADLOCK_STALL_SECONDS), not a fixed timeout sentinel — check this flag
        // rather than thresholding makespan to identify deadlocked rows.
        public bool DeadlockDetected;
        public double DeadlockSimTime = -1.0;

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

        // ── Per-decision log (routing + dispatch) ─────────────────────────────
        // Appended live by FactoryOrchestrator.ExecuteRoutingDecision/ExecuteDispatchDecision
        // as decisions happen, copied onto the record at FinaliseEpisode. One DecisionRecord
        // per actual decision (does NOT include the queue.Count<=1 / candidates.Length<=1
        // degenerate cases that DispatchingEngine short-circuits before any rule runs --
        // IsDegenerate flags those so they're still visible, not silently dropped).
        public List<DecisionRecord> DecisionRecords = new List<DecisionRecord>();
        // ── Per-window throughput log ─────────────────────────────────────────
        // Populated by EpisodeTracker.CloseThroughputWindow() each ThroughputTimingWindow.
        public List<ThroughputWindowRecord> ThroughputRecords = new List<ThroughputWindowRecord>();

        // ── Per-job completion log (Option A) ───────────────────────────────
        // Populated in FactoryOrchestrator.FinaliseEpisode from jobStore.AllJobs.
        // One row per job — the realized flow-time / wait-decomposition record that
        // job_operations.csv (static specs only) cannot provide.
        public List<JobCompletionRecord> JobCompletionRecords = new List<JobCompletionRecord>();

        // ── Derived flow-time summary (computed over completed jobs only) ────
        public double MeanFlowTime => JobCompletionRecords.Count == 0 ? 0
            : Mean(JobCompletionRecords, r => r.Completed, r => r.FlowTime);
        public double P95FlowTime => Percentile(JobCompletionRecords, 0.95);
        public double MaxFlowTime => JobCompletionRecords.Count == 0 ? 0
            : MaxOf(JobCompletionRecords, r => r.Completed, r => r.FlowTime);
        public double MeanTransportWait => JobCompletionRecords.Count == 0 ? 0
            : Mean(JobCompletionRecords, r => r.Completed, r => r.TimeWaitingPickup + r.TimeInTransit);
        public int JobsCensored => JobCompletionRecords.Count(r => !r.Completed);

        private static double Mean(List<JobCompletionRecord> records,
            System.Func<JobCompletionRecord, bool> filter, System.Func<JobCompletionRecord, double> select)
        {
            double sum = 0; int n = 0;
            foreach (var r in records) { if (!filter(r)) continue; sum += select(r); n++; }
            return n > 0 ? sum / n : 0;
        }

        private static double MaxOf(List<JobCompletionRecord> records,
            System.Func<JobCompletionRecord, bool> filter, System.Func<JobCompletionRecord, double> select)
        {
            double best = 0; bool any = false;
            foreach (var r in records)
            {
                if (!filter(r)) continue;
                double v = select(r);
                if (!any || v > best) { best = v; any = true; }
            }
            return best;
        }

        private static double Percentile(List<JobCompletionRecord> records, double p)
        {
            var flowTimes = new List<double>();
            foreach (var r in records) if (r.Completed) flowTimes.Add(r.FlowTime);
            if (flowTimes.Count == 0) return 0;
            flowTimes.Sort();
            int idx = (int)System.Math.Ceiling(p * flowTimes.Count) - 1;
            idx = System.Math.Clamp(idx, 0, flowTimes.Count - 1);
            return flowTimes[idx];
        }
    }
    // ── Per-window throughput ──

    /// @brief Factory completion throughput over one fixed time window. One row in throughput.csv.
    ///
    /// @details A "completion" is a job reaching JobState.Exited. JobsCompleted counts exits
    ///          inside [WindowStartTime, WindowEndTime); CumulativeCompleted is the running total.
    ///          WorkInProgress is jobs in the system (not Exited) at window close — pairing it with
    ///          throughput gives a Little's-Law read on whether the floor is filling or draining.
    public class ThroughputWindowRecord
    {
        public double WindowStartTime;      // sim-seconds, inclusive
        public double WindowEndTime;        // sim-seconds, exclusive (== SimTime for the trailing partial window)
        public float WindowSeconds;        // WindowEndTime − WindowStartTime (== window length except trailing)
        public int JobsCompleted;        // exits during this window
        public int CumulativeCompleted;  // exits since episode start
        public int WorkInProgress;       // jobs in system at window close

        // Derived
        public float ThroughputPerSec => WindowSeconds > 0f ? JobsCompleted / WindowSeconds : 0f;
        public float ThroughputPerMin => ThroughputPerSec * 60f;
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
        public int StallRecoveryCount;   // HandleZoneStall calls (suspected deadlock self-recoveries)

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

        /// @brief Actual AGV transit time (sim-seconds) for this operation.
        /// @details Stamped by FlagHarvester on delivery: job.OperationTravelTimes[CurrentOpIndex] = agv.LastTripDuration.
        /// Zero for operations where transit was not completed (e.g. first op from incoming belt).
        public float TravelTime;

        // ── Realized timeline (Option A) ────────────────────────────────────
        /// @brief Sim-time this op's job entered Queued state at its target machine. -1 if never reached.
        public float QueueEntryTime = -1f;
        /// @brief Sim-time this op began Processing. -1 if never started.
        public float ProcStartTime = -1f;
        /// @brief Sim-time this op finished Processing. -1 if never completed.
        public float ProcEndTime = -1f;

        // Derived
        public float ProcTimeSpread => MaxProcTime - MinProcTime;
        /// @brief Realized processing duration (ProcEndTime - ProcStartTime). -1 if op never completed.
        public float RealizedProcTime => (ProcStartTime >= 0 && ProcEndTime >= 0) ? ProcEndTime - ProcStartTime : -1f;
        /// @brief Time this op's job spent queued at the machine before dispatch. -1 if not applicable.
        public float QueueWaitTime => (QueueEntryTime >= 0 && ProcStartTime >= 0) ? ProcStartTime - QueueEntryTime : -1f;
    }

    /// @brief One record per routing or dispatch decision, logged to decision_log.csv.
    ///
    /// @details Added to directly test whether DispatchingEngine's rule logic ever actually
    ///          runs (vs. its queue.Count<=1 / candidates.Length<=1 early-exit firing first),
    ///          and whether different rules pick different candidates given the same options.
    ///          Candidate arrays are '|'-joined parallel lists -- reconstruct in analysis rather
    ///          than pre-deciding which stat "mattered" in C#.
    ///
    ///          Routing: CandidateIds = eligible machine IDs, ChosenId = the machine picked,
    ///          SubjectId = the job being routed. CandidateStatA = per-candidate job processing
    ///          time (CandidateJobTimes), CandidateStatB = per-candidate queue length
    ///          (CandidateQueueLengths), CandidateStatC unused.
    ///
    ///          Dispatch: CandidateIds = queued job IDs, ChosenId = the job picked, SubjectId =
    ///          the machine making the decision. CandidateStatA = per-candidate processing
    ///          duration (QueuedDurations), CandidateStatB = per-candidate total remaining work
    ///          (DispatchingEngine.GetRemainingWork), CandidateStatC = per-candidate arrival time
    ///          (for reconstructing SDT's wait-time = simTime - arrival).
    public class DecisionRecord
    {
        public double SimTime;
        public int DecisionIndex;
        public bool IsRouting;   // true = Routing, false = Dispatch
        public int SubjectId;    // Routing: job being routed. Dispatch: machine deciding.
        public int ChosenId;     // Routing: chosen machine ID. Dispatch: chosen job ID.
        public int CandidateCount;
        public bool IsDegenerate; // true if CandidateCount <= 1 (DispatchingEngine short-circuited, rule never ran)
        public string CandidateIds = "";
        public string CandidateStatA = "";
        public string CandidateStatB = "";
        public string CandidateStatC = "";

        // Routing rows only: which jobs were simultaneously competing for THIS routing slot
        // (DecisionCoordinator.SelectRoutingJobId / DispatchingEngine.SelectRoutingJob).
        // JobCandidateCount <= 1 means job-priority selection was itself degenerate (only one
        // job ready) -- separate from CandidateCount/IsDegenerate above, which describe the
        // MACHINE choice for whichever job SubjectId ended up being.
        public int JobCandidateCount;
        public bool IsJobSelectionDegenerate;
        public string JobCandidateIds = "";
    }

    // ── Per-job completion record ──

    /// @brief One record per job, logged to job_completions.csv. The realized-outcome
    ///        counterpart to job_operations.csv (which only records the static plan).
    ///
    /// @details Captures arrival→exit flow time and its full wait decomposition across the five
    ///          mutually-exclusive job states, so per-rule differences that don't show up in
    ///          makespan (e.g. how much of a job's life is spent waiting for transport) become
    ///          visible. Jobs still in the system at episode end (timeout) are logged with
    ///          Completed=false — include them for censoring-aware analysis, exclude them from
    ///          mean/percentile flow-time stats.
    public class JobCompletionRecord
    {
        public int JobId;
        public bool IsDynamic;
        public bool Completed;          // false if episode ended (timeout) before this job exited
        public float ArrivalTime;
        public float ExitTime;           // -1 if not completed
        public int TotalOperations;
        public int CompletedOps;

        /// @brief Sum of realized processing durations for completed ops; for censored jobs,
        ///        falls back to sum of per-op mean proc time (an estimate, not realized).
        public float WorkContent;

        // ── Wait decomposition (sim-seconds), mirrors JobData's per-state buckets ──
        public double TimeNeedsRouting;
        public double TimeWaitingPickup;
        public double TimeInTransit;
        public double TimeQueued;
        public double TimeProcessingState;

        // Derived
        public float FlowTime => Completed ? ExitTime - ArrivalTime : -1f;
    }
}