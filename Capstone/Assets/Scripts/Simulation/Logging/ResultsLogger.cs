using System;
using System.IO;
using UnityEngine;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation.Logging
{
    /// <summary>
    /// Serialises EpisodeRecord objects to CSV.
    ///
    /// results.csv               — one row per episode              (LogEpisode)
    /// machine_utilization.csv   — one row per machine per episode   (LogMachineUtilization)
    /// agv_performance.csv       — one row per AGV per episode       (LogAGVPerformance)
    /// segment_congestion.csv    — one row per zone per episode      (LogSegmentCongestion)
    /// job_operations.csv        — one row per (job × op), static plan + realized timestamps (LogJobOperations)
    /// throughput.csv            — one row per closed throughput window (LogThroughput)
    /// job_completions.csv       — one row per job, realized flow time + wait decomposition (LogJobCompletions)
    /// decision_log.csv          — one row per routing/dispatch decision, incl. degenerate cases (LogDecisions)
    ///
    /// All four share the same filename suffix set by SetFilenameSuffix so output
    /// from different run configurations lands in clearly named files:
    ///   results_det.csv / results_low.csv, etc.
    /// </summary>
    public static class ResultsLogger
    {
        public static string OutputDirectory = "";

        // ── Filename management ───────────────────────────────────────────────

        private static string _filename = "results.csv";
        private static string _machineFilename = "machine_utilization.csv";
        private static string _agvFilename = "agv_performance.csv";
        private static string _segmentFilename = "segment_congestion.csv";
        private static string _jobOpsFilename = "job_operations.csv";
        private static string _throughputFilename = "throughput.csv";
        private static string _jobCompletionsFilename = "job_completions.csv";
        private static string _decisionLogFilename = "decision_log.csv";

        public static void SetFilenameSuffix(string suffix)
        {
            const string ext = ".csv";
            _filename = StripExt(_filename, ext) + suffix + ext;
            _machineFilename = StripExt(_machineFilename, ext) + suffix + ext;
            _agvFilename = StripExt(_agvFilename, ext) + suffix + ext;
            _segmentFilename = StripExt(_segmentFilename, ext) + suffix + ext;
            _jobOpsFilename = StripExt(_jobOpsFilename, ext) + suffix + ext;
            _throughputFilename = StripExt(_throughputFilename, ext) + suffix + ext;
            _jobCompletionsFilename = StripExt(_jobCompletionsFilename, ext) + suffix + ext;
            _decisionLogFilename = StripExt(_decisionLogFilename, ext) + suffix + ext;
        }

        public static void SetSubdirectory(string subdir)
        {
            if (string.IsNullOrEmpty(subdir)) return;
            OutputDirectory = Path.Combine(OutputDirectory, subdir);
            Directory.CreateDirectory(OutputDirectory);
        }

        private static string FilePath => BuildPath(_filename);
        private static string MachineFilePath => BuildPath(_machineFilename);
        private static string AGVFilePath => BuildPath(_agvFilename);
        private static string SegmentFilePath => BuildPath(_segmentFilename);
        private static string JobOpsFilePath => BuildPath(_jobOpsFilename);
        private static string ThroughputFilePath => BuildPath(_throughputFilename);
        private static string JobCompletionsFilePath => BuildPath(_jobCompletionsFilename);
        private static string DecisionLogFilePath => BuildPath(_decisionLogFilename);

        // ── Convenience: write all logs in one call ───────────────────────────

        /// <summary>
        /// Standard call-site in HeadlessBatchRunner — writes all four CSVs.
        /// </summary>
        public static void LogAll(EpisodeRecord r)
        {
            LogEpisode(r);
            LogMachineUtilization(r);
            if (r.AGVRecords.Count > 0) LogAGVPerformance(r);
            if (r.SegmentRecords.Count > 0) LogSegmentCongestion(r);
            if (r.JobOperationRecords.Count > 0) LogJobOperations(r);
            if (r.ThroughputRecords.Count > 0) LogThroughput(r);
            if (r.JobCompletionRecords.Count > 0) LogJobCompletions(r);
            if (r.DecisionRecords.Count > 0) LogDecisions(r);
        }
        // ── Throughput log (throughput.csv) ───────────────────────────────────

        /// <summary>
        /// Appends one row per closed throughput window. Header written on first call.
        ///
        /// Key columns:
        ///   throughput_per_min — completions normalised to jobs/min (comparable across window sizes)
        ///   work_in_progress   — jobs in system at window close; pair with throughput for Little's Law
        /// </summary>
        public static void LogThroughput(EpisodeRecord r)
        {
            bool fileExists = File.Exists(ThroughputFilePath);
            using var writer = new StreamWriter(ThroughputFilePath, append: true);

            if (!fileExists)
                writer.WriteLine(
                    "timestamp,instance,rule,seed,makespan," +
                    "window_start,window_end,window_seconds," +
                    "jobs_completed,throughput_per_sec,throughput_per_min," +
                    "cumulative_completed,work_in_progress"
                );

            string ts = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            foreach (var w in r.ThroughputRecords)
            {
                writer.WriteLine(
                    $"{ts}," +
                    $"{r.InstanceName},{r.RuleName},{r.Seed},{r.Makespan:F2}," +
                    $"{w.WindowStartTime:F1},{w.WindowEndTime:F1},{w.WindowSeconds:F1}," +
                    $"{w.JobsCompleted},{w.ThroughputPerSec:F5},{w.ThroughputPerMin:F3}," +
                    $"{w.CumulativeCompleted},{w.WorkInProgress}"
                );
            }
        }

        // ── Episode log (results.csv) ─────────────────────────────────────────

        /// <summary>
        /// Appends one episode row. Header written automatically on first call.
        /// </summary>
        public static void LogEpisode(EpisodeRecord r)
        {
            bool hasMf = r.Stochastic != null && r.Stochastic.MachineFailuresEnabled;
            float weibullK = hasMf ? r.Stochastic.WeibullK : 0f;
            float weibullLambda = hasMf ? r.Stochastic.WeibullLambda : 0f;
            float meanTtfTheory = weibullLambda > 0f ? weibullLambda * 0.9027f : 0f;
            float repairLogMu = hasMf ? r.Stochastic.RepairLogMu : 0f;
            float repairLogSig = hasMf ? r.Stochastic.RepairLogSigma : 0f;

            bool fileExists = File.Exists(FilePath);
            using var writer = new StreamWriter(FilePath, append: true);

            if (!fileExists)
                writer.WriteLine(
                    "timestamp,instance,rule,seed,makespan," +
                    "jobs,machines,total_ops,agvCount,decisions,total_reward,timescale," +
                    "parking_method,pre_dispatching_method," +
                    "stochastic_tag,weibull_k,weibull_lambda,mean_ttf_theoretical," +
                    "repair_log_mu,repair_log_sigma," +
                    "episode_failures,total_repair_time," +
                    "dynamic_arrivals,arrival_lambda,mean_interarrival_theoretical," +
                    "mean_interarrival_realised,last_arrival_sim_time," +
                    "mean_flow_time,p95_flow_time,max_flow_time,mean_transport_wait,jobs_censored," +
                    "deadlock_detected,deadlock_sim_time"
                );

            writer.WriteLine(
                $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}," +
                $"{r.InstanceName},{r.RuleName},{r.Seed},{r.Makespan:F2}," +
                $"{r.JobCount},{r.MachineCount},{r.TotalOperations},{r.AGVCount}," +
                $"{r.DecisionPoints},{r.TotalReward:F4},{r.AverageTimeScale:F4}," +
                $"{r.ParkingMethod},{r.PreDispatchingMethod}," +
                $"{r.StochasticTag},{weibullK:F2},{weibullLambda:F1},{meanTtfTheory:F1}," +
                $"{repairLogMu:F3},{repairLogSig:F3}," +
                $"{r.MachineFailureCount},{r.MachineRepairTime:F1}," +
                $"{r.DynamicArrivals},{r.ArrivalLambda:F5},{r.MeanInterarrivalTime:F1}," +
                $"{r.RealisedMeanInterarrival:F1},{r.LastDynamicArrivalTime:F1}," +
                $"{r.MeanFlowTime:F2},{r.P95FlowTime:F2},{r.MaxFlowTime:F2},{r.MeanTransportWait:F2},{r.JobsCensored}," +
                $"{(r.DeadlockDetected ? 1 : 0)},{r.DeadlockSimTime:F1}"
            );

            Debug.Log($"[Results] {r.InstanceName} {r.RuleName} seed={r.Seed} " +
                      $"makespan={r.Makespan:F1} stochastic={r.StochasticTag} " +
                      $"failures={r.MachineFailureCount}");
        }

        // ── Machine utilization log ───────────────────────────────────────────

        /// <summary>
        /// Appends one row per machine. All rows share the same timestamp.
        /// </summary>
        public static void LogMachineUtilization(EpisodeRecord r)
        {
            bool fileExists = File.Exists(MachineFilePath);
            using var writer = new StreamWriter(MachineFilePath, append: true);

            if (!fileExists)
                writer.WriteLine(
                    "timestamp,instance,rule,seed,makespan," +
                    "machine_id,machine_type,ops_completed," +
                    "time_processing,time_operational," +
                    "utilization_rate,idle_time,idle_rate,availability_rate," +
                    "failure_count,total_repair_time"
                );

            string ts = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            foreach (var m in r.MachineRecords)
            {
                writer.WriteLine(
                    $"{ts}," +
                    $"{r.InstanceName},{r.RuleName},{r.Seed},{r.Makespan:F2}," +
                    $"{m.MachineId},{m.MachineType},{m.OpsCompleted}," +
                    $"{m.TimeProcessing:F2},{m.TimeOperational:F2}," +
                    $"{m.UtilizationRate:F4},{m.IdleTime:F2},{m.IdleRate:F4},{m.AvailabilityRate:F4}," +
                    $"{m.FailureCount},{m.RepairTime:F1}"
                );
            }
        }

        // ── AGV performance log (new) ─────────────────────────────────────────

        /// <summary>
        /// Appends one row per AGV per episode to agv_performance.csv.
        ///
        /// Key columns for diagnosing the AGV-bottleneck / congestion question:
        ///   time_waiting_route  — blocked by quadrant reservation (congestion signal)
        ///   time_idle           — parked with no work (over-provisioning signal)
        ///   congestion_fraction — time_waiting_route / total_accounted_time
        /// </summary>
        public static void LogAGVPerformance(EpisodeRecord r)
        {
            bool fileExists = File.Exists(AGVFilePath);
            using var writer = new StreamWriter(AGVFilePath, append: true);

            if (!fileExists)
                writer.WriteLine(
                    "timestamp,instance,rule,seed,makespan," +
                    "agv_id,total_trips,mean_trip_duration," +
                    "time_idle,time_waiting_route,time_traveling," +
                    "time_loading,time_unloading," +
                    "total_path_length,reroute_count,congestion_fraction,stall_recovery_count"
                );

            string ts = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            foreach (var a in r.AGVRecords)
            {
                writer.WriteLine(
                    $"{ts}," +
                    $"{r.InstanceName},{r.RuleName},{r.Seed},{r.Makespan:F2}," +
                    $"{a.AgvId},{a.TotalTrips},{a.MeanTripDuration:F2}," +
                    $"{a.TimeIdle:F2},{a.TimeWaitingRoute:F2},{a.TimeTraveling:F2}," +
                    $"{a.TimeLoading:F2},{a.TimeUnloading:F2}," +
                    $"{a.TotalPathLength:F2},{a.RerouteCount},{a.CongestionFraction:F4},{a.StallRecoveryCount}"
                );
            }
        }

        // ── Segment congestion log (new) ──────────────────────────────────────

        /// <summary>
        /// Appends one row per TrafficZone per episode to segment_congestion.csv.
        ///
        /// Key columns:
        ///   block_rate        — BlockEvents / (TraversalCount + BlockEvents)
        ///   mean_block_time   — how long each blockage lasts on average
        ///
        /// Sort by block_rate descending to find congestion hotspots.
        /// Zones whose name contains "RowAisle" and sit adjacent to spine
        /// intersections are expected hotspots given the unidirectional layout.
        /// </summary>
        public static void LogSegmentCongestion(EpisodeRecord r)
        {
            bool fileExists = File.Exists(SegmentFilePath);
            using var writer = new StreamWriter(SegmentFilePath, append: true);

            if (!fileExists)
                writer.WriteLine(
                    "timestamp,instance,rule,seed,makespan," +
                    "zone_id,zone_name,aisle_type,flow_direction," +
                    "traversal_count,block_events,total_block_time," +
                    "mean_block_time,block_rate"
                );

            string ts = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            foreach (var s in r.SegmentRecords)
            {
                writer.WriteLine(
                    $"{ts}," +
                    $"{r.InstanceName},{r.RuleName},{r.Seed},{r.Makespan:F2}," +
                    $"{s.ZoneId},{s.ZoneName},{s.AisleType},{s.FlowDirection}," +
                    $"{s.TraversalCount},{s.BlockEvents},{s.TotalBlockTime:F2}," +
                    $"{s.MeanBlockTime:F2},{s.BlockRate:F4}"
                );
            }
        }

        // ── Job operations log (job_operations.csv) ───────────────────────────

        /// <summary>
        /// Appends one row per (job × operation) to job_operations.csv.
        /// Populate EpisodeRecord.JobOperationRecords in EpisodeTracker at episode end
        /// by iterating jobStore.AllJobs and computing min/max/mean over EligibleMachinesPerOp[i].Values.
        /// </summary>
        public static void LogJobOperations(EpisodeRecord r)
        {
            bool fileExists = File.Exists(JobOpsFilePath);
            using var writer = new StreamWriter(JobOpsFilePath, append: true);

            if (!fileExists)
                writer.WriteLine(
                    "timestamp,instance,rule,seed,makespan," +
                    "job_id,is_dynamic,arrival_time," +
                    "op_index,machine_type_required," +
                    "eligible_machine_count," +
                    "min_proc_time,max_proc_time,mean_proc_time,proc_time_spread," +
                    "travel_time," +
                    "queue_entry_time,proc_start_time,proc_end_time,realized_proc_time,queue_wait_time"
                );

            string ts = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            foreach (var op in r.JobOperationRecords)
            {
                writer.WriteLine(
                    $"{ts}," +
                    $"{r.InstanceName},{r.RuleName},{r.Seed},{r.Makespan:F2}," +
                    $"{op.JobId},{(op.IsDynamic ? 1 : 0)},{op.ArrivalTime:F1}," +
                    $"{op.OpIndex},{op.MachineTypeRequired}," +
                    $"{op.EligibleMachineCount}," +
                    $"{op.MinProcTime:F1},{op.MaxProcTime:F1},{op.MeanProcTime:F1},{op.ProcTimeSpread:F1}," +
                    $"{op.TravelTime:F1}," +
                    $"{op.QueueEntryTime:F1},{op.ProcStartTime:F1},{op.ProcEndTime:F1}," +
                    $"{op.RealizedProcTime:F1},{op.QueueWaitTime:F1}"
                );
            }
        }

        // ── Decision log (decision_log.csv) ───────────────────────────────────

        /// <summary>
        /// Appends one row per routing/dispatch decision to decision_log.csv, including the
        /// candidate_count &lt;= 1 degenerate cases where DispatchingEngine.SelectMachine/SelectJob
        /// short-circuit before the active rule ever evaluates a candidate. Candidate_ids and the
        /// per-candidate stat columns are '|'-joined parallel lists (empty string if no candidates).
        /// See DecisionRecord for which stat is which for Routing vs Dispatch rows.
        /// </summary>
        public static void LogDecisions(EpisodeRecord r)
        {
            bool fileExists = File.Exists(DecisionLogFilePath);
            using var writer = new StreamWriter(DecisionLogFilePath, append: true);

            if (!fileExists)
                writer.WriteLine(
                    "timestamp,instance,rule,seed,makespan," +
                    "sim_time,decision_index,decision_type,subject_id,chosen_id," +
                    "candidate_count,is_degenerate," +
                    "candidate_ids,candidate_stat_a,candidate_stat_b,candidate_stat_c," +
                    "job_candidate_count,is_job_selection_degenerate,job_candidate_ids"
                );

            string ts = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            foreach (var d in r.DecisionRecords)
            {
                writer.WriteLine(
                    $"{ts}," +
                    $"{r.InstanceName},{r.RuleName},{r.Seed},{r.Makespan:F2}," +
                    $"{d.SimTime:F1},{d.DecisionIndex},{(d.IsRouting ? "Routing" : "Dispatch")}," +
                    $"{d.SubjectId},{d.ChosenId}," +
                    $"{d.CandidateCount},{(d.IsDegenerate ? 1 : 0)}," +
                    $"{d.CandidateIds},{d.CandidateStatA},{d.CandidateStatB},{d.CandidateStatC}," +
                    $"{d.JobCandidateCount},{(d.IsJobSelectionDegenerate ? 1 : 0)},{d.JobCandidateIds}"
                );
            }
        }

        // ── Job completions log (job_completions.csv) ─────────────────────────

        /// <summary>
        /// Appends one row per job to job_completions.csv — the realized-outcome counterpart
        /// to job_operations.csv. Includes censored (incomplete) jobs with completed=0 so
        /// timeout episodes can be analyzed without silently dropping unfinished work.
        ///
        /// Key columns:
        ///   flow_time              — exit_time - arrival_time (-1 if not completed)
        ///   time_needs_routing/... — wait decomposition across the five job states;
        ///                            sum of all five ≈ flow_time for completed jobs
        /// </summary>
        public static void LogJobCompletions(EpisodeRecord r)
        {
            bool fileExists = File.Exists(JobCompletionsFilePath);
            using var writer = new StreamWriter(JobCompletionsFilePath, append: true);

            if (!fileExists)
                writer.WriteLine(
                    "timestamp,instance,rule,seed,makespan," +
                    "job_id,is_dynamic,completed,arrival_time,exit_time,flow_time," +
                    "total_operations,completed_ops,work_content," +
                    "time_needs_routing,time_waiting_pickup,time_in_transit,time_queued,time_processing"
                );

            string ts = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            foreach (var j in r.JobCompletionRecords)
            {
                writer.WriteLine(
                    $"{ts}," +
                    $"{r.InstanceName},{r.RuleName},{r.Seed},{r.Makespan:F2}," +
                    $"{j.JobId},{(j.IsDynamic ? 1 : 0)},{(j.Completed ? 1 : 0)}," +
                    $"{j.ArrivalTime:F1},{j.ExitTime:F1},{j.FlowTime:F1}," +
                    $"{j.TotalOperations},{j.CompletedOps},{j.WorkContent:F1}," +
                    $"{j.TimeNeedsRouting:F1},{j.TimeWaitingPickup:F1},{j.TimeInTransit:F1}," +
                    $"{j.TimeQueued:F1},{j.TimeProcessingState:F1}"
                );
            }
        }

        // ── Private helpers ───────────────────────────────────────────────────

        private static string BuildPath(string filename)
        {
            string dir = string.IsNullOrEmpty(OutputDirectory)
                ? Application.persistentDataPath
                : OutputDirectory;
            return Path.Combine(dir, filename);
        }

        private static string StripExt(string name, string ext) =>
            name.EndsWith(ext) ? name[..^ext.Length] : name;
    }
}