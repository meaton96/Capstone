using System;
using System.IO;
using UnityEngine;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation.Logging
{
    /// <summary>
    /// Serialises EpisodeRecord objects to CSV.
    ///
    /// Four files:
    ///   results.csv               — one row per episode            (LogEpisode)
    ///   machine_utilization.csv   — one row per machine per episode (LogMachineUtilization)
    ///   agv_performance.csv       — one row per AGV per episode     (LogAGVPerformance)
    ///   segment_congestion.csv    — one row per zone per episode    (LogSegmentCongestion)
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

        public static void SetFilenameSuffix(string suffix)
        {
            const string ext = ".csv";
            _filename = StripExt(_filename, ext) + suffix + ext;
            _machineFilename = StripExt(_machineFilename, ext) + suffix + ext;
            _agvFilename = StripExt(_agvFilename, ext) + suffix + ext;
            _segmentFilename = StripExt(_segmentFilename, ext) + suffix + ext;
            _jobOpsFilename = StripExt(_jobOpsFilename, ext) + suffix + ext;
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
                    "mean_interarrival_realised,last_arrival_sim_time"
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
                $"{r.RealisedMeanInterarrival:F1},{r.LastDynamicArrivalTime:F1}"
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
                    "total_path_length,reroute_count,congestion_fraction"
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
                    $"{a.TotalPathLength:F2},{a.RerouteCount},{a.CongestionFraction:F4}"
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
                    "travel_time"
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
                    $"{op.TravelTime:F1}"
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