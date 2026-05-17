using System;
using System.IO;
using UnityEngine;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation.Logging
{
    /// <summary>
    /// Serialises EpisodeRecord objects to CSV.
    ///
    /// All logging methods take an EpisodeRecord — no more long parameter lists.
    /// Adding a new stochastic event means adding a field to EpisodeRecord and
    /// a column here. Nothing else needs to change.
    ///
    /// Two files:
    ///   baseline_results.csv      — one row per episode (LogEpisode)
    ///   machine_utilization.csv   — one row per machine per episode (LogMachineUtilization)
    /// </summary>
    public static class ResultsLogger
    {
        public static string OutputDirectory = "";

        // ── Episode log ───────────────────────────────────────────────────────

        private static string _filename = "baseline_results.csv";

        public static void SetFilenameSuffix(string suffix)
        {
            const string ext = ".csv";
            _filename = StripExt(_filename, ext) + suffix + ext;
            _machineFilename = StripExt(_machineFilename, ext) + suffix + ext;
        }

        public static void SetSubdirectory(string subdir)
        {
            if (string.IsNullOrEmpty(subdir)) return;
            OutputDirectory = Path.Combine(OutputDirectory, subdir);
            Directory.CreateDirectory(OutputDirectory);
        }

        private static string FilePath => BuildPath(_filename);

        /// <summary>
        /// Appends one episode row to baseline_results.csv.
        /// Header is written automatically on first call.
        /// </summary>
        public static void LogEpisode(EpisodeRecord r)
        {
            bool hasMf = r.Stochastic != null && r.Stochastic.MachineFailuresEnabled;
            string stochasticTag = r.StochasticTag;
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
                    "stochastic_tag,weibull_k,weibull_lambda,mean_ttf_theoretical," +
                    "repair_log_mu,repair_log_sigma," +
                    "episode_failures,total_repair_time"
                // Phase 3: append ",agv_failures,agv_repair_time" here
                // Phase 4: append ",dynamic_arrivals" here
                );

            writer.WriteLine(
                $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}," +
                $"{r.InstanceName},{r.RuleName},{r.Seed},{r.Makespan:F2}," +
                $"{r.JobCount},{r.MachineCount},{r.TotalOperations},{r.AGVCount}," +
                $"{r.DecisionPoints},{r.TotalReward:F4},{r.AverageTimeScale:F4}," +
                $"{stochasticTag},{weibullK:F2},{weibullLambda:F1},{meanTtfTheory:F1}," +
                $"{repairLogMu:F3},{repairLogSig:F3}," +
                $"{r.MachineFailureCount},{r.MachineRepairTime:F1}"
            );

            Debug.Log($"[Results] {r.InstanceName} {r.RuleName} seed={r.Seed} " +
                      $"makespan={r.Makespan:F1} stochastic={stochasticTag} " +
                      $"failures={r.MachineFailureCount}");
        }

        // ── Machine utilization log ───────────────────────────────────────────

        private static string _machineFilename = "machine_utilization.csv";
        private static string MachineFilePath => BuildPath(_machineFilename);

        /// <summary>
        /// Appends one row per machine in the EpisodeRecord to machine_utilization.csv.
        /// All machine rows share the same timestamp so the episode is atomic in the file.
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

        /// <summary>
        /// Logs both episode and machine rows in one call — standard call-site
        /// in HeadlessBatchRunner.
        /// </summary>
        public static void LogAll(EpisodeRecord r)
        {
            LogEpisode(r);
            LogMachineUtilization(r);
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