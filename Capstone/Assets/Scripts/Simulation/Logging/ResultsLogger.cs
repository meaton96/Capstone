using System;
using System.IO;
using UnityEngine;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation.Logging
{
    /// @brief Static utility for persisting simulation performance metrics to CSV.
    ///
    /// @details Two CSV files are maintained:
    ///   - baseline_results.csv      — one row per episode
    ///   - machine_utilization.csv   — one row per machine per episode
    ///
    /// Stochastic columns are always written. For deterministic runs they contain
    /// "none" / 0 defaults so deterministic and stochastic results share one schema
    /// and can be filtered/joined without a Python migration step.
    public static class ResultsLogger
    {
        public static string OutputDirectory = "";

        // ── Episode-level log ─────────────────────────────────────────────────

        private static string _filename = "baseline_results.csv";

        public static void SetFilenameSuffix(string suffix)
        {
            const string ext = ".csv";
            string baseName = _filename.EndsWith(ext) ? _filename[..^ext.Length] : _filename;
            _filename = baseName + suffix + ext;

            string machineBase = _machineFilename.EndsWith(ext)
                ? _machineFilename[..^ext.Length]
                : _machineFilename;
            _machineFilename = machineBase + suffix + ext;
        }

        public static void SetSubdirectory(string subdir)
        {
            if (string.IsNullOrEmpty(subdir)) return;
            OutputDirectory = Path.Combine(OutputDirectory, subdir);
            Directory.CreateDirectory(OutputDirectory);
        }

        private static string FilePath
        {
            get
            {
                string dir = string.IsNullOrEmpty(OutputDirectory)
                    ? Application.persistentDataPath
                    : OutputDirectory;
                return Path.Combine(dir, _filename);
            }
        }

        /// @brief Records the results of a single simulation episode to the CSV.
        ///
        /// @param instanceName     Config or benchmark name (e.g. "MK03", "rand_5j_5m").
        /// @param ruleName         Scheduling rule or policy identifier.
        /// @param seed             Random seed used for this episode.
        /// @param makespan         Total time to complete all jobs.
        /// @param jobCount         Number of jobs in the instance.
        /// @param machineCount     Number of machines on the floor.
        /// @param totalOps         Total operations across all jobs.
        /// @param decisionCount    Number of scheduling decisions made.
        /// @param totalReward      Cumulative reward.
        /// @param averageTimeScale Mean simulation speed multiplier.
        /// @param agvCount         Fleet size.
        /// @param stochastic       Optional stochastic config for this episode.
        ///                         Pass null for deterministic runs — defaults fill in.
        /// @param episodeFailures  Total machine failures that occurred. 0 if deterministic.
        /// @param totalRepairTime  Cumulative repair downtime (sim-seconds). 0 if deterministic.
        public static void LogEpisode(
            string instanceName, string ruleName, int seed, double makespan,
            int jobCount, int machineCount, int totalOps,
            int decisionCount, double totalReward, float averageTimeScale, int agvCount,
            StochasticConfig stochastic = null,
            int episodeFailures = 0,
            float totalRepairTime = 0f)
        {
            // ── Stochastic column values ──────────────────────────────────────
            bool hasStochastic = stochastic != null && stochastic.AnyEnabled;
            string stochasticTag = hasStochastic ? stochastic.Tag : "none";
            float weibullK = hasStochastic ? stochastic.WeibullK : 0f;
            float weibullLambda = hasStochastic && stochastic.MachineFailuresEnabled
                                       ? stochastic.WeibullLambda : 0f;
            float repairLogMu = hasStochastic && stochastic.MachineFailuresEnabled
                                       ? stochastic.RepairLogMu : 0f;
            float repairLogSigma = hasStochastic && stochastic.MachineFailuresEnabled
                                       ? stochastic.RepairLogSigma : 0f;

            // Derived: mean TTF from Weibull params (informational)
            // mean = lambda × Γ(1 + 1/k); at k=1.5, Γ(1.667) ≈ 0.9027
            float meanTtfTheoretical = weibullLambda > 0f ? weibullLambda * 0.9027f : 0f;

            bool fileExists = File.Exists(FilePath);
            using StreamWriter writer = new StreamWriter(FilePath, append: true);

            if (!fileExists)
                writer.WriteLine(
                    "timestamp,instance,rule,seed,makespan," +
                    "jobs,machines,total_ops,agvCount,decisions,total_reward,timescale," +
                    "stochastic_tag,weibull_k,weibull_lambda,mean_ttf_theoretical," +
                    "repair_log_mu,repair_log_sigma," +
                    "episode_failures,total_repair_time"
                );

            writer.WriteLine(
                $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}," +
                $"{instanceName},{ruleName},{seed},{makespan:F2}," +
                $"{jobCount},{machineCount},{totalOps},{agvCount}," +
                $"{decisionCount},{totalReward:F4},{averageTimeScale:F4}," +
                $"{stochasticTag},{weibullK:F2},{weibullLambda:F1},{meanTtfTheoretical:F1}," +
                $"{repairLogMu:F3},{repairLogSigma:F3}," +
                $"{episodeFailures},{totalRepairTime:F1}"
            );

            Debug.Log($"[Results] Logged: {instanceName} {ruleName} seed={seed} " +
                      $"makespan={makespan:F1} stochastic={stochasticTag} " +
                      $"failures={episodeFailures} - {FilePath}");
        }

        // ── Machine-level utilization log ─────────────────────────────────────

        private static string _machineFilename = "machine_utilization.csv";

        private static string MachineFilePath
        {
            get
            {
                string dir = string.IsNullOrEmpty(OutputDirectory)
                    ? Application.persistentDataPath
                    : OutputDirectory;
                return Path.Combine(dir, _machineFilename);
            }
        }

        /// @brief Records per-machine utilization statistics for a single episode.
        ///
        /// @param instanceName      Config or benchmark name.
        /// @param ruleName          Scheduling rule or policy used.
        /// @param seed              Random seed.
        /// @param makespan          Total episode duration.
        /// @param machineId         Unique machine identifier.
        /// @param machineType       MachineType enum name string.
        /// @param opsCompleted      Operations successfully completed by this machine.
        /// @param timeProcessing    Cumulative sim-seconds actively processing.
        /// @param timeOperational   Sim-seconds in Operational health state.
        ///                          Equals makespan in deterministic runs.
        ///                          Reduced by repair downtime in stochastic runs.
        /// @param failureCount      Failures on this specific machine this episode.
        /// @param totalRepairTime   Cumulative repair time for this machine this episode.
        public static void LogMachineUtilization(
            string instanceName, string ruleName, int seed, double makespan,
            int machineId, string machineType,
            int opsCompleted, double timeProcessing, double timeOperational,
            int failureCount = 0, float totalRepairTime = 0f)
        {
            double utilizationRate = timeOperational > 0.0 ? timeProcessing / timeOperational : 0.0;
            double idleTime = timeOperational - timeProcessing;
            double idleRate = timeOperational > 0.0 ? idleTime / timeOperational : 0.0;
            double availabilityRate = makespan > 0.0 ? timeOperational / makespan : 1.0;

            bool fileExists = File.Exists(MachineFilePath);
            using StreamWriter writer = new StreamWriter(MachineFilePath, append: true);

            if (!fileExists)
                writer.WriteLine(
                    "timestamp,instance,rule,seed,makespan," +
                    "machine_id,machine_type,ops_completed," +
                    "time_processing,time_operational," +
                    "utilization_rate,idle_time,idle_rate,availability_rate," +
                    "failure_count,total_repair_time"
                );

            writer.WriteLine(
                $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}," +
                $"{instanceName},{ruleName},{seed},{makespan:F2}," +
                $"{machineId},{machineType},{opsCompleted}," +
                $"{timeProcessing:F2},{timeOperational:F2}," +
                $"{utilizationRate:F4},{idleTime:F2},{idleRate:F4},{availabilityRate:F4}," +
                $"{failureCount},{totalRepairTime:F1}"
            );
        }
    }
}