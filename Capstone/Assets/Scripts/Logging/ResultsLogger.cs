using System;
using System.IO;
using UnityEngine;

namespace Assets.Scripts.Logging
{
    /// @brief Static utility for persisting simulation performance metrics to CSV.
    ///
    /// @details Manages file I/O for recording episode results, including makespan, 
    /// rewards, and environment configurations. Supports dynamic filename suffixes 
    /// to facilitate parallel headless runs.
    ///
    /// Two CSV files are maintained:
    ///   - @c baseline_results.csv        — one row per episode (unchanged schema)
    ///   - @c machine_utilization.csv     — one row per machine per episode
    public static class ResultsLogger
    {
        public static string OutputDirectory = "";

        // ── Episode-level log ─────────────────────────────────────────────────

        private static string _filename = "baseline_results.csv";

        /// @brief Appends a suffix to the results filename before the extension.
        ///
        /// @param suffix The string to append (e.g., "_SPT").
        ///
        /// @details Strips the @c .csv extension, applies the suffix, and re-appends 
        /// the extension. This is used by @c HeadlessBatchRunner to ensure parallel 
        /// workers do not encounter file lock contention.
        public static void SetFilenameSuffix(string suffix)
        {
            const string ext = ".csv";
            string baseName = _filename.EndsWith(ext)
                ? _filename[..^ext.Length]
                : _filename;
            _filename = baseName + suffix + ext;

            // Apply the same suffix to the machine utilization file so both files
            // from the same worker stay co-located and identifiable.
            string machineBase = _machineFilename.EndsWith(ext)
                ? _machineFilename[..^ext.Length]
                : _machineFilename;
            _machineFilename = machineBase + suffix + ext;
        }

        /// <summary>
        /// Nests results inside a subdirectory under the current OutputDirectory.
        /// e.g. if OutputDirectory is "Results/" and subdir is "brandimarte",
        /// results will go to "Results/brandimarte/".
        /// Creates the directory if it doesn't exist.
        /// </summary>
        public static void SetSubdirectory(string subdir)
        {
            if (string.IsNullOrEmpty(subdir)) return;
            OutputDirectory = Path.Combine(OutputDirectory, subdir);
            Directory.CreateDirectory(OutputDirectory);
        }

        /// @brief Computes the full absolute path for the episode log file.
        ///
        /// @details Returns @c OutputDirectory if defined; otherwise, defaults to 
        /// @c Application.persistentDataPath.
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
        /// @param ruleName The identifier for the scheduling rule or policy used.
        /// @param seed The random seed used to generate the environment.
        /// @param makespan The total time taken to complete all jobs.
        /// @param jobCount Total number of jobs in the problem instance.
        /// @param machineCount Total number of machines in the shop floor.
        /// @param totalOps Total count of operations processed.
        /// @param decisionCount Number of scheduling decisions made by the agent/rule.
        /// @param totalReward The cumulative reward achieved during the episode.
        /// @param averageTimeScale The mean simulation speed maintained during the run.
        /// @param agvCount Total number of AGVs in the fleet.
        ///
        /// @details Thread-safe for sequential writes. If the target file does not 
        /// exist, this method automatically writes the CSV header before appending data.
        public static void LogEpisode(string ruleName, int seed, double makespan,
                                       int jobCount, int machineCount, int totalOps,
                                       int decisionCount, double totalReward, float averageTimeScale, int agvCount)
        {
            bool fileExists = File.Exists(FilePath);
            using StreamWriter writer = new StreamWriter(FilePath, append: true);

            if (!fileExists)
                writer.WriteLine("timestamp,rule,seed,makespan,jobs,machines,total_ops,agvCount,decisions,total_reward,timescale");

            writer.WriteLine(
                $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}," +
                $"{ruleName}," +
                $"{seed}," +
                $"{makespan:F2}," +
                $"{jobCount}," +
                $"{machineCount}," +
                $"{totalOps}," +
                $"{agvCount}," +
                $"{decisionCount}," +
                $"{totalReward:F4}," +
                $"{averageTimeScale:F4}"
            );

            Debug.Log($"[Results] Logged: {ruleName} seed={seed} makespan={makespan:F1} - {FilePath}");
        }

        // ── Machine-level utilization log ─────────────────────────────────────

        private static string _machineFilename = "machine_utilization.csv";

        /// @brief Computes the full absolute path for the machine utilization log file.
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
        /// @param ruleName       The scheduling rule or policy used this episode.
        /// @param seed           The random seed used to generate the environment.
        /// @param makespan       The total episode duration (wall-clock SimTime).
        /// @param machineId      Unique identifier of the machine being logged.
        /// @param machineType    String name of the machine's @c MachineType enum value.
        /// @param opsCompleted   Number of operations successfully completed by this machine.
        /// @param timeProcessing Cumulative SimTime seconds this machine spent actively processing.
        /// @param timeOperational SimTime seconds this machine was in the Operational health state
        ///                       (equals @p makespan in deterministic runs; reduced by repair
        ///                       downtime in stochastic runs).
        ///
        /// @details Derived columns written to CSV:
        ///   - @c utilization_rate = @p timeProcessing / @p timeOperational
        ///   - @c idle_time        = @p timeOperational - @p timeProcessing
        ///   - @c idle_rate        = @c idle_time / @p timeOperational
        ///
        /// One row per machine is expected per episode. Call this in a loop over all
        /// machines immediately after @c LogEpisode inside @c FinaliseEpisode().
        ///
        /// The file schema is fixed regardless of machine count, making it trivially
        /// joinable to the episode log on (rule, seed).
        public static void LogMachineUtilization(
            string ruleName, int seed, double makespan,
            int machineId, string machineType,
            int opsCompleted, double timeProcessing, double timeOperational)
        {
            // Guard against divide-by-zero in pathological zero-length episodes.
            double utilizationRate = timeOperational > 0.0 ? timeProcessing / timeOperational : 0.0;
            double idleTime = timeOperational - timeProcessing;
            double idleRate = timeOperational > 0.0 ? idleTime / timeOperational : 0.0;

            bool fileExists = File.Exists(MachineFilePath);
            using StreamWriter writer = new StreamWriter(MachineFilePath, append: true);

            if (!fileExists)
                writer.WriteLine(
                    "timestamp,rule,seed,makespan," +
                    "machine_id,machine_type,ops_completed," +
                    "time_processing,time_operational," +
                    "utilization_rate,idle_time,idle_rate");

            writer.WriteLine(
                $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}," +
                $"{ruleName},{seed},{makespan:F2}," +
                $"{machineId},{machineType},{opsCompleted}," +
                $"{timeProcessing:F2},{timeOperational:F2}," +
                $"{utilizationRate:F4},{idleTime:F2},{idleRate:F4}"
            );
        }
    }
}