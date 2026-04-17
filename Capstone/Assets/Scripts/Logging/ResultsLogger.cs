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
    public static class ResultsLogger
    {
        public static string OutputDirectory = "";

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

        /// @brief Computes the full absolute path for the log file.
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
    }
}