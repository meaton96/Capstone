using System;
using System.IO;
using UnityEngine;

namespace Assets.Scripts.Logging
{
    public static class ResultsLogger
    {
        // Set by LoggingInitializer on Awake.
        public static string OutputDirectory = "";

        // Base filename — overridden by SetFilenameSuffix() when running
        // parallel headless workers so each process writes its own CSV.
        private static string _filename = "baseline_results.csv";

        /// <summary>
        /// Called by HeadlessBatchRunner when -outputsuffix is passed on the CLI.
        /// Example: suffix "_SPT_SMPT" → "baseline_results_SPT_SMPT.csv"
        /// The parallel launcher merges all per-rule CSVs at the end.
        /// </summary>
        public static void SetFilenameSuffix(string suffix)
        {
            const string ext = ".csv";
            string baseName = _filename.EndsWith(ext)
                ? _filename[..^ext.Length]
                : _filename;
            _filename = baseName + suffix + ext;
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

        public static void LogEpisode(string ruleName, int seed, double makespan,
                                       int jobCount, int machineCount, int totalOps,
                                       int decisionCount, double totalReward, float averageTimeScale)
        {
            bool fileExists = File.Exists(FilePath);
            using StreamWriter writer = new StreamWriter(FilePath, append: true);
            if (!fileExists)
                writer.WriteLine("timestamp,rule,seed,makespan,jobs,machines,total_ops,decisions,total_reward,timescale");
            writer.WriteLine(
                $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}," +
                $"{ruleName}," +
                $"{seed}," +
                $"{makespan:F2}," +
                $"{jobCount}," +
                $"{machineCount}," +
                $"{totalOps}," +
                $"{decisionCount}," +
                $"{totalReward:F4}," +
                $"{averageTimeScale:F4}"
            );
            Debug.Log($"[Results] Logged: {ruleName} seed={seed} makespan={makespan:F1} - {FilePath}");
        }
    }
}