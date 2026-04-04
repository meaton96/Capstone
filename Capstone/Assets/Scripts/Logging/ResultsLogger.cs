using System;
using System.IO;
using UnityEngine;

namespace Assets.Scripts.Logging
{
    public static class ResultsLogger
    {
        public static string OutputDirectory = "";
        private static string FilePath
        {
            get
            {
                string dir = string.IsNullOrEmpty(OutputDirectory)
                    ? Application.persistentDataPath
                    : OutputDirectory;
                return Path.Combine(dir, "baseline_results.csv");
            }
        }

        public static void LogEpisode(string ruleName, int seed, double makespan,
                                       int jobCount, int machineCount, int totalOps,
                                       int decisionCount, double totalReward, float averageTimeScale)
        {
            bool fileExists = File.Exists(FilePath);

            using StreamWriter writer = new StreamWriter(FilePath, append: true);

            if (!fileExists)
                writer.WriteLine("timestamp,rule,seed,makespan,jobs,machines,total_ops,decisions,total_reward");

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

            Debug.Log($"[Results] Logged: {ruleName} seed={seed} makespan={makespan:F1} → {FilePath}");
        }
    }
}