using UnityEngine;
using System.IO;
using System;

namespace Assets.Scripts.Logging
{
    /// @brief Defines the verbosity levels for the simulation logger.
    public enum LogLevel
    {
        Error = 0,
        Low = 1,
        Medium = 2,
        High = 3,
    }

    /// @brief Static utility for handling console and file-based logging within the simulation.
    public static class SimLogger
    {
        public static LogLevel ActiveLevel = LogLevel.High;
        private static string _filePath;
        private static bool _isFileLoggingEnabled = false;

        /// @brief Helper to extract command line arguments passed from the OS/PowerShell.
        private static string GetArg(string name)
        {
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == name && args.Length > i + 1)
                {
                    return args[i + 1];
                }
            }
            return null;
        }

        /// @brief Configures the directory and dynamically names the file for persistent logging.
        public static void InitializeFileLogging(string folderPath, string baseFileName = "simulation_log")
        {
            try
            {
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }

                // Intercept the suffix passed by PowerShell (e.g., "_SPT_SRWT"). 
                // If null (e.g., running normally in the Editor or your 9th instance), it defaults to an empty string.
                string suffix = GetArg("-outputsuffix") ?? "";

                // Construct the highly specific filename
                string finalFileName = $"{baseFileName}{suffix}.txt";

                _filePath = Path.Combine(folderPath, finalFileName);
                File.WriteAllText(_filePath, $"--- Log Started: {DateTime.Now} ---\n");

                _isFileLoggingEnabled = true;
                Debug.Log($"[SimLogger] File logging initialized at: {_filePath}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SimLogger] Failed to initialize file logging: {ex.Message}");
            }
        }

        private static void WriteToFile(string message)
        {
            if (!_isFileLoggingEnabled) return;

            try
            {
                string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
                File.AppendAllText(_filePath, $"{timestamp} {message}\n");
            }
            catch
            {
                // Silent failure to prevent infinite recursion or crash.
            }
        }

        public static void Log(LogLevel level, string message)
        {
            string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
            if (level <= ActiveLevel)
            {
                Debug.Log($"{timestamp} | {message}");
                WriteToFile($"{message}");
            }
        }

        public static void LogWarning(string message)
        {
            Debug.LogWarning(message);
            WriteToFile($"[Warning] {message}");
        }

        public static void LogError(string message)
        {
            Debug.LogError(message);
            WriteToFile($"[Error] {message}");
        }

        public static void Low(string message) => Log(LogLevel.Low, message);
        public static void Medium(string message) => Log(LogLevel.Medium, message);
        public static void High(string message) => Log(LogLevel.High, message);
        public static void Error(string message) => Log(LogLevel.Error, message);
    }
}