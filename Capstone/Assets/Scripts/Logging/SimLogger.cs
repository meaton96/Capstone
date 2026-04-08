using UnityEngine;
using System.IO;
using System;

namespace Assets.Scripts.Logging
{
    public enum LogLevel
    {
        Error = 0,
        Low = 1,
        Medium = 2,
        High = 3,
    }

    public static class SimLogger
    {
        public static LogLevel ActiveLevel = LogLevel.Low;
        private static string _filePath;
        private static bool _isFileLoggingEnabled = false;

        // CHANGED: Now accepts the safe folder path from the wrapper
        public static void InitializeFileLogging(string folderPath, string fileName = "simulation_log.txt")
        {
            try
            {
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }

                _filePath = Path.Combine(folderPath, fileName);
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
                // Fail silently to avoid infinite recursion
            }
        }

        public static void Log(LogLevel level, string message)
        {
            if (level <= ActiveLevel)
            {
                Debug.Log(message);
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