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
    ///
    /// @details Provides filtered logging based on an @c ActiveLevel and supports 
    /// persistent storage of logs to a text file for post-run analysis.
    public static class SimLogger
    {
        public static LogLevel ActiveLevel = LogLevel.High;
        private static string _filePath;
        private static bool _isFileLoggingEnabled = false;

        /// @brief Configures the directory and file for persistent logging.
        ///
        /// @param folderPath The absolute path to the directory where logs should be stored.
        /// @param fileName The name of the log file, including extension.
        ///
        /// @details Creates the target directory if it does not exist. It overwrites 
        /// any existing file with the same name to start a fresh log for the current session.
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

        /// @brief Appends a string to the current log file with a precise timestamp.
        ///
        /// @param message The raw string to write to the file.
        ///
        /// @details This method fails silently if an I/O error occurs to prevent 
        /// recursive logging loops or simulation crashes during file access contention.
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

        /// @brief Logs a message if its level is within the current @c ActiveLevel.
        ///
        /// @param level The importance of the message.
        /// @param message The string content to log.
        ///
        /// @details Validates the @c level against @c ActiveLevel before printing 
        /// to the Unity Console and invoking @c WriteToFile.
        public static void Log(LogLevel level, string message)
        {
            string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
            if (level <= ActiveLevel)
            {
                Debug.Log($"{timestamp} | {message}");
                WriteToFile($"{message}");
            }
        }

        /// @brief Forces a warning message to the console and file regardless of @c ActiveLevel.
        public static void LogWarning(string message)
        {
            Debug.LogWarning(message);
            WriteToFile($"[Warning] {message}");
        }

        /// @brief Forces an error message to the console and file regardless of @c ActiveLevel.
        public static void LogError(string message)
        {
            Debug.LogError(message);
            WriteToFile($"[Error] {message}");
        }

        /// @brief Shorthand for logging at @c LogLevel.Low.
        public static void Low(string message) => Log(LogLevel.Low, message);

        /// @brief Shorthand for logging at @c LogLevel.Medium.
        public static void Medium(string message) => Log(LogLevel.Medium, message);

        /// @brief Shorthand for logging at @c LogLevel.High.
        public static void High(string message) => Log(LogLevel.High, message);

        /// @brief Shorthand for logging at @c LogLevel.Error.
        public static void Error(string message) => Log(LogLevel.Error, message);
    }
}