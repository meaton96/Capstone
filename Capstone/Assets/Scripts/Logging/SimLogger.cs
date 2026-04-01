using UnityEngine;

namespace Assets.Scripts.Logging
{
    /// @brief Verbosity levels for @c SimLogger, ordered from least to most detailed.
    public enum LogLevel
    {
        /// @brief Critical errors only. Always printed regardless of the active level.
        Error = 0,

        /// @brief High-level milestones: episode start/end, makespan, decision count.
        Low = 1,

        /// @brief Per-decision and per-job state transitions.
        Medium = 2,

        /// @brief Frame-level detail: AGV dispatches, queue arrivals, progress ticks.
        High = 3,
    }

    /// @brief Static levelled logger that gates @c UnityEngine.Debug output behind a
    ///        configurable verbosity threshold.
    ///
    /// @details Set @c ActiveLevel once at startup (or at any time from the Inspector
    ///          via a thin MonoBehaviour wrapper) and every call site automatically
    ///          respects the threshold. Messages below the active level are no-ops with
    ///          no string allocation cost because the caller controls the level tag.
    ///
    /// @par Usage
    /// @code
    /// SimLogger.ActiveLevel = LogLevel.Medium;
    ///
    /// SimLogger.Log(LogLevel.Low,    "[SimBridge] Episode started.");
    /// SimLogger.Log(LogLevel.High,   $"[Ghost AGV] Dispatched Job {jobId} to Machine {machineId}.");
    /// SimLogger.LogError(            "[SimBridge] Parse error: " + ex.Message);
    /// @endcode
    public static class SimLogger
    {
        /// @brief The minimum level a message must meet to be written to the console.
        ///        Defaults to @c Low so only important milestones appear out of the box.
        public static LogLevel ActiveLevel = LogLevel.Low;

        // ─────────────────────────────────────────────────────────
        //  Core API
        // ─────────────────────────────────────────────────────────

        /// @brief Writes @p message to the console if @p level is within the active threshold.
        /// @param level    Verbosity classification of this message.
        /// @param message  Text to output.
        public static void Log(LogLevel level, string message)
        {
            if (level <= ActiveLevel)
                Debug.Log(message);
        }

        /// @brief Writes @p message as a @c Debug.LogWarning, always visible.
        /// @param message  Warning text to output.
        public static void LogWarning(string message)
        {
            Debug.LogWarning(message);
        }

        /// @brief Writes @p message as a @c Debug.LogError, always visible.
        /// @param message  Error text to output.
        public static void LogError(string message)
        {
            Debug.LogError(message);
        }

        // ─────────────────────────────────────────────────────────
        //  Convenience Shorthands
        // ─────────────────────────────────────────────────────────

        /// @brief Shorthand for @c Log(LogLevel.Low, message).
        public static void Low(string message) => Log(LogLevel.Low, message);

        /// @brief Shorthand for @c Log(LogLevel.Medium, message).
        public static void Medium(string message) => Log(LogLevel.Medium, message);

        /// @brief Shorthand for @c Log(LogLevel.High, message).
        public static void High(string message) => Log(LogLevel.High, message);

        public static void Error(string message) => Log(LogLevel.Error, message);
    }
}