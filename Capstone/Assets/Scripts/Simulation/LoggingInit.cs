using System.IO;
using UnityEngine;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation
{
    /// @brief Orchestrates the initialization of static logging utilities.
    ///
    /// @details Ensures that both @c ResultsLogger and @c SimLogger are configured 
    /// with appropriate directory paths and verbosity levels. It handles cross-platform 
    /// directory resolution to ensure logs are written to persistent storage locations 
    /// outside of read-only application bundles.
    public class LoggingInitializer : MonoBehaviour
    {
        [Header("Logging Settings")]
        [SerializeField] private LogLevel simLoggerLevel = LogLevel.Low;
        [SerializeField] private bool enableFileLogging = true;

        public static LoggingInitializer Instance;

        /// @brief Performs the initial setup of logging paths and configurations.
        ///
        /// @details Resolves the "Results" directory based on the execution environment 
        /// (Editor, Windows/Linux Standalone, or macOS Standalone). It checks for the 
        /// @c -loglevel command-line argument to allow for runtime verbosity overrides 
        /// before initializing the file systems for both the @c ResultsLogger and 
        /// the @c SimLogger.
        private void Awake()
        {
            Instance = this;

            string cliLevel = GetCLIArg("-loglevel");
            if (!string.IsNullOrEmpty(cliLevel) &&
                System.Enum.TryParse(cliLevel, ignoreCase: true, out LogLevel parsedLevel))
            {
                simLoggerLevel = parsedLevel;
            }

            string folderPath;

#if UNITY_EDITOR
            folderPath = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Results");
#elif UNITY_STANDALONE_OSX
            folderPath = Path.Combine(Directory.GetParent(Application.dataPath).Parent.FullName, "Results");
#else
            folderPath = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Results");
#endif

            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }

            ResultsLogger.OutputDirectory = folderPath;

            SimLogger.ActiveLevel = simLoggerLevel;
            if (enableFileLogging)
            {
                SimLogger.InitializeFileLogging(folderPath, "simulation_log.txt");
            }
        }

        /// @brief Retrieves a specific value from the application's command-line arguments.
        ///
        /// @param key The argument flag to search for (e.g., "-loglevel").
        /// @return The value associated with the key if found; otherwise, @c null.
        ///
        /// @details Iterates through @c System.Environment.GetCommandLineArgs to find 
        /// the key and returns the subsequent array element as the value.
        private static string GetCLIArg(string key)
        {
            string[] args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == key) return args[i + 1];
            return null;
        }
    }
}