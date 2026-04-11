using System.IO;
using UnityEngine;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// Bootstrapper for the static logging classes. 
    /// Ensures logs are written safely outside of the macOS .app bundle.
    /// </summary>
    public class LoggingInitializer : MonoBehaviour
    {
        [Header("Logging Settings")]
        [SerializeField] private LogLevel simLoggerLevel = LogLevel.Low;
        [SerializeField] private bool enableFileLogging = true;

        public static LoggingInitializer Instance;

        private void Awake()
        {
            Instance = this;

            // ── CLI override for log level ────────────────────────
            // Usage: -loglevel Low | Medium | High
            // Lets headless batch runs suppress verbose output without
            // recompiling. Inspector value is used if no CLI arg present.
            string cliLevel = GetCLIArg("-loglevel");
            if (!string.IsNullOrEmpty(cliLevel) &&
                System.Enum.TryParse(cliLevel, ignoreCase: true, out LogLevel parsedLevel))
            {
                simLoggerLevel = parsedLevel;
            }

            // 1. Resolve OS-Safe Directory
            string folderPath;

#if UNITY_EDITOR
            folderPath = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Results");
#elif UNITY_STANDALONE_OSX
            // Mac: Go up two levels to get outside the .app bundle
            folderPath = Path.Combine(Directory.GetParent(Application.dataPath).Parent.FullName, "Results");
#else
            // Windows/Linux: Go up one level to get outside the _Data folder
            folderPath = Path.Combine(Directory.GetParent(Application.dataPath).FullName, "Results");
#endif

            // Ensure the directory exists
            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }

            // 2. Initialize Results Logger
            ResultsLogger.OutputDirectory = folderPath;

            // 3. Initialize Sim Logger
            SimLogger.ActiveLevel = simLoggerLevel;
            if (enableFileLogging)
            {
                SimLogger.InitializeFileLogging(folderPath, "simulation_log.txt");
            }
        }
        private static string GetCLIArg(string key)
        {
            string[] args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == key) return args[i + 1];
            return null;
        }
    }
}