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

        private void Awake()
        {
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
    }
}