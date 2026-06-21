using UnityEngine;
using System.IO;
namespace Assets.Scripts.Scheduling.Validation
{
    /// @brief Unity entry point that exports C# makespan results for cross-validation against job_shop_lib.
    ///
    /// @details Attach to any GameObject in the scene. On Play, the component locates all Taillard
    /// JSON files in @ref instanceDirectory, runs @ref ValidationExporter.RunAndExport to simulate
    /// every instance under every @ref DispatchingRule, and writes @c csharp_makespans.csv and
    /// @c csharp_schedules.json to @ref outputDirectory. The exported files are then consumed by
    /// @c compare_results.py to diff C# results against the Python job_shop_lib reference.
    ///
    /// @par Setup
    /// -# Place Taillard JSON files in @c Assets/Resources/Instances/.
    /// -# Copy the same files to a regular folder (e.g. @c Assets/StreamingAssets/Instances/)
    ///    because @ref ValidationExporter reads directly from disk, not via @c Resources.Load.
    /// -# Set @ref instanceDirectory and @ref outputDirectory in the Inspector.
    /// -# Enter Play Mode.
    /// -# Run @c compare_results.py from the output directory:
    ///    @code
    ///    cd Assets/StreamingAssets/Validation
    ///    python compare_results.py
    ///    @endcode
    ///
    /// @note This component does not use the Unity Resources system. Both directories must be
    /// accessible as real filesystem paths at runtime, not Unity virtual paths.
    public class CrossValidator : MonoBehaviour
    {
        /// @brief Absolute or project-relative path to the folder containing Taillard @c .json files.
        /// @details Must be a real filesystem path readable by @c System.IO — not a @c Resources/
        /// virtual path. Defaults to @c Assets/StreamingAssets/Instances. Validated on @ref Start;
        /// a missing directory aborts the export with a @c Debug.LogError.
        [Header("Paths")]
        [Tooltip("Folder containing Taillard .json files")]
        public string instanceDirectory = "Assets/StreamingAssets/Instances";

        /// @brief Absolute or project-relative path to the folder where export files will be written.
        /// @details Created automatically by @ref Start if it does not already exist.
        /// @ref ValidationExporter.RunAndExport writes @c csharp_makespans.csv and
        /// @c csharp_schedules.json to this location.
        [Tooltip("Folder to write csharp_makespans.csv and csharp_schedules.json")]
        public string outputDirectory = "Assets/StreamingAssets/Validation";

        /// @brief Unity lifecycle callback — runs the full export pipeline on scene start.
        ///
        /// @details Performs the following steps in order:
        /// -# Creates @ref outputDirectory if it does not exist.
        /// -# Validates that @ref instanceDirectory exists; logs a @c Debug.LogError and aborts if not.
        /// -# Logs the resolved paths for both directories.
        /// -# Delegates to @ref ValidationExporter.RunAndExport to simulate all instances and write output files.
        /// -# Logs the @c compare_results.py invocation command for convenience.
        void Start()
        {
            if (!Directory.Exists(outputDirectory))
                Directory.CreateDirectory(outputDirectory);

            if (!Directory.Exists(instanceDirectory))
            {
                Debug.LogError($"Instance directory not found: {instanceDirectory}");
                return;
            }

            Debug.Log($"<color=cyan>Running cross-validation export...</color>");
            Debug.Log($"  Instances: {instanceDirectory}");
            Debug.Log($"  Output:    {outputDirectory}");

            ValidationExporter.RunAndExport(instanceDirectory, outputDirectory);

            Debug.Log("<color=green>Export complete. Now run compare_results.py:</color>");
            Debug.Log($"  cd {outputDirectory}");
            Debug.Log($"  python compare_results.py");
        }
    }
}