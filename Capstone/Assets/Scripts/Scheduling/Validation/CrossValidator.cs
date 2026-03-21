using UnityEngine;
using System.IO;
using Scheduling.Validation;

namespace Scheduling.Unity
{
    /// <summary>
    /// Attach to a GameObject. On Play, exports C# makespan results
    /// for cross-validation against job_shop_lib.
    ///
    /// Setup:
    ///   1. Place Taillard JSON files in Assets/Resources/Instances/
    ///   2. Also copy them to a regular folder (e.g., Assets/StreamingAssets/Instances/)
    ///      because ValidationExporter reads files directly from disk.
    ///   3. Set instanceDirectory and outputDirectory in the Inspector.
    ///   4. Hit Play.
    ///   5. Run compare_results.py in the output directory.
    /// </summary>
    public class CrossValidator : MonoBehaviour
    {
        [Header("Paths")]
        [Tooltip("Folder containing Taillard .json files")]
        public string instanceDirectory = "Assets/StreamingAssets/Instances";

        [Tooltip("Folder to write csharp_makespans.csv and csharp_schedules.json")]
        public string outputDirectory = "Assets/StreamingAssets/Validation";

        void Start()
        {
            // Ensure output directory exists
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
