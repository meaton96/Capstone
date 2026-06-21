#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace Assets.Scripts.Editor
{
    /// @brief Copies data files (configs, scripts) into the build output folder.
    ///
    /// @details Runs automatically after every build. Files in the source folder
    ///          are copied next to the built executable so headless batch runs
    ///          can reference them with relative paths.
    ///
    /// Setup:
    ///   1. Place this script in any Editor/ folder
    ///   2. Create Assets/StreamingAssets/BatchConfigs/ (or edit SourceFolder below)
    ///   3. Drop your .json config files in there
    ///   4. Build — they appear in the output folder under BatchConfigs/
    public class CopyConfigsPostBuild : IPostprocessBuildWithReport
    {
        /// @brief Runs after all other post-processors (higher = later).
        public int callbackOrder => 0;

        /// @brief Folder inside Assets/ containing files to copy.
        ///        Edit this to change the source location.
        private static readonly string SourceFolder = Path.Combine(Application.dataPath, "BatchConfigs");

        public void OnPostprocessBuild(BuildReport report)
        {
            string buildPath = report.summary.outputPath;
            string buildDir;

            // outputPath points to the executable — get its directory
            if (report.summary.platform == BuildTarget.StandaloneOSX)
            {
                // macOS: outputPath is MyBuild.app — put configs next to it
                buildDir = Path.GetDirectoryName(buildPath);
            }
            else
            {
                // Windows/Linux: outputPath is MyBuild.exe — same directory
                buildDir = Path.GetDirectoryName(buildPath);
            }

            if (string.IsNullOrEmpty(buildDir)) return;

            string destFolder = Path.Combine(buildDir, "BatchConfigs");

            if (!Directory.Exists(SourceFolder))
            {
                Debug.Log($"[PostBuild] No BatchConfigs folder at {SourceFolder} — skipping copy. " +
                          "Create Assets/BatchConfigs/ and add your .json files there.");
                return;
            }

            // Create destination
            Directory.CreateDirectory(destFolder);

            // Copy all files (non-recursive — add SearchOption.AllDirectories if you want subfolders)
            string[] files = Directory.GetFiles(SourceFolder, "*.*", SearchOption.AllDirectories);
            int copied = 0;

            foreach (string srcFile in files)
            {
                // Skip Unity .meta files
                if (srcFile.EndsWith(".meta")) continue;

                string relativePath = srcFile.Substring(SourceFolder.Length + 1);
                string destFile = Path.Combine(destFolder, relativePath);

                // Ensure subdirectory exists
                string destSubDir = Path.GetDirectoryName(destFile);
                if (!string.IsNullOrEmpty(destSubDir))
                    Directory.CreateDirectory(destSubDir);

                File.Copy(srcFile, destFile, overwrite: true);
                copied++;
            }

            Debug.Log($"[PostBuild] Copied {copied} config files to {destFolder}");
        }
    }
}
#endif