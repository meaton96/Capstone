#if UNITY_EDITOR
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace Assets.Scripts.Editor
{
    /// @brief Headless entry point for building the Linux dedicated-server player used by
    ///        mlagents-learn, so the build can be produced without opening the editor.
    ///
    /// Usage (from the repo root):
    ///   ~/Unity/Hub/Editor/6000.3.15f1/Editor/Unity -batchmode -quit -nographics \
    ///       -projectPath Capstone -buildTarget Linux64 -standaloneBuildSubtarget Server \
    ///       -executeMethod Assets.Scripts.Editor.LinuxServerBuild.Build -logFile build.log
    public static class LinuxServerBuild
    {
        /// @brief Output executable, relative to the Unity project folder.
        private static readonly string OutputPath = Path.Combine("..", "linux_server", "capstone.x86_64");

        [MenuItem("Build/Linux Server (ML-Agents)")]
        public static void Build()
        {
            var options = new BuildPlayerOptions
            {
                scenes = EditorBuildSettings.scenes.Where(s => s.enabled).Select(s => s.path).ToArray(),
                locationPathName = OutputPath,
                target = BuildTarget.StandaloneLinux64,
                targetGroup = BuildTargetGroup.Standalone,
                subtarget = (int)StandaloneBuildSubtarget.Server,
                options = BuildOptions.None,
            };

            BuildReport report = BuildPipeline.BuildPlayer(options);
            Debug.Log($"[LinuxServerBuild] {report.summary.result} -> {Path.GetFullPath(OutputPath)} " +
                      $"({report.summary.totalErrors} errors, {report.summary.totalTime})");

            // Only signal failure via exit code when headless — from the menu this would close the editor.
            if (report.summary.result != BuildResult.Succeeded && Application.isBatchMode)
                EditorApplication.Exit(1);
        }
    }
}
#endif
