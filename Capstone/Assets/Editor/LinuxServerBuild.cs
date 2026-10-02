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
    /// While a sweep runs many players, the per-user inotify limit (128) can be used up; script compilation then
    /// fails with "Can't find file /tmp/ilpp.sock-..." (the IL post-processor cannot start) although Unity exits 0.
    /// Prefix the command with DOTNET_USE_POLLING_FILE_WATCHER=1 DOTNET_hostBuilder__reloadConfigOnChange=false,
    /// and check the log for "[LinuxServerBuild] Succeeded".
    public static class LinuxServerBuild
    {
        /// @brief Output executable, relative to the Unity project folder.
        private static readonly string OutputPath = Path.Combine("..", "linux_server", "capstone.x86_64");

        /// @brief Second build folder, so a new build can be tested while a sweep keeps running on linux_server/
        ///        (overwriting a player's files under running processes is unsafe). run_experiment_queue.py
        ///        --exe ../linux_server2/capstone.x86_64 runs it; its results go to linux_server2/Results.
        private static readonly string AltOutputPath = Path.Combine("..", "linux_server2", "capstone.x86_64");

        /// @brief Third build folder for development builds, so neither linux_server/ (the baseline a change is
        ///        compared against) nor linux_server2/ (which may have a player running) is overwritten.
        private static readonly string DevOutputPath = Path.Combine("..", "linux_server_dev", "capstone.x86_64");

        [MenuItem("Build/Linux Server (ML-Agents)")]
        public static void Build() => BuildTo(OutputPath);

        /// Headless: -executeMethod Assets.Scripts.Editor.LinuxServerBuild.BuildAlt
        [MenuItem("Build/Linux Server (ML-Agents) - linux_server2")]
        public static void BuildAlt() => BuildTo(AltOutputPath);

        /// Headless: -executeMethod Assets.Scripts.Editor.LinuxServerBuild.BuildDev
        [MenuItem("Build/Linux Server (ML-Agents) - linux_server_dev")]
        public static void BuildDev() => BuildTo(DevOutputPath);

        /// <summary>Player for the event-based twin validation (env/des_twin), kept apart from the sweep players.</summary>
        [MenuItem("Build/Linux Server (ML-Agents) - linux_server_des")]
        public static void BuildDes() => BuildTo(Path.Combine("..", "linux_server_des", "capstone.x86_64"));

        private static void BuildTo(string outputPath)
        {
            var options = new BuildPlayerOptions
            {
                scenes = EditorBuildSettings.scenes.Where(s => s.enabled).Select(s => s.path).ToArray(),
                locationPathName = outputPath,
                target = BuildTarget.StandaloneLinux64,
                targetGroup = BuildTargetGroup.Standalone,
                subtarget = (int)StandaloneBuildSubtarget.Server,
                options = BuildOptions.None,
            };

            BuildReport report = BuildPipeline.BuildPlayer(options);
            // Provenance for the committed player (thesis section 7.2), checked by env/player_manifest.py.
            if (report.summary.result == BuildResult.Succeeded)
                BuildManifest.Write(outputPath);
            Debug.Log($"[LinuxServerBuild] {report.summary.result} -> {Path.GetFullPath(outputPath)} " +
                      $"({report.summary.totalErrors} errors, {report.summary.totalTime})");

            // Only signal failure via exit code when headless — from the menu this would close the editor.
            if (report.summary.result != BuildResult.Succeeded && Application.isBatchMode)
                EditorApplication.Exit(1);
        }
    }
}
#endif
