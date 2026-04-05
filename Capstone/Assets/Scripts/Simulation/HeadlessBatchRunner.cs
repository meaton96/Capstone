using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Assets.Scripts.Logging;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation
{
    /// @brief Headless batch runner that drives the simulation through multiple configs × rules.
    ///
    /// @details Designed for data-collection builds. Reads a batch config JSON from the
    ///          command line, then runs every (config, heuristicRule) combination in series.
    ///          Each episode's results append to the existing CSV via ResultsLogger.
    ///
    /// Usage (headless build):
    /// @code
    ///   ./MyBuild.exe -batchmode -nographics -timescale 100 \
    ///       -batchconfig path/to/batch_configs.json \
    ///       -repeats 3
    /// @endcode
    ///
    /// The runner will:
    ///   1. Parse the batch config file (array of FJSSPConfig JSON objects)
    ///   2. For each config, for each DispatchingRule, run the simulation to completion
    ///   3. Optionally repeat each combo N times with different seeds
    ///   4. Quit the application when all runs finish
    ///
    /// Attach this MonoBehaviour to the same GameObject as SimulationBridge.
    /// In headless mode it takes over episode lifecycle; in editor mode it does nothing
    /// unless you manually call RunBatch().
    public class HeadlessBatchRunner : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private SimulationBridge bridge;
        [SerializeField] private SchedulingAgent agent;

        [Header("Fallback Settings (used if no CLI args)")]
        [SerializeField] private TextAsset fallbackBatchJson;
        [SerializeField] private int fallbackRepeats = 1;

        /// @brief All dispatching rules to sweep across. Mirrors SimulationBridge.ActionToRule.
        private static readonly DispatchingRule[] AllRules = new DispatchingRule[]
        {
            DispatchingRule.SPT_SMPT,
            DispatchingRule.SPT_SRWT,
            DispatchingRule.LPT_MMUR,
            DispatchingRule.LPT_SMPT,
            DispatchingRule.SRT_SRWT,
            DispatchingRule.SRT_SMPT,
            DispatchingRule.LRT_MMUR,
            DispatchingRule.SDT_SRWT
        };

        private bool isBatchRunning;
        private int totalRuns;
        private int completedRuns;

        // ─────────────────────────────────────────────────────────
        //  Unity Lifecycle
        // ─────────────────────────────────────────────────────────

        private void Start()
        {
            string batchPath = GetCLIArg("-batchconfig");

            // Only auto-start in batchmode, or if a batchconfig was explicitly passed
            if (Application.isBatchMode || !string.IsNullOrEmpty(batchPath))
            {
                int repeats = 1;
                string repeatsStr = GetCLIArg("-repeats");
                if (!string.IsNullOrEmpty(repeatsStr))
                    int.TryParse(repeatsStr, out repeats);

                FJSSPConfig[] configs = LoadConfigs(batchPath);
                if (configs.Length > 0)
                    StartCoroutine(RunBatchCoroutine(configs, repeats));
                else
                    QuitWithError("No valid configs found.");
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Public API (for editor / UI triggering)
        // ─────────────────────────────────────────────────────────

        /// @brief Starts a batch run from script or a UI button.
        /// @param configs  Array of configurations to sweep.
        /// @param repeats  How many times to repeat each (config, rule) pair with offset seeds.
        public void RunBatch(FJSSPConfig[] configs, int repeats = 1)
        {
            if (isBatchRunning)
            {
                Debug.LogWarning("[BatchRunner] Batch already in progress.");
                return;
            }
            StartCoroutine(RunBatchCoroutine(configs, repeats));
        }

        /// @brief Starts a batch from a JSON file path.
        public void RunBatchFromFile(string path, int repeats = 1)
        {
            var configs = ConfigLoader.LoadBatch(path);
            if (configs.Length == 0)
            {
                Debug.LogError($"[BatchRunner] No configs in {path}");
                return;
            }
            RunBatch(configs, repeats);
        }

        // ─────────────────────────────────────────────────────────
        //  Core Batch Loop
        // ─────────────────────────────────────────────────────────

        private IEnumerator RunBatchCoroutine(FJSSPConfig[] configs, int repeats)
        {
            isBatchRunning = true;
            totalRuns = configs.Length * AllRules.Length * repeats;
            completedRuns = 0;

            Debug.Log($"[BatchRunner] Starting batch: {configs.Length} configs × " +
                      $"{AllRules.Length} rules × {repeats} repeats = {totalRuns} total runs");

            float startWall = Time.realtimeSinceStartup;

            foreach (var baseConfig in configs)
            {
                for (int rep = 0; rep < repeats; rep++)
                {
                    foreach (var rule in AllRules)
                    {
                        // Clone config with offset seed for this repeat
                        FJSSPConfig runConfig = CloneWithSeed(baseConfig, baseConfig.Seed + rep);

                        Debug.Log($"[BatchRunner] Run {completedRuns + 1}/{totalRuns}: " +
                                  $"config={runConfig.Name} rule={rule} seed={runConfig.Seed}");

                        yield return RunSingleEpisode(runConfig, rule);

                        completedRuns++;

                        float elapsed = Time.realtimeSinceStartup - startWall;
                        float avgPerRun = elapsed / completedRuns;
                        float eta = avgPerRun * (totalRuns - completedRuns);
                        Debug.Log($"[BatchRunner] Progress: {completedRuns}/{totalRuns} " +
                                  $"({elapsed:F1}s elapsed, ETA {eta:F1}s)");
                    }
                }
            }

            float totalTime = Time.realtimeSinceStartup - startWall;
            Debug.Log($"[BatchRunner] Batch complete: {totalRuns} runs in {totalTime:F1}s");
            isBatchRunning = false;

            if (Application.isBatchMode)
            {
                Debug.Log("[BatchRunner] Headless mode — quitting application.");
                Application.Quit();
            }
        }

        /// @brief Runs one (config, rule) episode to completion, yielding each frame.
        private IEnumerator RunSingleEpisode(FJSSPConfig config, DispatchingRule rule)
        {
            // Set the agent's heuristic to the target rule
            if (agent != null)
                agent.SetHeuristicRule(rule);

            // Phase 1: Load config
            bridge.LoadConfig(config);
            yield return null;

            // Phase 2: Spawn factory
            bridge.SpawnFactory();
            yield return null; // let physics settle one frame

            // Phase 3: Arm the agent and start simulation
            if (agent != null)
                agent.IsArmed = true;
            bridge.StartSimulation();

            // Phase 4: Wait for episode to finish
            while (bridge.IsEpisodeActive)
            {
                yield return null;
            }

            // Small cooldown between episodes
            yield return new WaitForSecondsRealtime(0.1f);
        }

        // ─────────────────────────────────────────────────────────
        //  Helpers
        // ─────────────────────────────────────────────────────────

        // (heuristic rule is now set via agent.SetHeuristicRule() directly)

        private FJSSPConfig CloneWithSeed(FJSSPConfig source, int newSeed)
        {
            return new FJSSPConfig
            {
                Name = source.Name,
                Seed = newSeed,
                JobCount = source.JobCount,
                MachinesPerType = source.MachinesPerType,
                MachineTypeLayout = (MachineType[])source.MachineTypeLayout.Clone(),
                MinProcTime = source.MinProcTime,
                MaxProcTime = source.MaxProcTime,
                MinOpsPerJob = source.MinOpsPerJob,
                MaxOpsPerJob = source.MaxOpsPerJob,
                MaxArrivalTime = source.MaxArrivalTime,
            };
        }

        private FJSSPConfig[] LoadConfigs(string cliPath)
        {
            // CLI path takes priority
            if (!string.IsNullOrEmpty(cliPath))
                return ConfigLoader.LoadBatch(cliPath);

            // Fallback to editor-assigned TextAsset
            if (fallbackBatchJson != null)
                return ConfigLoader.ParseBatch(fallbackBatchJson.text);

            Debug.LogError("[BatchRunner] No batch config source available.");
            return Array.Empty<FJSSPConfig>();
        }

        private static string GetCLIArg(string key)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == key)
                    return args[i + 1];
            return null;
        }

        private void QuitWithError(string message)
        {
            Debug.LogError($"[BatchRunner] {message}");
            if (Application.isBatchMode)
                Application.Quit(1);
        }
    }
}