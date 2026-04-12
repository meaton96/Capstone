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
    ///   ./capstone.exe -batchmode -nographics -timescale 100 `
    ///      -batchconfig ./BatchConfigs/BatchConfigs.json `
    ///      -repeats 1
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

        // ── Active rule set (filtered by -rules CLI arg) ──────────
        // Populated in Start(). Defaults to AllRules if no filter given.
        private DispatchingRule[] activeRules;

        // ─────────────────────────────────────────────────────────
        //  Unity Lifecycle
        // ─────────────────────────────────────────────────────────

        private void Start()
        {
            string batchPath = GetCLIArg("-batchconfig");

            // Only auto-start in batchmode, or if a batchconfig was explicitly passed
            if (Application.isBatchMode || !string.IsNullOrEmpty(batchPath))
            {
                // ── Timescale ─────────────────────────────────────────
                string timeScaleStr = GetCLIArg("-timescale");
                if (!string.IsNullOrEmpty(timeScaleStr) && float.TryParse(timeScaleStr, out float parsedScale))
                {
                    Time.timeScale = parsedScale;
                    SimLogger.Low($"[BatchRunner] TimeScale set to {parsedScale}x via CLI.");
                }
                else
                {
                    Time.timeScale = 100f;
                    SimLogger.Low("[BatchRunner] No timescale provided. Defaulting to 100x.");
                }

                // ── Rules filter ───────────────────────────────────────
                // -rules SPT_SMPT,LPT_MMUR   (comma-separated, no spaces)
                // Lets the parallel launcher assign each process a subset.
                activeRules = ParseRulesArg(GetCLIArg("-rules"));
                SimLogger.Low($"[BatchRunner] Active rules ({activeRules.Length}): " +
                              string.Join(", ", activeRules));

                // ── Output suffix ──────────────────────────────────────
                // -outputsuffix _SPT_SMPT  →  results_SPT_SMPT.csv
                // Prevents CSV collisions when N processes write simultaneously.
                //
                // ResultsLogger.SetFilenameSuffix() must append the suffix to
                // whatever base filename ResultsLogger uses internally, e.g.:
                //   public static void SetFilenameSuffix(string suffix) {
                //       _filename = "results" + suffix + ".csv";
                //   }
                // Add this one-liner to ResultsLogger.cs if not already present.
                string suffix = GetCLIArg("-outputsuffix") ?? string.Empty;
                if (!string.IsNullOrEmpty(suffix))
                    ResultsLogger.SetFilenameSuffix(suffix);

                // ── Repeats ────────────────────────────────────────────
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
                SimLogger.LogWarning("[BatchRunner] Batch already in progress.");
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
                SimLogger.LogError($"[BatchRunner] No configs in {path}");
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
            // Use filtered rule list (set from CLI -rules arg, defaults to all rules)
            if (activeRules == null || activeRules.Length == 0)
                activeRules = AllRules;

            totalRuns = configs.Length * activeRules.Length * repeats;
            completedRuns = 0;

            SimLogger.Low($"[BatchRunner] Starting batch: {configs.Length} configs x " +
                      $"{activeRules.Length} rules x {repeats} repeats = {totalRuns} total runs");

            float startWall = Time.realtimeSinceStartup;

            foreach (var baseConfig in configs)
            {
                for (int rep = 0; rep < repeats; rep++)
                {
                    foreach (var rule in activeRules)
                    {
                        // Clone config with offset seed for this repeat
                        FJSSPConfig runConfig = CloneWithSeed(baseConfig, baseConfig.Seed + rep);

                        SimLogger.Low($"[BatchRunner] Run {completedRuns + 1}/{totalRuns}: " +
                                  $"config={runConfig.Name} rule={rule} seed={runConfig.Seed}");

                        yield return RunSingleEpisode(runConfig, rule);

                        completedRuns++;

                        float elapsed = Time.realtimeSinceStartup - startWall;
                        float avgPerRun = elapsed / completedRuns;
                        float eta = avgPerRun * (totalRuns - completedRuns);
                        SimLogger.Low($"[BatchRunner] Progress: {completedRuns}/{totalRuns} " +
                                  $"({elapsed:F1}s elapsed, ETA {eta:F1}s)");


                    }
                }
            }

            float totalTime = Time.realtimeSinceStartup - startWall;
            SimLogger.Low($"[BatchRunner] Batch complete: {totalRuns} runs in {totalTime:F1}s");
            isBatchRunning = false;

            if (Application.isBatchMode)
            {
                SimLogger.Low("[BatchRunner] Headless mode — quitting application.");
                Application.Quit();
            }
        }

        /// @brief Runs one (config, rule) episode to completion, yielding each frame.
        /// @brief Runs one (config, rule) episode to completion, yielding each frame.
        private IEnumerator RunSingleEpisode(FJSSPConfig config, DispatchingRule rule)
        {
            //bridge.AutoStartOnPlay = true;
            EpisodeResult runResult = null;
            UnityEngine.Events.UnityAction<EpisodeResult> onFinish = res => runResult = res;
            bridge.OnEpisodeFinished.AddListener(onFinish);
            // Phase 1: Set the agent's heuristic to the target rule
            if (agent != null)
                agent.SetHeuristicRule(rule);

            // Phase 2: Load config (this sets IsFactoryReady = false internally)
            bridge.LoadConfig(config);

            // Phase 3: Give the agent its single-use ticket to start the episode
            if (agent != null)
                agent.ArmAndStart();

            // Phase 4: Wait for ML-Agents to trigger OnEpisodeBegin() on the next FixedUpdate
            // This will call bridge.StartEpisode(), which spawns the factory and sets episodeActive = true
            while (!bridge.IsEpisodeActive)
            {
                yield return null;
            }

            // Phase 5: The simulation is now running. Wait for the orchestrator to finish it.
            while (bridge.IsEpisodeActive)
            {
                yield return null;
            }

            int totalOps = 0;
            if (bridge.Jobs != null)
            {
                foreach (var job in bridge.Jobs.AllJobs)
                    totalOps += job.TotalOperations;
            }

            if (runResult != null)
            {
                ResultsLogger.LogEpisode(
                    ruleName: rule.ToString(),
                    seed: config.Seed,
                    makespan: runResult.Makespan,
                    jobCount: config.JobCount,
                    machineCount: config.MachineTypeLayout.Length,
                    totalOps: totalOps,
                    decisionCount: runResult.DecisionPoints,
                    totalReward: runResult.TotalReward,
                    averageTimeScale: Time.timeScale
                );
            }

            // Small cooldown to let physics and ML-Agents buffers settle between episodes
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

            SimLogger.LogError("[BatchRunner] No batch config source available.");
            return Array.Empty<FJSSPConfig>();
        }

        /// <summary>
        /// Parses a comma-separated rule list from the -rules CLI argument.
        /// Returns AllRules if the argument is absent or unparseable.
        /// Example: -rules SPT_SMPT,LPT_MMUR,SRT_SRWT
        /// </summary>
        private static DispatchingRule[] ParseRulesArg(string arg)
        {
            if (string.IsNullOrEmpty(arg)) return AllRules;

            var result = new List<DispatchingRule>();
            foreach (string token in arg.Split(','))
            {
                if (Enum.TryParse(token.Trim(), ignoreCase: true, out DispatchingRule rule))
                    result.Add(rule);
                else
                    SimLogger.LogWarning($"[BatchRunner] Unknown rule in -rules arg: '{token}'");
            }
            return result.Count > 0 ? result.ToArray() : AllRules;
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
            SimLogger.LogError($"[BatchRunner] {message}");
            if (Application.isBatchMode)
                Application.Quit(1);
        }
    }
}