using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Assets.Scripts.Simulation.Logging;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Jobs;
using Unity.MLAgents;

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
    ///   # Generated job data sweep (deterministic)
    ///   ./capstone.exe -batchmode -nographics -timescale 100 \
    ///      -batchconfig ./BatchConfigs/BatchConfigs.json \
    ///      -outputdir generated_baseline \
    ///      -repeats 1
    ///
    ///   # Single Brandimarte benchmark (deterministic)
    ///   ./capstone.exe -batchmode -nographics -timescale 100 \
    ///      -benchmark ./BatchConfigs/Benchmarks/mk01.json \
    ///      -outputdir brandimarte \
    ///      -repeats 3
    ///
    ///   # All Brandimarte benchmarks (deterministic)
    ///   ./capstone.exe -batchmode -nographics -timescale 100 \
    ///      -benchmarkdir ./BatchConfigs/Benchmarks \
    ///      -outputdir brandimarte \
    ///      -repeats 1
    ///
    ///   # All Brandimarte benchmarks (low disruption stochastic)
    ///   ./capstone.exe -batchmode -nographics -timescale 100 \
    ///      -benchmarkdir ./BatchConfigs/Benchmarks \
    ///      -outputdir brandimarte_stochastic_low \
    ///      -disruption low \
    ///      -repeats 10
    ///
    ///   # All Brandimarte benchmarks (high disruption stochastic)
    ///   ./capstone.exe -batchmode -nographics -timescale 100 \
    ///      -benchmarkdir ./BatchConfigs/Benchmarks \
    ///      -outputdir brandimarte_stochastic_high \
    ///      -disruption high \
    ///      -repeats 10
    /// @endcode
    ///
    /// Attach this MonoBehaviour to the same GameObject as SimulationBridge.
    /// In headless mode it takes over episode lifecycle; in editor mode it does nothing
    /// unless you manually call RunBatch().
    public class HeadlessBatchRunner : MonoBehaviour
    {
        [Header("References")]
        //[SerializeField] private FactoryOrchestrator orchestrator;
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
            DispatchingRule.SDT_SRWT,
            DispatchingRule.Random
        };

        /// <summary>
        /// Flag indicating whether a batch run is currently in progress.
        /// </summary>
        private bool isBatchRunning;

        /// <summary>
        /// Total number of runs scheduled for the current batch.
        /// </summary>
        private int totalRuns;

        /// <summary>
        /// Number of runs completed so far in the current batch.
        /// </summary>
        private int completedRuns;

        /// <summary>
        /// Active dispatching rules to sweep, filtered by the -rules CLI argument.
        /// </summary>
        private DispatchingRule[] activeRules;

        /// <summary>
        /// Wall-clock start time for the current batch, used for ETA calculations.
        /// </summary>
        private float startWall;

        // ─────────────────────────────────────────────────────────
        //  Unity Lifecycle
        // ─────────────────────────────────────────────────────────

        private void Start()
        {
            if (Academy.Instance.IsCommunicatorOn)
            {
                SimLogger.Medium("[BatchRunner] ML-Agents communicator detected — disabling batch runner.");
                enabled = false;
                return;
            }

            string batchPath = GetCLIArg("-batchconfig");
            string benchmarkPath = GetCLIArg("-benchmark");
            string benchmarkDirPath = GetCLIArg("-benchmarkdir");

            // Auto-start in batchmode, or if any config source was explicitly passed
            if (!Application.isBatchMode
                && string.IsNullOrEmpty(batchPath)
                && string.IsNullOrEmpty(benchmarkPath)
                && string.IsNullOrEmpty(benchmarkDirPath))
                return;

            // ── Shared setup ─────────────────────────────────────────

            // Timescale
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

            // Rules filter
            activeRules = ParseRulesArg(GetCLIArg("-rules"));
            SimLogger.Low($"[BatchRunner] Active rules ({activeRules.Length}): " +
                          string.Join(", ", activeRules));

            // Output suffix
            string suffix = GetCLIArg("-outputsuffix") ?? string.Empty;
            if (!string.IsNullOrEmpty(suffix))
                ResultsLogger.SetFilenameSuffix(suffix);

            // Output subdirectory
            string outputDir = GetCLIArg("-outputdir");
            if (!string.IsNullOrEmpty(outputDir))
            {
                ResultsLogger.SetSubdirectory(outputDir);
                SimLogger.Low($"[BatchRunner] Results subdirectory: {outputDir}");
            }

            // Repeats
            int repeats = 1;
            string repeatsStr = GetCLIArg("-repeats");
            if (!string.IsNullOrEmpty(repeatsStr))
                int.TryParse(repeatsStr, out repeats);

            // Disruption level — only applies to benchmark modes (not generated batch)
            StochasticDisruption disruption = StochasticDisruption.None;
            string disruptionStr = GetCLIArg("-disruption");
            if (!string.IsNullOrEmpty(disruptionStr))
            {
                if (Enum.TryParse(disruptionStr, ignoreCase: true, out StochasticDisruption parsed))
                    disruption = parsed;
                else
                    SimLogger.LogWarning($"[BatchRunner] Unknown -disruption value '{disruptionStr}'. " +
                                         "Valid values: none, low, high. Defaulting to none.");
            }
            SimLogger.Low($"[BatchRunner] Disruption mode: {disruption}");

            // ── Route to the correct coroutine ──────────────────────

            if (!string.IsNullOrEmpty(benchmarkDirPath))
            {
                StartCoroutine(RunMultiBenchmarkCoroutine(benchmarkDirPath, repeats, disruption));
            }
            else if (!string.IsNullOrEmpty(benchmarkPath))
            {
                StartCoroutine(RunBenchmarkCoroutine(benchmarkPath, repeats, disruption));
            }
            else
            {
                // Generated job data — disruption flag is ignored here since
                // stochastic params for generated configs come from FJSSPConfig.Stochastic
                // set directly in the batch JSON (not calibrated from instance data).
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
        //  Core Batch Loop (generated job data)
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Coroutine that executes a batch of simulation runs over generated job data configurations.
        /// </summary>
        /// <param name="configs">Array of FJSSPConfig objects defining each simulation scenario.</param>
        /// <param name="repeats">Number of times to repeat each (config, rule) combination.</param>
        /// <returns>IEnumerator for coroutine execution.</returns>
        /// <remarks>
        /// Iterates over all configs, repeats, and active rules in nested order. Each run is logged
        /// and its results appended to the results CSV via ResultsLogger.
        /// </remarks>
        private IEnumerator RunBatchCoroutine(FJSSPConfig[] configs, int repeats)
        {
            isBatchRunning = true;
            if (activeRules == null || activeRules.Length == 0)
                activeRules = AllRules;

            totalRuns = configs.Length * activeRules.Length * repeats;
            completedRuns = 0;

            SimLogger.Low($"[BatchRunner] Starting batch: {configs.Length} configs x " +
                          $"{activeRules.Length} rules x {repeats} repeats = {totalRuns} total runs");

            startWall = Time.realtimeSinceStartup;

            foreach (var baseConfig in configs)
            {
                for (int rep = 0; rep < repeats; rep++)
                {
                    foreach (var rule in activeRules)
                    {
                        FJSSPConfig runConfig = CloneWithSeed(baseConfig, baseConfig.Seed + rep);

                        SimLogger.Low($"[BatchRunner] Run {completedRuns + 1}/{totalRuns}: " +
                                      $"config={runConfig.Name} rule={rule} seed={runConfig.Seed}");

                        yield return RunSingleEpisode(runConfig, rule);

                        completedRuns++;
                        LogProgress();
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

        // ─────────────────────────────────────────────────────────
        //  Multi-Benchmark Loop (all .json files in a directory)
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Coroutine that runs benchmarks from all JSON files in a specified directory.
        /// </summary>
        /// <param name="dirPath">Path to the directory containing benchmark JSON files.</param>
        /// <param name="repeats">Number of times to repeat each (benchmark, rule) combination.</param>
        /// <param name="disruption">The stochastic disruption level to apply to benchmarks.</param>
        /// <returns>IEnumerator for coroutine execution.</returns>
        /// <remarks>
        /// Loads each .json file in the directory as a benchmark, sorts files alphabetically,
        /// and runs all (benchmark, rule, repeat) combinations. Supports stochastic disruption
        /// calibration via BrandimartLoader.LoadDeferredWithStochastic.
        /// </remarks>
        private IEnumerator RunMultiBenchmarkCoroutine(string dirPath, int repeats,
                                                        StochasticDisruption disruption = StochasticDisruption.None)
        {
            isBatchRunning = true;
            if (activeRules == null || activeRules.Length == 0)
                activeRules = AllRules;

            if (!Directory.Exists(dirPath))
            {
                QuitWithError($"Benchmark directory not found: {dirPath}");
                yield break;
            }

            string[] files = Directory.GetFiles(dirPath, "*.json");
            Array.Sort(files, StringComparer.OrdinalIgnoreCase);

            if (files.Length == 0)
            {
                QuitWithError($"No .json files found in {dirPath}");
                yield break;
            }

            var benchmarks = new List<(string path, FJSSPConfig config,
                Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs)>();

            foreach (string file in files)
            {
                var (config, buildJobs) = LoadBenchmark(file, disruption);
                if (config != null)
                {
                    benchmarks.Add((file, config, buildJobs));
                    SimLogger.Low($"[BatchRunner] Loaded benchmark: {config.Name} " +
                                  $"({config.JobCount} jobs, {config.MachineTypeLayout.Length} machines) " +
                                  $"disruption={disruption}");
                }
                else
                {
                    SimLogger.LogWarning($"[BatchRunner] Skipping invalid benchmark: {file}");
                }
            }

            totalRuns = benchmarks.Count * activeRules.Length * repeats;
            completedRuns = 0;

            SimLogger.Low($"[BatchRunner] Multi-benchmark: {benchmarks.Count} files x " +
                          $"{activeRules.Length} rules x {repeats} repeats = {totalRuns} total runs " +
                          $"[disruption={disruption}]");

            startWall = Time.realtimeSinceStartup;

            foreach (var (path, config, buildJobs) in benchmarks)
            {
                SimLogger.Low($"[BatchRunner] ─── {config.Name} " +
                              $"({config.JobCount}j × {config.MachineTypeLayout.Length}m) ───");

                yield return RunBenchmarkEpisodes(config, buildJobs, repeats);
            }

            float totalTime = Time.realtimeSinceStartup - startWall;
            SimLogger.Low($"[BatchRunner] All benchmarks complete: {totalRuns} runs in {totalTime:F1}s");
            isBatchRunning = false;

            if (Application.isBatchMode)
            {
                SimLogger.Low("[BatchRunner] Headless mode — quitting application.");
                Application.Quit();
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Single Benchmark File (entry point for -benchmark)
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Coroutine that runs a single benchmark file through all rule and repeat combinations.
        /// </summary>
        /// <param name="jsonPath">Path to the benchmark JSON file.</param>
        /// <param name="repeats">Number of times to repeat each rule application.</param>
        /// <param name="disruption">The stochastic disruption level to apply.</param>
        /// <returns>IEnumerator for coroutine execution.</returns>
        /// <remarks>
        /// Entry point for the -benchmark CLI flag. Loads the benchmark, then delegates to
        /// RunBenchmarkEpisodes for episode execution.
        /// </remarks>
        private IEnumerator RunBenchmarkCoroutine(string jsonPath, int repeats,
                                                   StochasticDisruption disruption = StochasticDisruption.None)
        {
            isBatchRunning = true;
            if (activeRules == null || activeRules.Length == 0)
                activeRules = AllRules;

            var (config, buildJobs) = LoadBenchmark(jsonPath, disruption);
            if (config == null)
            {
                QuitWithError($"Failed to load benchmark: {jsonPath}");
                yield break;
            }

            totalRuns = activeRules.Length * repeats;
            completedRuns = 0;

            SimLogger.Low($"[BatchRunner] Benchmark: {config.Name}, " +
                          $"{activeRules.Length} rules x {repeats} repeats = {totalRuns} runs " +
                          $"[disruption={disruption}]");

            startWall = Time.realtimeSinceStartup;

            yield return RunBenchmarkEpisodes(config, buildJobs, repeats);

            float totalTime = Time.realtimeSinceStartup - startWall;
            SimLogger.Low($"[BatchRunner] Benchmark complete: {totalRuns} runs in {totalTime:F1}s");
            isBatchRunning = false;

            if (Application.isBatchMode)
            {
                SimLogger.Low("[BatchRunner] Headless mode — quitting application.");
                Application.Quit();
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Benchmark Episode Runner (shared by single and multi)
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Runs all episode combinations for a single benchmark configuration.
        /// </summary>
        /// <param name="config">The benchmark configuration to simulate.</param>
        /// <param name="buildJobs">Factory function that creates job definitions from machine layout.</param>
        /// <param name="repeats">Number of repeats per dispatching rule.</param>
        /// <returns>IEnumerator for coroutine execution.</returns>
        /// <remarks>
        /// For each repeat and rule combination: clones the config with an adjusted seed, sets up
        /// the agent heuristic, loads the factory and prebuilt jobs, runs the episode to completion,
        /// and logs results. A short delay (0.1s) is yielded between runs for cleanup.
        /// </remarks>
        private IEnumerator RunBenchmarkEpisodes(
            FJSSPConfig config,
            Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs,
            int repeats)
        {
            for (int rep = 0; rep < repeats; rep++)
            {
                foreach (var rule in activeRules)
                {
                    FJSSPConfig runConfig = CloneWithSeed(config, config.Seed + rep);
                    runConfig.dispatchingRule = rule;
                    SimLogger.Low($"[BatchRunner] Run {completedRuns + 1}/{totalRuns}: " +
                                  $"benchmark={runConfig.Name} rule={rule} seed={runConfig.Seed}");

                    EpisodeRecord runResult = null;
                    UnityEngine.Events.UnityAction<EpisodeRecord> onFinish = res => runResult = res;
                    FactoryOrchestrator.Instance.OnEpisodeFinished.AddListener(onFinish);

                    if (agent != null)
                        agent.SetHeuristicRule(rule);

                    FactoryOrchestrator.Instance.LoadConfig(runConfig);
                    FactoryOrchestrator.Instance.SpawnFactory();

                    var jobs = buildJobs(FactoryOrchestrator.Instance.CachedMachinesByType);
                    FactoryOrchestrator.Instance.LoadPrebuiltJobs(jobs);

                    if (agent != null)
                        agent.ArmAndStart();

                    while (!FactoryOrchestrator.Instance.IsEpisodeActive)
                        yield return null;

                    while (FactoryOrchestrator.Instance.IsEpisodeActive)
                        yield return null;

                    if (runResult != null)
                        ResultsLogger.LogAll(runResult);

                    FactoryOrchestrator.Instance.OnEpisodeFinished.RemoveListener(onFinish);

                    completedRuns++;
                    LogProgress();

                    yield return new WaitForSecondsRealtime(0.1f);
                }
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Single Episode Runner (used by generated batch loop)
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Runs a single simulation episode for a generated job data configuration.
        /// </summary>
        /// <param name="config">The FJSSPConfig defining the simulation scenario.</param>
        /// <param name="rule">The dispatching rule to apply during the episode.</param>
        /// <returns>IEnumerator for coroutine execution.</returns>
        /// <remarks>
        /// Loads the config, sets the agent's heuristic rule, starts the episode, waits for
        /// completion, logs results, and yields a short delay for cleanup. Used by the
        /// generated batch loop (RunBatchCoroutine).
        /// </remarks>
        private IEnumerator RunSingleEpisode(FJSSPConfig config, DispatchingRule rule)
        {
            EpisodeRecord runResult = null;
            UnityEngine.Events.UnityAction<EpisodeRecord> onFinish = res => runResult = res;
            FactoryOrchestrator.Instance.OnEpisodeFinished.AddListener(onFinish);

            if (agent != null)
                agent.SetHeuristicRule(rule);

            FactoryOrchestrator.Instance.LoadConfig(config);

            if (agent != null)
                agent.ArmAndStart();

            while (!FactoryOrchestrator.Instance.IsEpisodeActive)
                yield return null;

            while (FactoryOrchestrator.Instance.IsEpisodeActive)
                yield return null;

            if (runResult != null)
                ResultsLogger.LogAll(runResult);

            FactoryOrchestrator.Instance.OnEpisodeFinished.RemoveListener(onFinish);

            yield return new WaitForSecondsRealtime(0.1f);
        }

        // ─────────────────────────────────────────────────────────
        //  Helpers
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Loads a benchmark file, attaching a calibrated StochasticConfig
        /// when disruption != None. Single point of truth for the loader call
        /// so both single and multi benchmark coroutines stay in sync.
        /// </summary>
        private static (FJSSPConfig config,
                         Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs)
            LoadBenchmark(string jsonPath, StochasticDisruption disruption)
        {
            return disruption == StochasticDisruption.None
                ? BrandimartLoader.LoadDeferred(jsonPath)
                : BrandimartLoader.LoadDeferredWithStochastic(jsonPath, disruption);
        }

        /// <summary>
        /// Logs the current batch progress with elapsed time and estimated time of completion.
        /// </summary>
        /// <remarks>
        /// Calculates ETA based on average time per completed run. Logs at the Low priority
        /// level for verbose batch monitoring.
        /// </remarks>
        private void LogProgress()
        {
            float elapsed = Time.realtimeSinceStartup - startWall;
            float eta = completedRuns > 0
                ? (elapsed / completedRuns) * (totalRuns - completedRuns)
                : 0f;
            SimLogger.Low($"[BatchRunner] Progress: {completedRuns}/{totalRuns} " +
                          $"({elapsed:F1}s elapsed, ETA {eta:F1}s)");
        }

        /// <summary>
        /// Creates a shallow clone of an FJSSPConfig with a modified seed value.
        /// </summary>
        /// <param name="source">The source configuration to clone.</param>
        /// <param name="newSeed">The new seed value for the cloned config.</param>
        /// <returns>A new FJSSPConfig instance with the updated seed and cloned machine layout.</returns>
        /// <remarks>
        /// The Stochastic config is shared across repeats by design, as stochastic parameters
        /// are defined in the batch JSON and should remain consistent across repeat runs.
        /// The MachineTypeLayout is deep-cloned via Array.Clone to prevent cross-contamination.
        /// </remarks>
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
                AGVCount = source.AGVCount,
                ProcTimeParams = source.ProcTimeParams,
                Stochastic = source.Stochastic,
            };
        }

        /// <summary>
        /// Loads FJSSPConfig arrays from the CLI-specified path or fallback sources.
        /// </summary>
        /// <param name="cliPath">Path to the batch config JSON file from CLI arguments.</param>
        /// <returns>Array of FJSSPConfig objects, or empty array if no source is available.</returns>
        /// <remarks>
        /// Resolution order: 1) CLI -batchconfig path, 2) serialized fallback BatchJson field,
        /// 3) error and empty array if neither is provided.
        /// </remarks>
        private FJSSPConfig[] LoadConfigs(string cliPath)
        {
            if (!string.IsNullOrEmpty(cliPath))
                return ConfigLoader.LoadBatch(cliPath);

            if (fallbackBatchJson != null)
                return ConfigLoader.ParseBatch(fallbackBatchJson.text);

            SimLogger.LogError("[BatchRunner] No batch config source available.");
            return Array.Empty<FJSSPConfig>();
        }

        /// <summary>
        /// Parses the -rules CLI argument into an array of DispatchingRule values.
        /// </summary>
        /// <param name="arg">Raw string from the -rules CLI flag (comma-separated rule names).</param>
        /// <returns>Array of parsed DispatchingRule values, or AllRules if arg is empty/invalid.</returns>
        /// <remarks>
        /// Parses comma-separated rule names case-insensitively. Invalid rule names are logged
        /// as warnings and skipped. Returns AllRules if no valid rules are found or if arg is null/empty.
        /// </remarks>
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

        /// <summary>
        /// Retrieves a value argument from the command line by key name.
        /// </summary>
        /// <param name="key">The CLI flag key (e.g., "-batchconfig").</param>
        /// <returns>The value following the key, or null if not found.</returns>
        /// <remarks>
        /// Parses Environment.GetCommandLineArgs() looking for key/value pairs. The value
        /// returned is the token immediately following the key.
        /// </remarks>
        private static string GetCLIArg(string key)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == key)
                    return args[i + 1];
            return null;
        }

        /// <summary>
        /// Logs an error message and quits the application in batch mode.
        /// </summary>
        /// <param name="message">The error message to log.</param>
        /// <remarks>
        /// In batch mode, calls Application.Quit(1) after logging. In editor mode, the
        /// function logs the error but does not quit, allowing manual inspection.
        /// </remarks>
        private void QuitWithError(string message)
        {
            SimLogger.LogError($"[BatchRunner] {message}");
            if (Application.isBatchMode)
                Application.Quit(1);
        }
    }
}
