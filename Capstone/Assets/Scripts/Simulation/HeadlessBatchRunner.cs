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
    public class HeadlessBatchRunner : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private SchedulingAgent agent;

        [Header("Fallback Settings (used if no CLI args)")]
        [SerializeField] private TextAsset fallbackBatchJson;
        [SerializeField] private int fallbackRepeats = 1;

        /// @brief All dispatching rules to sweep across.
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

        private bool isBatchRunning;
        private int totalRuns;
        private int completedRuns;
        private DispatchingRule[] activeRules;
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

            // Look for both the new "-config" and the old "-batchconfig" for backward compatibility
            string batchPath = GetCLIArg("-config") ?? GetCLIArg("-batchconfig");
            string configName = GetCLIArg("-configname");
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

            // AGV count override
            int agvCountOverride = -1;
            string agvCountStr = GetCLIArg("-agvcount");
            if (!string.IsNullOrEmpty(agvCountStr) && int.TryParse(agvCountStr, out int parsedAgv))
            {
                agvCountOverride = parsedAgv;
                SimLogger.Low($"[BatchRunner] AGV count override: {agvCountOverride}");
            }

            // Repeats
            int repeats = 1;
            string repeatsStr = GetCLIArg("-repeats");
            if (!string.IsNullOrEmpty(repeatsStr))
                int.TryParse(repeatsStr, out repeats);

            // Disruption level
            StochasticDisruption disruption = StochasticDisruption.None;
            string disruptionStr = GetCLIArg("-disruption");
            if (!string.IsNullOrEmpty(disruptionStr))
            {
                if (Enum.TryParse(disruptionStr, ignoreCase: true, out StochasticDisruption parsed))
                    disruption = parsed;
                else
                    SimLogger.LogWarning($"[BatchRunner] Unknown -disruption '{disruptionStr}'. Defaulting to none.");
            }

            // ── Route to the correct coroutine ──────────────────────

            if (!string.IsNullOrEmpty(benchmarkDirPath))
            {
                StartCoroutine(RunMultiBenchmarkCoroutine(benchmarkDirPath, repeats, disruption, agvCountOverride));
            }
            else if (!string.IsNullOrEmpty(benchmarkPath))
            {
                StartCoroutine(RunBenchmarkCoroutine(benchmarkPath, repeats, disruption, agvCountOverride));
            }
            else
            {
                // Generated job data — load the whole array from the JSON
                FJSSPConfig[] configs = LoadConfigs(batchPath);

                // If the bash script asked for a specific config name, filter down to just that one
                if (!string.IsNullOrEmpty(configName) && configs != null)
                {
                    configs = Array.FindAll(configs, c => c.Name == configName);
                }

                if (configs != null && configs.Length > 0)
                    StartCoroutine(RunBatchCoroutine(configs, repeats));
                else
                    QuitWithError($"No valid configs found matching name '{configName}' in {batchPath}");
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Public API (for editor / UI triggering)
        // ─────────────────────────────────────────────────────────

        public void RunBatch(FJSSPConfig[] configs, int repeats = 1)
        {
            if (isBatchRunning)
            {
                SimLogger.LogWarning("[BatchRunner] Batch already in progress.");
                return;
            }
            StartCoroutine(RunBatchCoroutine(configs, repeats));
        }

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
                        runConfig.dispatchingRule = rule;
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

        private IEnumerator RunMultiBenchmarkCoroutine(string dirPath, int repeats,
                                                        StochasticDisruption disruption = StochasticDisruption.None,
                                                        int agvCountOverride = -1)
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
                var (config, buildJobs) = LoadBenchmark(file, disruption, agvCountOverride);
                if (config != null)
                {
                    benchmarks.Add((file, config, buildJobs));
                    SimLogger.Low($"[BatchRunner] Loaded benchmark: {config.Name} " +
                                  $"({config.JobCount} jobs, {config.MachineTypeLayout.Length} machines, " +
                                  $"{config.AGVCount} AGVs) disruption={disruption}");
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

        private IEnumerator RunBenchmarkCoroutine(string jsonPath, int repeats,
                                                   StochasticDisruption disruption = StochasticDisruption.None,
                                                   int agvCountOverride = -1)
        {
            isBatchRunning = true;
            if (activeRules == null || activeRules.Length == 0)
                activeRules = AllRules;

            var (config, buildJobs) = LoadBenchmark(jsonPath, disruption, agvCountOverride);
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

        private IEnumerator RunSingleEpisode(FJSSPConfig config, DispatchingRule rule)
        {
            EpisodeRecord runResult = null;
            UnityEngine.Events.UnityAction<EpisodeRecord> onFinish = res => runResult = res;
            FactoryOrchestrator.Instance.OnEpisodeFinished.AddListener(onFinish);
            config.dispatchingRule = rule;
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

        private static (FJSSPConfig config,
                         Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs)
            LoadBenchmark(string jsonPath, StochasticDisruption disruption, int agvCountOverride = -1)
        {
            return disruption == StochasticDisruption.None
                ? BrandimartLoader.LoadDeferred(jsonPath, agvCountOverride: agvCountOverride)
                : BrandimartLoader.LoadDeferredWithStochastic(jsonPath, disruption,
                                                               agvCountOverride: agvCountOverride);
        }

        private void LogProgress()
        {
            float elapsed = Time.realtimeSinceStartup - startWall;
            float eta = completedRuns > 0
                ? (elapsed / completedRuns) * (totalRuns - completedRuns)
                : 0f;
            SimLogger.Low($"[BatchRunner] Progress: {completedRuns}/{totalRuns} " +
                          $"({elapsed:F1}s elapsed, ETA {eta:F1}s)");
        }

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
                AGVCount = source.AGVCount,
                ProcTimeParams = source.ProcTimeParams,
                Stochastic = source.Stochastic,
                dispatchingRule = source.dispatchingRule
            };
        }

        private FJSSPConfig[] LoadConfigs(string cliPath)
        {
            if (!string.IsNullOrEmpty(cliPath))
                return ConfigLoader.LoadBatch(cliPath);

            if (fallbackBatchJson != null)
                return ConfigLoader.ParseBatch(fallbackBatchJson.text);

            SimLogger.LogError("[BatchRunner] No batch config source available.");
            return Array.Empty<FJSSPConfig>();
        }

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