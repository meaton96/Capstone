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
            DispatchingRule.FIFO_SRWT,
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
            // Hand-crafted scenario instances (ScenarioLoader) -- explicit job sets built to
            // force a specific contention structure (a bottleneck machine, two jobs pinned to
            // the same machine, an identical-job batch, ...) rather than i.i.d. random jobs.
            string scenarioPath = GetCLIArg("-scenario");
            string scenarioDirPath = GetCLIArg("-scenariodir");
            // Baseline decision-drain: when set, heuristic decisions drain per-frame instead of
            // one-per-frame. Valid ONLY for heuristic batch runs (no neural policy in the loop).
            // HasCLIFlag: GetCLIArg needs a following argument, so it missed the switch when passed last (audit G7).
            bool baselineDrain = HasCLIFlag("-baselinedrain");
            // Event-based twin export (Logging/DesTwinExport): des_floor.json, des_jobs.json, agv_events.csv.
            DesTwinExport.Enabled = HasCLIFlag("-destrace");
            if (DesTwinExport.Enabled)
                SimLogger.Low("[BatchRunner] DES twin trace ENABLED (des_floor.json, des_jobs.json, agv_events.csv).");
            if (baselineDrain)
                SimLogger.Low("[BatchRunner] Baseline drain mode ENABLED — heuristic decisions " +
                              "drain per frame (removes one-decision-per-frame throttle).");

            // Auto-start in batchmode, or if any config source was explicitly passed
            if (!Application.isBatchMode
                && string.IsNullOrEmpty(batchPath)
                && string.IsNullOrEmpty(benchmarkPath)
                && string.IsNullOrEmpty(benchmarkDirPath)
                && string.IsNullOrEmpty(scenarioPath)
                && string.IsNullOrEmpty(scenarioDirPath))
                return;

            // ── Shared setup ─────────────────────────────────────────

            // Timescale
            string unknown = FirstUnknownFlag();
            if (unknown != null)
            {
                QuitWithError($"Unknown command-line flag '{unknown}' (typo?). Known flags: " +
                              string.Join(" ", KnownFlags));
                enabled = false;
                return;
            }

            string timeScaleStr = GetCLIArg("-timescale");
            float parsedScale = 100f;
            if (!string.IsNullOrEmpty(timeScaleStr) &&
                (!float.TryParse(timeScaleStr, System.Globalization.NumberStyles.Float,
                                 System.Globalization.CultureInfo.InvariantCulture, out parsedScale) ||
                 parsedScale <= 0f))
            {
                QuitWithError($"Invalid -timescale '{timeScaleStr}': expected a number > 0.");
                enabled = false;
                return;
            }
            if (!string.IsNullOrEmpty(timeScaleStr))
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
            int parsedAgv = -1;
            string agvCountStr = GetCLIArg("-agvcount");
            if (!string.IsNullOrEmpty(agvCountStr) &&
                (!int.TryParse(agvCountStr, System.Globalization.NumberStyles.Integer,
                               System.Globalization.CultureInfo.InvariantCulture, out parsedAgv) || parsedAgv < 1))
            {
                QuitWithError($"Invalid -agvcount '{agvCountStr}': expected an integer >= 1.");
                enabled = false;
                return;
            }
            if (!string.IsNullOrEmpty(agvCountStr))
            {
                agvCountOverride = parsedAgv;
                SimLogger.Low($"[BatchRunner] AGV count override: {agvCountOverride}");
            }

            // CLI overrides that select a variant of the simulation are validated up front. A bad value
            // must end the process (exit 1): an exception here is only logged by Unity and would leave a
            // batch-mode worker idling forever, stalling a whole sweep.
            try
            {
                // Reservation protocol override (applied to every config by FactoryOrchestrator.ApplyConfigOverrides).
                // Validated here so a typo aborts before any run instead of silently using the default.
                string reservationStr = GetCLIArg("-reservation");
                if (!string.IsNullOrEmpty(reservationStr))
                {
                    ReservationProtocolParser.Parse(reservationStr);
                    ConfigOverrides.ReservationProtocol = reservationStr;
                    SimLogger.Low($"[BatchRunner] Reservation protocol override: {reservationStr}");
                }

                // Routing trigger override ("onTransport" | "onReady"), same single-point mechanism.
                string routingStr = GetCLIArg("-routingtrigger");
                if (!string.IsNullOrEmpty(routingStr))
                {
                    RoutingTriggerParser.Parse(routingStr);
                    ConfigOverrides.RoutingTrigger = routingStr;
                    SimLogger.Low($"[BatchRunner] Routing trigger override: {routingStr}");
                }

                // Parking method override ("single" | "multiple" | "lane"), same single-point mechanism.
                string parkingStr = GetCLIArg("-parking");
                if (!string.IsNullOrEmpty(parkingStr))
                {
                    ConfigOverrides.ParkingMethod = ConfigOverrides.ValidatedParkingMethod(parkingStr);
                    SimLogger.Low($"[BatchRunner] Parking method override: {ConfigOverrides.ParkingMethod}");
                }

                // I/O belt dock override ("corner" | "siding"), same single-point mechanism.
                string ioDocksStr = GetCLIArg("-iodocks");
                if (!string.IsNullOrEmpty(ioDocksStr))
                {
                    ConfigOverrides.IoDocks = ConfigOverrides.ValidatedIoDocks(ioDocksStr);
                    SimLogger.Low($"[BatchRunner] I/O docks override: {ConfigOverrides.IoDocks}");
                }

                // Tiled floor overrides (TilingSpec): tile count and release rule, same single-point mechanism.
                string tilesStr = GetCLIArg("-tiles");
                if (!string.IsNullOrEmpty(tilesStr))
                {
                    if (!int.TryParse(tilesStr, out int tiles) || tiles < 1)
                        throw new ArgumentException($"Invalid -tiles '{tilesStr}': expected an integer >= 1.");
                    ConfigOverrides.Tiles = tiles;
                    SimLogger.Low($"[BatchRunner] Tiles override: {tiles}");
                }
                string releaseStr = GetCLIArg("-releaserule");
                if (!string.IsNullOrEmpty(releaseStr))
                {
                    ConfigOverrides.ReleaseRule = TilingSpec.ReleaseToString(TilingSpec.ParseRelease(releaseStr));
                    SimLogger.Low($"[BatchRunner] Release rule override: {ConfigOverrides.ReleaseRule}");
                }
                // "-releaseweights 3,2,1,1,1,1,1" (one per tile) implies "-releaserule weighted".
                string weightsStr = GetCLIArg("-releaseweights");
                if (!string.IsNullOrEmpty(weightsStr))
                {
                    ConfigOverrides.ReleaseWeights = TilingSpec.ParseWeights(weightsStr);
                    ConfigOverrides.ReleaseRule = TilingSpec.ReleaseToString(ReleaseRule.Weighted);
                    SimLogger.Low($"[BatchRunner] Release weights override: {string.Join(",", ConfigOverrides.ReleaseWeights)}");
                }

                // TECT travel price λ (FJSSPConfig.TravelPrice).
                string priceStr = GetCLIArg("-travelprice");
                if (!string.IsNullOrEmpty(priceStr))
                {
                    if (!float.TryParse(priceStr, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float price)
                        || !(price >= 0f) || float.IsInfinity(price))
                        throw new ArgumentException($"Invalid -travelprice '{priceStr}': expected a number >= 0.");
                    ConfigOverrides.TravelPrice = price;
                    SimLogger.Low($"[BatchRunner] Travel price override: {price}");
                }
                // Machine buffer sizes (FJSSPConfig.InputBufferCapacity / OutputBufferCapacity; 0 = unbounded).
                ConfigOverrides.InputBufferCapacity = ParseBufferArg("-inbuf") ?? ConfigOverrides.InputBufferCapacity;
                ConfigOverrides.OutputBufferCapacity = ParseBufferArg("-outbuf") ?? ConfigOverrides.OutputBufferCapacity;
                // Linked tiles (TilingSpec.AgvsPooled / JobsOpen): "-jobscope open" alone implies "-agvassignment pooled".
                string scopeStr = GetCLIArg("-jobscope");
                if (!string.IsNullOrEmpty(scopeStr))
                {
                    ConfigOverrides.JobScope = TilingSpec.ParseJobScope(scopeStr) ? "open" : "tile";
                    SimLogger.Low($"[BatchRunner] Job scope override: {ConfigOverrides.JobScope}");
                }
                string assignStr = GetCLIArg("-agvassignment");
                if (!string.IsNullOrEmpty(assignStr))
                {
                    ConfigOverrides.AgvAssignment = TilingSpec.ParseAgvAssignment(assignStr) ? "pooled" : "tile";
                    SimLogger.Low($"[BatchRunner] AGV assignment override: {ConfigOverrides.AgvAssignment}");
                }

                // Machine flexibility overrides (FJSSPConfig.MachineFlexibilityProbability / SecondaryTimeMultiplier).
                string flexStr = GetCLIArg("-flex");
                if (!string.IsNullOrEmpty(flexStr))
                {
                    if (!float.TryParse(flexStr, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float flex)
                        || flex < 0f || flex > 1f)
                        throw new ArgumentException($"Invalid -flex '{flexStr}': expected a probability in [0, 1].");
                    ConfigOverrides.MachineFlexibility = flex;
                    SimLogger.Low($"[BatchRunner] Machine flexibility override: {flex}");
                }
                string flexMultStr = GetCLIArg("-flexmult");
                if (!string.IsNullOrEmpty(flexMultStr))
                {
                    if (!float.TryParse(flexMultStr, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float mult)
                        || mult <= 0f)
                        throw new ArgumentException($"Invalid -flexmult '{flexMultStr}': expected a number > 0.");
                    ConfigOverrides.SecondaryTimeMultiplier = mult;
                    SimLogger.Low($"[BatchRunner] Secondary time multiplier override: {mult}");
                }

                // Layout override (LayoutSpec name A-J), same single-point mechanism. Validated here so a typo
                // or a layout that is not built yet aborts before any run.
                string layoutStr = GetCLIArg("-layout");
                if (!string.IsNullOrEmpty(layoutStr))
                {
                    ConfigOverrides.Layout = LayoutSpec.FromPreset(layoutStr).Id;
                    SimLogger.Low($"[BatchRunner] Layout override: {ConfigOverrides.Layout}");
                }

            }
            catch (Exception ex)
            {
                SimLogger.LogError($"[BatchRunner] Invalid command-line override: {ex.Message}");
                enabled = false;
                Application.Quit(1);
                return;
            }

            // Repeats
            int repeats = 1;
            string repeatsStr = GetCLIArg("-repeats");
            if (!string.IsNullOrEmpty(repeatsStr) &&
                (!int.TryParse(repeatsStr, System.Globalization.NumberStyles.Integer,
                               System.Globalization.CultureInfo.InvariantCulture, out repeats) || repeats < 1))
            {
                // Used to leave repeats at 0: zero runs, exit code 0, no results (audit G7).
                QuitWithError($"Invalid -repeats '{repeatsStr}': expected an integer >= 1.");
                enabled = false;
                return;
            }

            // Disruption level
            StochasticDisruption disruption = StochasticDisruption.None;
            string disruptionStr = GetCLIArg("-disruption");
            if (!string.IsNullOrEmpty(disruptionStr))
            {
                if (Enum.TryParse(disruptionStr, ignoreCase: true, out StochasticDisruption parsed))
                    disruption = parsed;
                else
                {
                    QuitWithError($"Unknown -disruption '{disruptionStr}'. Expected one of: " +
                                  string.Join(", ", Enum.GetNames(typeof(StochasticDisruption))));
                    enabled = false;
                    return;
                }
            }

            if (baselineDrain && FactoryOrchestrator.Instance != null)
                FactoryOrchestrator.Instance.BaselineDrainMode = true;

            // ── Route to the correct coroutine ──────────────────────

            if (!string.IsNullOrEmpty(benchmarkDirPath))
            {
                StartCoroutine(RunMultiBenchmarkCoroutine(benchmarkDirPath, repeats, disruption, agvCountOverride));
            }
            else if (!string.IsNullOrEmpty(benchmarkPath))
            {
                StartCoroutine(RunBenchmarkCoroutine(benchmarkPath, repeats, disruption, agvCountOverride));
            }
            else if (!string.IsNullOrEmpty(scenarioDirPath))
            {
                StartCoroutine(RunMultiScenarioCoroutine(scenarioDirPath, repeats, agvCountOverride));
            }
            else if (!string.IsNullOrEmpty(scenarioPath))
            {
                StartCoroutine(RunScenarioCoroutine(scenarioPath, repeats, agvCountOverride));
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
                        FJSSPConfig runConfig = baseConfig.CloneWithSeed(baseConfig.Seed + rep);
                        runConfig.dispatchingRule = rule;
                        SimLogger.Low($"[BatchRunner] Run {completedRuns + 1}/{totalRuns}: " +
                                      $"config={runConfig.Name} rule={rule} seed={runConfig.Seed}");

                        yield return RunSingleEpisode(runConfig, rule);
                        if (_aborted) yield break;   // RunSingleEpisode quit the player

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
                if (_aborted) yield break;
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
            if (_aborted) yield break;

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
        //  Hand-Crafted Scenario Loop (single file, via -scenario)
        // ─────────────────────────────────────────────────────────

        private IEnumerator RunScenarioCoroutine(string jsonPath, int repeats, int agvCountOverride = -1)
        {
            isBatchRunning = true;
            if (activeRules == null || activeRules.Length == 0)
                activeRules = AllRules;

            FJSSPConfig config = null;
            Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs = null;
            try
            {
                (config, buildJobs) = ScenarioLoader.LoadDeferred(jsonPath, agvCountOverride: agvCountOverride);
            }
            catch (Exception e)
            {
                // An uncaught exception here kills this coroutine silently — Unity logs it
                // but nothing calls Application.Quit(), so a batch-mode process spins on
                // empty frames at ~100% CPU forever with zero output, looking like a hang
                // rather than a crash. Catch and fail loudly instead.
                SimLogger.LogError($"[BatchRunner] Exception loading scenario {jsonPath}: {e}");
            }
            if (config == null)
            {
                QuitWithError($"Failed to load scenario: {jsonPath}");
                yield break;
            }

            totalRuns = activeRules.Length * repeats;
            completedRuns = 0;

            SimLogger.Low($"[BatchRunner] Scenario: {config.Name}, " +
                          $"{activeRules.Length} rules x {repeats} repeats = {totalRuns} runs");

            startWall = Time.realtimeSinceStartup;

            yield return RunBenchmarkEpisodes(config, buildJobs, repeats);
            if (_aborted) yield break;

            float totalTime = Time.realtimeSinceStartup - startWall;
            SimLogger.Low($"[BatchRunner] Scenario complete: {totalRuns} runs in {totalTime:F1}s");
            isBatchRunning = false;

            if (Application.isBatchMode)
            {
                SimLogger.Low("[BatchRunner] Headless mode — quitting application.");
                Application.Quit();
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Hand-Crafted Scenario Loop (all .json files in a directory, via -scenariodir)
        // ─────────────────────────────────────────────────────────

        private IEnumerator RunMultiScenarioCoroutine(string dirPath, int repeats, int agvCountOverride = -1)
        {
            isBatchRunning = true;
            if (activeRules == null || activeRules.Length == 0)
                activeRules = AllRules;

            if (!Directory.Exists(dirPath))
            {
                QuitWithError($"Scenario directory not found: {dirPath}");
                yield break;
            }

            string[] files = Directory.GetFiles(dirPath, "*.json");
            Array.Sort(files, StringComparer.OrdinalIgnoreCase);

            if (files.Length == 0)
            {
                QuitWithError($"No .json files found in {dirPath}");
                yield break;
            }

            var scenarios = new List<(string path, FJSSPConfig config,
                Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs)>();

            foreach (string file in files)
            {
                FJSSPConfig config = null;
                Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs = null;
                try
                {
                    (config, buildJobs) = ScenarioLoader.LoadDeferred(file, agvCountOverride: agvCountOverride);
                }
                catch (Exception e)
                {
                    // Same reasoning as RunScenarioCoroutine -- an uncaught exception here
                    // would silently kill this coroutine, and with it the whole directory
                    // sweep, leaving the batch-mode process spinning at ~100% CPU forever
                    // with zero output. Log and skip just this file instead.
                    SimLogger.LogError($"[BatchRunner] Exception loading scenario {file}: {e}");
                }
                if (config != null)
                {
                    scenarios.Add((file, config, buildJobs));
                    SimLogger.Low($"[BatchRunner] Loaded scenario: {config.Name} " +
                                  $"({config.JobCount} jobs, {config.MachineTypeLayout.Length} machines, " +
                                  $"{config.AGVCount} AGVs)");
                }
                else
                {
                    SimLogger.LogWarning($"[BatchRunner] Skipping invalid scenario: {file}");
                }
            }

            totalRuns = scenarios.Count * activeRules.Length * repeats;
            completedRuns = 0;

            SimLogger.Low($"[BatchRunner] Multi-scenario: {scenarios.Count} files x " +
                          $"{activeRules.Length} rules x {repeats} repeats = {totalRuns} total runs");

            startWall = Time.realtimeSinceStartup;

            foreach (var (path, config, buildJobs) in scenarios)
            {
                SimLogger.Low($"[BatchRunner] ─── {config.Name} " +
                              $"({config.JobCount}j × {config.MachineTypeLayout.Length}m) ───");

                yield return RunBenchmarkEpisodes(config, buildJobs, repeats);
                if (_aborted) yield break;
            }

            float totalTime = Time.realtimeSinceStartup - startWall;
            SimLogger.Low($"[BatchRunner] All scenarios complete: {totalRuns} runs in {totalTime:F1}s");
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
                    FJSSPConfig runConfig = config.CloneWithSeed(config.Seed + rep);
                    runConfig.dispatchingRule = rule;
                    SimLogger.Low($"[BatchRunner] Run {completedRuns + 1}/{totalRuns}: " +
                                  $"benchmark={runConfig.Name} rule={rule} seed={runConfig.Seed}");

                    EpisodeRecord runResult = null;
                    UnityEngine.Events.UnityAction<EpisodeRecord> onFinish = res => runResult = res;
                    FactoryOrchestrator.Instance.OnEpisodeFinished.AddListener(onFinish);

                    if (agent != null)
                        agent.SetHeuristicRule(rule);

                    bool loadFailed = false;
                    try
                    {
                        FactoryOrchestrator.Instance.LoadConfig(runConfig);
                        FactoryOrchestrator.Instance.SpawnFactory();

                        var jobs = buildJobs(FactoryOrchestrator.Instance.CachedMachinesByType);
                        FactoryOrchestrator.Instance.LoadPrebuiltJobs(jobs);
                    }
                    catch (Exception e)
                    {
                        // An uncaught exception here would kill this coroutine silently --
                        // Unity logs it but nothing calls Application.Quit(), so the batch
                        // process spins on empty frames at ~100% CPU forever with zero
                        // output. Log, skip this one run, and keep going instead.
                        SimLogger.LogError($"[BatchRunner] Exception building run " +
                                            $"{runConfig.Name}/{rule}: {e}");
                        loadFailed = true;
                    }

                    if (loadFailed)
                    {
                        FactoryOrchestrator.Instance.OnEpisodeFinished.RemoveListener(onFinish);
                        completedRuns++;
                        LogProgress();
                        continue;
                    }

                    if (agent != null && !TryOrQuit(agent.ArmAndStart, $"starting {runConfig.Name}/{rule}"))
                        yield break;

                    float startWait = Time.realtimeSinceStartup;
                    while (!FactoryOrchestrator.Instance.IsEpisodeActive)
                    {
                        if (StartTimedOut(startWait, $"{runConfig.Name}/{rule}")) yield break;
                        yield return null;
                    }

                    while (FactoryOrchestrator.Instance.IsEpisodeActive)
                        yield return null;

                    if (runResult != null &&
                        !TryOrQuit(() => ResultsLogger.LogAll(runResult), $"logging {runConfig.Name}/{rule}"))
                        yield break;

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

            if (!TryOrQuit(() => FactoryOrchestrator.Instance.LoadConfig(config), $"loading {config.Name}/{rule}"))
                yield break;

            if (agent != null && !TryOrQuit(agent.ArmAndStart, $"starting {config.Name}/{rule}"))
                yield break;

            float startWait = Time.realtimeSinceStartup;
            while (!FactoryOrchestrator.Instance.IsEpisodeActive)
            {
                if (StartTimedOut(startWait, $"{config.Name}/{rule}")) yield break;
                yield return null;
            }

            while (FactoryOrchestrator.Instance.IsEpisodeActive)
                yield return null;

            if (runResult != null &&
                !TryOrQuit(() => ResultsLogger.LogAll(runResult), $"logging {config.Name}/{rule}"))
                yield break;

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

        /// Runs one batch step; on an exception logs it and quits with exit code 1. An exception
        /// escaping a coroutine only kills the coroutine, and the player then idles forever with no
        /// output (audit G6). Callers `yield break` when this returns false.
        private static bool TryOrQuit(Action action, string what)
        {
            try { action(); return true; }
            catch (Exception e)
            {
                SimLogger.LogError($"[BatchRunner] Exception {what}: {e}. Quitting with exit code 1.");
                _aborted = true;
                Application.Quit(1);
                return false;
            }
        }

        /// Set when TryOrQuit/StartTimedOut has quit the player, so outer batch loops stop too.
        private static bool _aborted;

        /// Real seconds to wait for an armed episode to become active before giving up.
        private const float EPISODE_START_TIMEOUT_SECONDS = 120f;

        /// True (after logging and quitting with exit code 1) if the episode never started.
        private static bool StartTimedOut(float since, string what)
        {
            if (Time.realtimeSinceStartup - since < EPISODE_START_TIMEOUT_SECONDS) return false;
            SimLogger.LogError($"[BatchRunner] Episode {what} did not start within " +
                               $"{EPISODE_START_TIMEOUT_SECONDS:F0}s. Quitting with exit code 1.");
            _aborted = true;
            Application.Quit(1);
            return true;
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

        // private FJSSPConfig CloneWithSeed(FJSSPConfig source, int newSeed)
        // {
        //     return new FJSSPConfig
        //     {
        //         Name = source.Name,
        //         Seed = newSeed,
        //         JobCount = source.JobCount,
        //         MachinesPerType = source.MachinesPerType,
        //         MachineTypeLayout = (MachineType[])source.MachineTypeLayout.Clone(),
        //         MinProcTime = source.MinProcTime,
        //         MaxProcTime = source.MaxProcTime,
        //         MinOpsPerJob = source.MinOpsPerJob,
        //         MaxOpsPerJob = source.MaxOpsPerJob,
        //         AGVCount = source.AGVCount,
        //         ProcTimeParams = source.ProcTimeParams,
        //         Stochastic = source.Stochastic,
        //         dispatchingRule = source.dispatchingRule,
        //         parkingMethod = source.parkingMethod,
        //         preDispatchingMethod = source.preDispatchingMethod,
        //         MachineFlexibilityProbability = source.MachineFlexibilityProbability,
        //     };
        // }

        private FJSSPConfig[] LoadConfigs(string cliPath)
        {
            if (!string.IsNullOrEmpty(cliPath))
                return ConfigLoader.LoadBatch(cliPath);

            if (fallbackBatchJson != null)
                return ConfigLoader.ParseBatch(fallbackBatchJson.text);

            SimLogger.LogError("[BatchRunner] No batch config source available.");
            return Array.Empty<FJSSPConfig>();
        }
        /// <summary>Parses -inbuf / -outbuf (an int >= 0, 0 = unbounded); null when the flag is absent.</summary>
        private static int? ParseBufferArg(string flag)
        {
            string str = GetCLIArg(flag);
            if (string.IsNullOrEmpty(str)) return null;
            if (!int.TryParse(str, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out int n) || n < 0)
                throw new ArgumentException($"Invalid {flag} '{str}': expected an integer >= 0 (0 = unbounded).");
            SimLogger.Low($"[BatchRunner] {flag} override: {n}");
            return n;
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

        /// Every flag the player reads (batch runner, logging, observation caps, visuals, validators).
        /// Keep in sync when adding a flag: anything else starting with a single '-' aborts a batch run.
        private static readonly string[] KnownFlags =
        {
            "-config", "-batchconfig", "-configname", "-benchmark", "-benchmarkdir", "-scenario", "-scenariodir",
            "-baselinedrain", "-rldecisiondrain", "-destrace", "-timescale", "-rules", "-outputsuffix", "-outputdir",
            "-agvcount", "-reservation", "-routingtrigger", "-parking", "-iodocks", "-tiles", "-releaserule",
            "-releaseweights", "-travelprice", "-jobscope", "-agvassignment", "-flex", "-flexmult", "-layout",
            "-inbuf", "-outbuf", "-repeats", "-disruption", "-loglevel", "-decisionlogdir", "-obsmaxjobs",
            "-obsmaxmachines", "-kenneyvisuals", "-primitivevisuals", "-allowunsafefleet", "-validatestochastic",
        };

        /// Unity's own player flags that batch launches use (case-insensitive), plus prefixes of flag families.
        private static readonly string[] UnityFlags =
        {
            "-batchmode", "-nographics", "-logfile", "-quit", "-executemethod", "-projectpath", "-popupwindow",
            "-nolog", "-silent-crashes", "-disable-gpu-skinning", "-single-instance",
        };
        private static readonly string[] UnityFlagPrefixes = { "-screen-", "-force-", "-window-", "-monitor" };

        /// First argument that looks like a single-dash flag but is not one the player or Unity knows, or null.
        /// Unknown flags used to be ignored, so a typo silently ran the default variant (audit G7).
        /// "--" flags (ML-Agents) and negative numbers are values, not checked.
        private static string FirstUnknownFlag()
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 1; i < args.Length; i++)
            {
                string a = args[i];
                if (a.Length < 2 || a[0] != '-' || a[1] == '-' || !char.IsLetter(a[1])) continue;
                if (Array.IndexOf(KnownFlags, a) >= 0) continue;
                string lower = a.ToLowerInvariant();
                if (Array.IndexOf(UnityFlags, lower) >= 0) continue;
                bool prefixed = false;
                foreach (string p in UnityFlagPrefixes) if (lower.StartsWith(p)) { prefixed = true; break; }
                if (prefixed) continue;
                return a;
            }
            return null;
        }

        /// <summary>True if a value-less switch is present anywhere on the command line (GetCLIArg needs a
        ///          following argument, so it misses a switch passed last).</summary>
        private static bool HasCLIFlag(string key) => Array.IndexOf(Environment.GetCommandLineArgs(), key) >= 0;

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