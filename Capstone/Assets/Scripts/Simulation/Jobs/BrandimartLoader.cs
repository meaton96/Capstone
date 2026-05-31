using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Newtonsoft.Json.Linq;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Stochastic;
using Assets.Scripts.Simulation.Logging;

namespace Assets.Scripts.Simulation.Jobs
{
    /// <summary>
    /// Loads Brandimarte FJSP benchmark instances from JSON and produces
    /// FJSSPJobDefinition arrays and FJSSPConfig objects that integrate with
    /// the SimulationBridge pipeline.
    /// </summary>
    /// <remarks>
    /// Machine mapping uses round-robin assignment across MachineType values:
    /// BM 0 → Mill, BM 1 → Lathe, BM 2 → Weld, BM 3 → Inspect,
    /// BM 4 → Assemble, BM 5 → Mill (2nd), BM 6 → Lathe (2nd), ...
    ///
    /// Stochastic calibration derives WeibullLambda from instance processing
    /// time statistics so mean TTF is a safe multiple of mean operation cycle
    /// time, preventing infinite-loop failures where TTF < processing time.
    /// </remarks>
    public static class BrandimartLoader
    {
        private static readonly MachineType[] AllTypes =
            (MachineType[])Enum.GetValues(typeof(MachineType));

        // ── Stochastic calibration constants ─────────────────────────────────

        /// <summary>Estimated AGV round-trip travel overhead per operation in simulation seconds.</summary>
        private const float AGV_TRAVEL_OVERHEAD = 60f;

        /// <summary>
        /// Weibull mean factor for k=1.5: mean TTF = lambda × Γ(1 + 1/k) ≈ lambda × 0.9027.
        /// </summary>
        private const float WEIBULL_MEAN_FACTOR = 0.9027f;

        /// <summary>Low disruption: mean TTF is this multiple of mean cycle time.</summary>
        private const float LOW_TTF_FACTOR = 8f;

        /// <summary>
        /// High disruption: mean TTF is this multiple of mean cycle time.
        /// Clamped so mean TTF exceeds max_cycle_time to prevent infinite restart loops.
        /// </summary>
        private const float HIGH_TTF_FACTOR = 3f;

        /// <summary>
        /// Repair log-normal sigma parameter. Repair durations are kept proportional
        /// to mean processing time so repairs are disruptive but not job-blocking.
        /// </summary>
        private const float REPAIR_SIGMA = 0.4f;

        /// <summary>
        /// Scales Brandimarte abstract time units to sim-seconds.
        /// Brandimarte (1993) units are dimensionless integers typically in [1, 20].
        /// A factor of 300 maps 1 unit → 5 minutes, giving realistic industrial
        /// processing times relative to AGV travel overhead of ~60 seconds.
        /// Adjust based on desired utilization target.
        /// </summary>
        private const float PROC_TIME_SCALE = 20f;

        // ─────────────────────────────────────────────────────────────────────
        //  Public: one-shot convenience
        // ─────────────────────────────────────────────────────────────────────

        /// <summary>
        /// Loads a Brandimarte benchmark instance and returns a deferred configuration
        /// and job builder. Uses deterministic mode (no stochastic disruptions) by default.
        /// </summary>
        /// <param name="jsonPath">Path to the JSON benchmark file.</param>
        /// <param name="seed">Random seed for reproducibility.</param>
        /// <returns>A tuple of FJSSPConfig and a job builder function.</returns>
        public static (FJSSPConfig config,
                        Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs)
            LoadDeferred(string jsonPath, int seed = 42, int agvCountOverride = -1)
            => LoadDeferredInternal(jsonPath, seed, StochasticDisruption.None, agvCountOverride);

        /// <summary>
        /// Loads a Brandimarte benchmark instance with calibrated stochastic disruptions.
        /// WeibullLambda and repair parameters are derived from the instance's processing
        /// time distribution to ensure realistic failure behavior without infinite
        /// operation-restart loops.
        /// </summary>
        /// <param name="jsonPath">Path to the JSON benchmark file.</param>
        /// <param name="disruption">The disruption level (None, Low, or High).</param>
        /// <param name="seed">Random seed for reproducibility.</param>
        /// <returns>A tuple of FJSSPConfig and a job builder function.</returns>
        public static (FJSSPConfig config,
                        Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs)
            LoadDeferredWithStochastic(string jsonPath, StochasticDisruption disruption,
                                       int seed = 42, int agvCountOverride = -1)
            => LoadDeferredInternal(jsonPath, seed, disruption, agvCountOverride);

        // ─────────────────────────────────────────────────────────────────────
        //  Internal loader
        // ─────────────────────────────────────────────────────────────────────

        /// <summary>
        /// Internal loader that parses JSON and constructs the configuration and deferred job builder.
        /// </summary>
        /// <param name="jsonPath">Path to the JSON benchmark file.</param>
        /// <param name="seed">Random seed for reproducibility.</param>
        /// <param name="disruption">The stochastic disruption level.</param>
        /// <returns>A tuple of FJSSPConfig and a job builder function.</returns>
        private static (FJSSPConfig config,
                         Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs)
            LoadDeferredInternal(string jsonPath, int seed, StochasticDisruption disruption,
                                 int agvCountOverride = -1)
        {
            if (!File.Exists(jsonPath))
            {
                SimLogger.LogError($"[BrandimartLoader] File not found: {jsonPath}");
                return (null, null);
            }

            string json = File.ReadAllText(jsonPath);
            string name = Path.GetFileNameWithoutExtension(jsonPath);

            FJSSPConfig config = BuildConfig(json, name, seed, disruption, agvCountOverride);
            if (config == null) return (null, null);

            return (config, machinesByType => BuildJobs(json, machinesByType));
        }

        /// <summary>
        /// Builds the FJSSPConfig from JSON content, including stochastic parameter
        /// calibration based on processing time statistics.
        /// </summary>
        /// <param name="json">Raw JSON string of the benchmark instance.</param>
        /// <param name="name">Benchmark name (derived from filename).</param>
        /// <param name="seed">Random seed for reproducibility.</param>
        /// <param name="disruption">The stochastic disruption level.</param>
        /// <returns>A configured FJSSPConfig, or null on failure.</returns>
        private static FJSSPConfig BuildConfig(string json, string name, int seed,
                                                StochasticDisruption disruption,
                                                int agvCountOverride = -1)
        {
            JObject root = JObject.Parse(json);
            int numMachines = root["machines"].Value<int>();
            JArray jobsArray = (JArray)root["jobs"];

            int numTypes = AllTypes.Length;
            int machinesPerType = Mathf.CeilToInt((float)numMachines / numTypes);
            int agvCount = agvCountOverride > 0
                ? agvCountOverride
                : Mathf.CeilToInt(machinesPerType * 1.5f);

            var typeLayout = new MachineType[numMachines];
            for (int bm = 0; bm < numMachines; bm++)
                typeLayout[bm] = AllTypes[bm % numTypes];
            int totalOps = 0;
            // ── Scan processing times to derive statistics ────────────────────
            int minOps = int.MaxValue, maxOps = 0;
            float minProc = float.MaxValue, maxProc = 0f;
            double sumProc = 0.0;
            int countProc = 0;

            foreach (JArray job in jobsArray)
            {
                minOps = Mathf.Min(minOps, job.Count);
                maxOps = Mathf.Max(maxOps, job.Count);
                totalOps += job.Count;
                foreach (JArray op in job)
                    foreach (JObject opt in op)
                    {
                        float p = opt["processing"].Value<float>() * PROC_TIME_SCALE;
                        minProc = Mathf.Min(minProc, p);
                        maxProc = Mathf.Max(maxProc, p);
                        sumProc += p;
                        countProc++;
                    }
            }

            float meanProc = countProc > 0 ? (float)(sumProc / countProc) : maxProc;

            // ── Derive stochastic parameters from processing statistics ───────
            StochasticConfig stochastic = disruption == StochasticDisruption.None
                ? null
                : BuildStochasticConfig(disruption, meanProc, maxProc);

            if (stochastic != null)
            {
                SimLogger.Low($"[BrandimartLoader] Stochastic config for {name} " +
                    $"(disruption={disruption}): " +
                    $"meanProc={meanProc:F1}s maxProc={maxProc:F1}s " +
                    $"WeibullLambda={stochastic.WeibullLambda:F1} " +
                    $"RepairMu={stochastic.RepairLogMu:F2} " +
                    $"meanTTF≈{stochastic.WeibullLambda * WEIBULL_MEAN_FACTOR:F1}s " +
                    $"meanRepair≈{Mathf.Exp(stochastic.RepairLogMu + 0.5f * stochastic.RepairLogSigma * stochastic.RepairLogSigma):F1}s " +
                    $"expectedFailuresPerMachinePerEpisode≈{(meanProc * (float)totalOps / numMachines) / (stochastic.WeibullLambda * WEIBULL_MEAN_FACTOR):F1}");
            }

            return new FJSSPConfig
            {
                Name = disruption == StochasticDisruption.None
                    ? $"brandimarte_{name}"
                    : $"brandimarte_{name}_{disruption.ToString().ToLower()}",
                Seed = seed,
                JobCount = jobsArray.Count,
                MachinesPerType = machinesPerType,
                MachineTypeLayout = typeLayout,
                MinProcTime = minProc,
                MaxProcTime = maxProc,
                MinOpsPerJob = minOps,
                MaxOpsPerJob = maxOps,
                MaxArrivalTime = 0f,
                AGVCount = agvCount,
                Stochastic = stochastic,
            };
        }

        /// <summary>
        /// Derives calibrated stochastic parameters from instance processing statistics.
        /// </summary>
        /// <param name="disruption">The disruption level (None, Low, or High).</param>
        /// <param name="meanProc">Mean processing time across all operations.</param>
        /// <param name="maxProc">Maximum processing time across all operations.</param>
        /// <returns>A calibrated StochasticConfig, or null if disruption is None.</returns>
        /// <remarks>
        /// Core constraint: mean TTF must exceed max_cycle_time to prevent infinite
        /// operation restart loops. max_cycle_time = maxProc + AGV_TRAVEL_OVERHEAD.
        ///
        /// WeibullLambda = max(
        ///     factor × mean_cycle_time / WEIBULL_MEAN_FACTOR,
        ///     (max_cycle_time × 1.5) / WEIBULL_MEAN_FACTOR   ← hard floor
        /// )
        ///
        /// Repair duration is proportional to mean processing time so that repairs
        /// are a meaningful disruption but not longer than typical operations.
        /// </remarks>
        private static StochasticConfig BuildStochasticConfig(
            StochasticDisruption disruption, float meanProc, float maxProc)
        {
            float meanCycleTime = meanProc + AGV_TRAVEL_OVERHEAD;
            float maxCycleTime = maxProc + AGV_TRAVEL_OVERHEAD;

            float ttfFactor = disruption == StochasticDisruption.High
                ? HIGH_TTF_FACTOR
                : LOW_TTF_FACTOR;

            // Desired mean TTF from the factor
            float desiredMeanTtf = ttfFactor * meanCycleTime;

            // Hard floor: mean TTF must be at least 1.5× the longest possible operation
            // so even the worst-case operation can complete before the next failure.
            float minAllowedMeanTtf = maxCycleTime * 1.5f;

            float meanTtf = Mathf.Max(desiredMeanTtf, minAllowedMeanTtf);
            float lambda = meanTtf / WEIBULL_MEAN_FACTOR * PROC_TIME_SCALE;

            // Repair duration: log-normal mu derived so mean repair = repairFraction × meanProc
            // repairFraction: low=0.15, high=0.25
            float repairFraction = disruption == StochasticDisruption.High ? 0.25f : 0.15f;
            float meanRepair = repairFraction * meanProc;

            // log-normal: mean = exp(mu + sigma²/2), so mu = log(mean) - sigma²/2
            float repairMu = Mathf.Log(meanRepair) - 0.5f * REPAIR_SIGMA * REPAIR_SIGMA;

            return new StochasticConfig
            {
                MachineFailuresEnabled = true,
                WeibullK = 1.5f,
                WeibullLambda = lambda,
                RepairLogMu = repairMu,
                RepairLogSigma = REPAIR_SIGMA,
                AGVFailuresEnabled = false,   // Phase 3 — not yet
                DynamicArrivalsEnabled = false,   // Phase 4 — not yet
            };
        }

        /// <summary>
        /// Builds FJSSPJobDefinition arrays from JSON content and machine type mappings.
        /// </summary>
        /// <param name="json">Raw JSON string of the benchmark instance.</param>
        /// <param name="machinesByType">Mapping of MachineType to runtime machine IDs.</param>
        /// <returns>An array of configured FJSSPJobDefinition objects.</returns>
        public static FJSSPJobDefinition[] BuildJobs(
            string json,
            Dictionary<MachineType, List<int>> machinesByType)
        {
            JObject root = JObject.Parse(json);
            int numMachines = root["machines"].Value<int>();
            JArray jobsArray = (JArray)root["jobs"];

            int numTypes = AllTypes.Length;

            var bmMap = new (MachineType type, int idxInType)[numMachines];
            var countPerType = new int[numTypes];

            for (int bm = 0; bm < numMachines; bm++)
            {
                MachineType t = AllTypes[bm % numTypes];
                int typeEnum = (int)t;
                bmMap[bm] = (t, countPerType[typeEnum]);
                countPerType[typeEnum]++;
            }

            var jobs = new FJSSPJobDefinition[jobsArray.Count];

            for (int j = 0; j < jobsArray.Count; j++)
            {
                JArray rawJob = (JArray)jobsArray[j];
                int opCount = rawJob.Count;

                var opSequence = new MachineType[opCount];
                var eligible = new Dictionary<int, float>[opCount];

                for (int o = 0; o < opCount; o++)
                {
                    JArray rawOp = (JArray)rawJob[o];
                    eligible[o] = new Dictionary<int, float>();

                    bool firstSet = false;

                    foreach (JObject opt in rawOp)
                    {
                        int bmIdx = opt["machine"].Value<int>();
                        float pTime = opt["processing"].Value<float>() * PROC_TIME_SCALE;
                        var (type, idxInType) = bmMap[bmIdx];

                        if (!firstSet)
                        {
                            opSequence[o] = type;
                            firstSet = true;
                        }

                        if (machinesByType.TryGetValue(type, out var idList)
                            && idxInType < idList.Count)
                        {
                            eligible[o][idList[idxInType]] = pTime;
                        }
                        else
                        {
                            SimLogger.LogWarning(
                                $"[BrandimartLoader] Job {j} Op {o}: no runtime machine " +
                                $"for BM index {bmIdx} ({type}[{idxInType}])");
                        }
                    }
                }

                jobs[j] = new FJSSPJobDefinition
                {
                    JobId = j,
                    ArrivalTime = 0f,
                    OperationSequence = opSequence,
                    EligibleMachinesPerOp = eligible,
                };
            }

            int runtimeTotal = 0;
            foreach (var list in machinesByType.Values) runtimeTotal += list.Count;

            SimLogger.Low($"[BrandimartLoader] Loaded {jobs.Length} jobs from benchmark " +
                          $"({numMachines} BM machines → {runtimeTotal} runtime machines " +
                          $"across {machinesByType.Count} types)");

            return jobs;
        }
    }

    /// <summary>
    /// Disruption level for stochastic Brandimarte runs.
    /// None  = deterministic (no StochasticConfig attached)
    /// Low   = failures occur but rarely disrupt completion
    /// High  = frequent failures, significant makespan impact
    /// </summary>
    public enum StochasticDisruption
    {
        None,
        Low,
        High,
    }
}