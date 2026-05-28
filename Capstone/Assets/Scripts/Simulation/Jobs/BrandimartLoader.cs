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
    /// FJSSPJobDefinition[] + FJSSPConfig that plug directly into the
    /// existing SimulationBridge pipeline.
    ///
    /// Machine mapping: Brandimarte uses generic numbered machines (0..M-1).
    /// This loader assigns them round-robin across MachineType values:
    ///   BM 0 → Mill,  BM 1 → Lathe,  BM 2 → Weld,  BM 3 → Inspect,
    ///   BM 4 → Assemble,  BM 5 → Mill (2nd),  BM 6 → Lathe (2nd), ...
    ///
    /// Stochastic calibration:
    ///   WeibullLambda is derived from instance processing time statistics so
    ///   that mean TTF is always a safe multiple of mean operation cycle time.
    ///   "Cycle time" = mean processing time + AGV travel overhead (estimated).
    ///   This prevents the infinite-loop failure mode where TTF < processing time.
    ///
    ///   Disruption regimes:
    ///     None  — deterministic, no StochasticConfig attached
    ///     Low   — mean TTF ≈ LOW_TTF_FACTOR  × mean_cycle_time
    ///     High  — mean TTF ≈ HIGH_TTF_FACTOR × mean_cycle_time
    ///             (clamped to always exceed max_proc_time + travel overhead)
    /// </summary>
    public static class BrandimartLoader
    {
        private static readonly MachineType[] AllTypes =
            (MachineType[])Enum.GetValues(typeof(MachineType));

        // ── Stochastic calibration constants ─────────────────────────────────

        /// @brief Estimated AGV round-trip travel overhead per operation (sim-seconds).
        /// Pickup transit + dropoff transit. Calibrated from observed ~30s each way.
        private const float AGV_TRAVEL_OVERHEAD = 60f;

        /// @brief k=1.5 Weibull: mean TTF = lambda × Γ(1 + 1/k) ≈ lambda × 0.9027
        private const float WEIBULL_MEAN_FACTOR = 0.9027f;

        /// @brief Low disruption: mean TTF = this multiple of mean cycle time.
        private const float LOW_TTF_FACTOR = 8f;

        /// @brief High disruption: mean TTF = this multiple of mean cycle time.
        /// Clamped so mean TTF > max_cycle_time (prevents infinite restart loops).
        private const float HIGH_TTF_FACTOR = 3f;

        /// @brief Repair log-normal parameters. Real-space mean = exp(mu + sigma²/2).
        /// These are kept proportional to mean processing time so repairs are
        /// disruptive but not job-blocking.
        /// RepairMu_low  → mean repair ≈ 0.15 × mean_proc_time
        /// RepairMu_high → mean repair ≈ 0.25 × mean_proc_time
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
        /// Returns a config (for SpawnFactory) and a deferred job builder.
        /// The config will have Stochastic = null (deterministic) by default.
        /// Call LoadDeferredWithStochastic for stochastic variants.
        /// </summary>
        public static (FJSSPConfig config,
                        Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs)
            LoadDeferred(string jsonPath, int seed = 42)
            => LoadDeferredInternal(jsonPath, seed, StochasticDisruption.None);

        /// <summary>
        /// Returns a config with a calibrated StochasticConfig attached.
        /// WeibullLambda and repair parameters are derived from the instance's
        /// actual processing time distribution so that failures are realistic
        /// but never produce infinite operation-restart loops.
        /// </summary>
        public static (FJSSPConfig config,
                        Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs)
            LoadDeferredWithStochastic(string jsonPath, StochasticDisruption disruption,
                                       int seed = 42)
            => LoadDeferredInternal(jsonPath, seed, disruption);

        // ─────────────────────────────────────────────────────────────────────
        //  Internal loader
        // ─────────────────────────────────────────────────────────────────────

        private static (FJSSPConfig config,
                         Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs)
            LoadDeferredInternal(string jsonPath, int seed, StochasticDisruption disruption)
        {
            if (!File.Exists(jsonPath))
            {
                SimLogger.LogError($"[BrandimartLoader] File not found: {jsonPath}");
                return (null, null);
            }

            string json = File.ReadAllText(jsonPath);
            string name = Path.GetFileNameWithoutExtension(jsonPath);

            FJSSPConfig config = BuildConfig(json, name, seed, disruption);
            if (config == null) return (null, null);

            return (config, machinesByType => BuildJobs(json, machinesByType));
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Config builder
        // ─────────────────────────────────────────────────────────────────────

        private static FJSSPConfig BuildConfig(string json, string name, int seed,
                                                StochasticDisruption disruption)
        {
            JObject root = JObject.Parse(json);
            int numMachines = root["machines"].Value<int>();
            JArray jobsArray = (JArray)root["jobs"];

            int numTypes = AllTypes.Length;
            int machinesPerType = Mathf.CeilToInt((float)numMachines / numTypes);
            int agvCount = Mathf.CeilToInt(machinesPerType * 1.5f);

            var typeLayout = new MachineType[numMachines];
            for (int bm = 0; bm < numMachines; bm++)
                typeLayout[bm] = AllTypes[bm % numTypes];
            int totalOps = 0;
            // ── Scan processing times ─────────────────────────────────────────
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

            // ── Derive stochastic parameters ──────────────────────────────────
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

        // ─────────────────────────────────────────────────────────────────────
        //  Stochastic parameter derivation
        // ─────────────────────────────────────────────────────────────────────

        /// <summary>
        /// Derives calibrated stochastic parameters from instance processing statistics.
        ///
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
        /// </summary>
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

        // ─────────────────────────────────────────────────────────────────────
        //  Job builder (unchanged)
        // ─────────────────────────────────────────────────────────────────────

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