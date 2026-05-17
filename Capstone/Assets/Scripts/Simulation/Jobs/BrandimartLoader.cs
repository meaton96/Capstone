using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Newtonsoft.Json.Linq;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Logging;

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
    /// Requires Newtonsoft.Json (com.unity.nuget.newtonsoft-json) because
    /// Unity's JsonUtility cannot deserialize nested generic lists.
    ///
    /// Usage from HeadlessBatchRunner:
    ///   var (config, buildJobs) = BrandimartLoader.LoadDeferred(path);
    ///   bridge.LoadConfig(config);
    ///   bridge.SpawnFactory();                       // creates machines, assigns IDs
    ///   var jobs = buildJobs(bridge.CachedMachinesByType);
    ///   bridge.LoadPrebuiltJobs(jobs);
    ///   // ... then start episode as normal
    /// </summary>
    public static class BrandimartLoader
    {
        private static readonly MachineType[] AllTypes =
            (MachineType[])Enum.GetValues(typeof(MachineType));

        // ─────────────────────────────────────────────────────────
        //  Public: one-shot convenience
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Returns a config (for SpawnFactory) and a deferred job builder
        /// (call after spawn with the runtime machinesByType dictionary).
        /// </summary>
        public static (FJSSPConfig config,
                        Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs)
            LoadDeferred(string jsonPath, int seed = 42)
        {
            if (!File.Exists(jsonPath))
            {
                SimLogger.LogError($"[BrandimartLoader] File not found: {jsonPath}");
                return (null, null);
            }

            string json = File.ReadAllText(jsonPath);
            string name = Path.GetFileNameWithoutExtension(jsonPath);

            FJSSPConfig config = BuildConfig(json, name, seed);
            if (config == null) return (null, null);

            // Capture json in closure — jobs are built later once IDs exist
            return (config, machinesByType => BuildJobs(json, machinesByType));
        }

        // ─────────────────────────────────────────────────────────
        //  Config builder (call BEFORE SpawnFactory)
        // ─────────────────────────────────────────────────────────

        private static FJSSPConfig BuildConfig(string json, string name, int seed)
        {
            JObject root = JObject.Parse(json);
            int numMachines = root["machines"].Value<int>();
            JArray jobsArray = (JArray)root["jobs"];

            int numTypes = AllTypes.Length;
            int machinesPerType = Mathf.CeilToInt((float)numMachines / numTypes);
            int agvCount = Mathf.CeilToInt(machinesPerType * 1.5f);

            // Build MachineTypeLayout with one entry per BM machine using
            // the same round-robin assignment that BuildJobs uses.
            // FactoryLayoutManager reads MachineTypeLayout.Length as the
            // total machine count, so this array must have exactly
            // numMachines entries — not just one per type.
            var typeLayout = new MachineType[numMachines];
            for (int bm = 0; bm < numMachines; bm++)
                typeLayout[bm] = AllTypes[bm % numTypes];

            // Scan job data for ops/proc-time ranges (informational for config)
            int minOps = int.MaxValue, maxOps = 0;
            float minProc = float.MaxValue, maxProc = 0f;

            foreach (JArray job in jobsArray)
            {
                minOps = Mathf.Min(minOps, job.Count);
                maxOps = Mathf.Max(maxOps, job.Count);

                foreach (JArray op in job)
                    foreach (JObject opt in op)
                    {
                        float p = opt["processing"].Value<float>();
                        minProc = Mathf.Min(minProc, p);
                        maxProc = Mathf.Max(maxProc, p);
                    }
            }

            return new FJSSPConfig
            {
                Name = $"brandimarte_{name}",
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
                MachineFlexibilityProbability = 0f,
            };
        }

        // ─────────────────────────────────────────────────────────
        //  Job builder (call AFTER SpawnFactory)
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Parses the benchmark JSON and maps BM machine indices to runtime
        /// machine IDs using the machinesByType dictionary from the layout
        /// manager. Returns ready-to-use FJSSPJobDefinition[].
        /// </summary>
        public static FJSSPJobDefinition[] BuildJobs(
            string json,
            Dictionary<MachineType, List<int>> machinesByType)
        {
            JObject root = JObject.Parse(json);
            int numMachines = root["machines"].Value<int>();
            JArray jobsArray = (JArray)root["jobs"];

            int numTypes = AllTypes.Length;

            // ── Round-robin mapping: BM index → (MachineType, indexWithinType) ──
            var bmMap = new (MachineType type, int idxInType)[numMachines];
            var countPerType = new int[numTypes];

            for (int bm = 0; bm < numMachines; bm++)
            {
                MachineType t = AllTypes[bm % numTypes];
                int typeEnum = (int)t;
                bmMap[bm] = (t, countPerType[typeEnum]);
                countPerType[typeEnum]++;
            }

            // ── Convert each BM job ─────────────────────────────
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
                        float pTime = opt["processing"].Value<float>();

                        var (type, idxInType) = bmMap[bmIdx];

                        // First eligible machine's type becomes the operation's
                        // canonical type for OperationSequence. The PDR machine-
                        // selection heuristic uses EligibleMachinesPerOp (all
                        // eligible machines) regardless of this field.
                        if (!firstSet)
                        {
                            opSequence[o] = type;
                            firstSet = true;
                        }

                        // Map to runtime ID
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
}