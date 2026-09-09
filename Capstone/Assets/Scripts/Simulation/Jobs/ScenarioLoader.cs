using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Newtonsoft.Json.Linq;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Logging;

namespace Assets.Scripts.Simulation.Jobs
{
    /// <summary>
    /// Loads hand-crafted scenario instances from JSON — fully explicit job sets (arrival
    /// times, operation sequences, machine assignments, durations) instead of the i.i.d.
    /// random generation FJSSPJobGenerator produces. Structural scenarios (a forced
    /// bottleneck machine, two jobs pinned to compete for the same machine, an identical-job
    /// batch, etc.) need contention placed deliberately — i.i.d. random jobs average that
    /// structure out across enough machines that dispatching/routing rule choice rarely
    /// matters, which is why Brandimarte-style structured instances differentiate rules and
    /// generated ones mostly don't.
    /// </summary>
    /// <remarks>
    /// Mirrors BrandimartLoader's two-phase deferred-loading contract exactly: BuildConfig
    /// runs immediately (JSON → FJSSPConfig, including the floor's MachineTypeLayout so
    /// BuildFloor can lay it out), BuildJobs is deferred until after SpawnFactory() has run
    /// so it has the real runtime machine IDs (machinesByType) needed to resolve each op's
    /// "machineIndex" into an actual machine.
    ///
    /// JSON schema:
    /// {
    ///   "name": "single_bottleneck_demo",
    ///   "seed": 42,
    ///   "agvCount": 7,
    ///   "machineTypeLayout": ["Mill","Mill","Mill","Lathe","Lathe","Lathe","Weld","Weld","Weld"],
    ///   "jobs": [
    ///     {
    ///       "id": 0,
    ///       "arrivalTime": 0.0,
    ///       "operations": [
    ///         { "machineType": "Mill", "machineIndex": "any", "duration": 40.0 },
    ///         { "machineType": "Weld", "machineIndex": 1,     "duration": 60.0 }
    ///       ]
    ///     }
    ///   ]
    /// }
    ///
    /// "machineIndex" is one of three forms: the string "any" (eligible = every runtime
    /// machine currently registered under that type — the same all-of-type eligibility
    /// FJSSPJobGenerator uses); a single 0-based index within that type, pinning the op to
    /// exactly one specific runtime machine (machinesByType[type][index]) — forces jobs onto
    /// the same machine with no routing choice left; or an array of indices (e.g. [0, 1]),
    /// eligible = exactly that subset — forces routing contention onto a small, specific set
    /// of candidates instead of diluting across every instance of the type.
    /// "duration" is normally a single scalar applied to every eligible machine on that op.
    /// When "machineIndex" is an array, "duration" may instead be an array of the same
    /// length, pairing durations[k] with machineIndex[k] — machines can then have genuinely
    /// different costs for a given job. Without this, every candidate is tied on cost, so
    /// SMPT's ArgMinIdx tie-break deterministically always picks the same machine (a
    /// total-collapse failure mode, not graded imbalance) — the array form lets a scenario
    /// test whether a routing rule picks the genuinely cheaper option instead.
    /// </remarks>
    public static class ScenarioLoader
    {
        // ─────────────────────────────────────────────────────────────────────
        //  Public: one-shot convenience
        // ─────────────────────────────────────────────────────────────────────

        /// <summary>
        /// Loads a hand-crafted scenario instance and returns a deferred configuration
        /// and job builder, matching BrandimartLoader's contract so HeadlessBatchRunner's
        /// existing RunBenchmarkEpisodes coroutine can drive either loader unchanged.
        /// </summary>
        /// <param name="jsonPath">Path to the scenario JSON file.</param>
        /// <param name="seedOverride">Overrides the JSON's "seed" field if &gt;= 0.</param>
        /// <param name="agvCountOverride">Overrides the JSON's "agvCount" field if &gt; 0.</param>
        public static (FJSSPConfig config,
                        Func<Dictionary<MachineType, List<int>>, FJSSPJobDefinition[]> buildJobs)
            LoadDeferred(string jsonPath, int seedOverride = -1, int agvCountOverride = -1)
        {
            if (!File.Exists(jsonPath))
            {
                SimLogger.LogError($"[ScenarioLoader] File not found: {jsonPath}");
                return (null, null);
            }

            string json = File.ReadAllText(jsonPath);
            string fileName = Path.GetFileNameWithoutExtension(jsonPath);

            FJSSPConfig config = BuildConfig(json, fileName, seedOverride, agvCountOverride);
            if (config == null) return (null, null);

            return (config, machinesByType => BuildJobs(json, machinesByType));
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Phase 1: config (runs before SpawnFactory, no runtime machine IDs yet)
        // ─────────────────────────────────────────────────────────────────────

        private static FJSSPConfig BuildConfig(string json, string fileName,
                                                 int seedOverride, int agvCountOverride)
        {
            JObject root = JObject.Parse(json);
            JArray jobsArray = (JArray)root["jobs"];
            if (jobsArray == null || jobsArray.Count == 0)
            {
                SimLogger.LogError($"[ScenarioLoader] '{fileName}': no jobs array, or it's empty.");
                return null;
            }

            JArray layoutArray = (JArray)root["machineTypeLayout"];
            if (layoutArray == null || layoutArray.Count == 0)
            {
                SimLogger.LogError($"[ScenarioLoader] '{fileName}': missing/empty machineTypeLayout — " +
                                    "hand-crafted scenarios must specify the floor explicitly so " +
                                    "machineIndex references (e.g. \"Weld\"[1]) are unambiguous.");
                return null;
            }

            var layout = new MachineType[layoutArray.Count];
            for (int i = 0; i < layoutArray.Count; i++)
            {
                if (!Enum.TryParse((string)layoutArray[i], ignoreCase: true, out MachineType t))
                {
                    SimLogger.LogError($"[ScenarioLoader] '{fileName}': unknown machine type " +
                                        $"'{layoutArray[i]}' at machineTypeLayout[{i}].");
                    return null;
                }
                layout[i] = t;
            }

            int numTypes = Enum.GetValues(typeof(MachineType)).Length;
            int seed = seedOverride >= 0 ? seedOverride : (root["seed"]?.Value<int>() ?? 42);
            int agvCount = agvCountOverride > 0
                ? agvCountOverride
                : (root["agvCount"]?.Value<int>() ?? Mathf.CeilToInt(layout.Length / (float)numTypes * 1.5f));

            // Scan durations/op-counts for metadata only — actual job structure comes entirely
            // from the explicit per-op durations at build time, not from these ranges.
            int minOps = int.MaxValue, maxOps = 0;
            float minDur = float.MaxValue, maxDur = 0f;
            foreach (JObject job in jobsArray)
            {
                JArray ops = (JArray)job["operations"];
                minOps = Mathf.Min(minOps, ops.Count);
                maxOps = Mathf.Max(maxOps, ops.Count);
                foreach (JObject op in ops)
                {
                    float d = op["duration"].Value<float>();
                    minDur = Mathf.Min(minDur, d);
                    maxDur = Mathf.Max(maxDur, d);
                }
            }

            return new FJSSPConfig
            {
                Name = (string)root["name"] ?? $"scenario_{fileName}",
                Seed = seed,
                JobCount = jobsArray.Count,
                MachinesPerType = Mathf.CeilToInt(layout.Length / (float)numTypes),
                MachineTypeLayout = layout,
                MinProcTime = minDur,
                MaxProcTime = maxDur,
                MinOpsPerJob = minOps,
                MaxOpsPerJob = maxOps,
                AGVCount = agvCount,
                Stochastic = null,   // deterministic — episode ends once every job exits
            };
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Phase 2: jobs (deferred until machinesByType exists, post-SpawnFactory)
        // ─────────────────────────────────────────────────────────────────────

        private static FJSSPJobDefinition[] BuildJobs(
            string json, Dictionary<MachineType, List<int>> machinesByType)
        {
            JObject root = JObject.Parse(json);
            JArray jobsArray = (JArray)root["jobs"];

            var jobs = new FJSSPJobDefinition[jobsArray.Count];
            int nextAutoId = 0;

            for (int j = 0; j < jobsArray.Count; j++)
            {
                JObject rawJob = (JObject)jobsArray[j];
                int jobId = rawJob["id"]?.Value<int>() ?? nextAutoId;
                nextAutoId = Mathf.Max(nextAutoId, jobId + 1);
                float arrivalTime = rawJob["arrivalTime"]?.Value<float>() ?? 0f;

                JArray rawOps = (JArray)rawJob["operations"];
                var opSequence = new MachineType[rawOps.Count];
                var eligible = new Dictionary<int, float>[rawOps.Count];

                for (int o = 0; o < rawOps.Count; o++)
                {
                    JObject rawOp = (JObject)rawOps[o];

                    if (!Enum.TryParse((string)rawOp["machineType"], ignoreCase: true, out MachineType type))
                    {
                        SimLogger.LogError($"[ScenarioLoader] Job {jobId} op {o}: unknown machine " +
                                            $"type '{rawOp["machineType"]}'.");
                        continue;
                    }
                    JToken durationToken = rawOp["duration"];
                    opSequence[o] = type;
                    eligible[o] = new Dictionary<int, float>();

                    if (!machinesByType.TryGetValue(type, out var idList) || idList.Count == 0)
                    {
                        SimLogger.LogError($"[ScenarioLoader] Job {jobId} op {o}: no runtime " +
                                            $"machines of type {type} exist on the floor.");
                        continue;
                    }

                    JToken idxToken = rawOp["machineIndex"];
                    bool isAny = idxToken == null
                        || (idxToken.Type == JTokenType.String
                            && string.Equals((string)idxToken, "any", StringComparison.OrdinalIgnoreCase));

                    if (isAny)
                    {
                        // All-of-type eligibility, same convention FJSSPJobGenerator uses —
                        // routing rule chooses among every instance of this type. "any" only
                        // takes a scalar duration (no natural per-machine order to pair against).
                        float duration = durationToken.Value<float>();
                        foreach (int machineId in idList)
                            eligible[o][machineId] = duration;
                    }
                    else if (idxToken.Type == JTokenType.Array)
                    {
                        // A chosen subset (e.g. [0, 1]) — eligibility is exactly those
                        // machines, so the routing rule (SMPT/SRWT/MMUR) actually has to
                        // choose between a small, specific set of contended candidates
                        // instead of diluting across every instance of the type ("any")
                        // or removing the choice entirely (a single pinned index).
                        //
                        // "duration" here is either a single scalar (every candidate costs
                        // the same for this job — SMPT's score is then genuinely tied, and
                        // its ArgMinIdx tie-break deterministically always picks the same
                        // machine, a total-collapse failure mode rather than graded
                        // imbalance) or an array the same length as machineIndex, pairing
                        // durations[k] with machineIndex[k] so machines can have genuinely
                        // different costs for this job — the graded case, where SMPT
                        // routing to the wrong one is an actual mistake, not a coin flip.
                        JArray indices = (JArray)idxToken;
                        JArray durations = durationToken.Type == JTokenType.Array ? (JArray)durationToken : null;
                        if (durations != null && durations.Count != indices.Count)
                        {
                            SimLogger.LogError($"[ScenarioLoader] Job {jobId} op {o}: duration " +
                                                $"array length ({durations.Count}) must match " +
                                                $"machineIndex array length ({indices.Count}).");
                            continue;
                        }
                        float uniformDuration = durations == null ? durationToken.Value<float>() : 0f;

                        for (int k = 0; k < indices.Count; k++)
                        {
                            int idx = indices[k].Value<int>();
                            if (idx < 0 || idx >= idList.Count)
                            {
                                SimLogger.LogError($"[ScenarioLoader] Job {jobId} op {o}: " +
                                                    $"machineIndex {idx} out of range for {type} " +
                                                    $"(only {idList.Count} on the floor).");
                                continue;
                            }
                            eligible[o][idList[idx]] = durations != null ? durations[k].Value<float>() : uniformDuration;
                        }
                    }
                    else
                    {
                        float duration = durationToken.Value<float>();
                        int idx = idxToken.Value<int>();
                        if (idx < 0 || idx >= idList.Count)
                        {
                            SimLogger.LogError($"[ScenarioLoader] Job {jobId} op {o}: " +
                                                $"machineIndex {idx} out of range for {type} " +
                                                $"(only {idList.Count} on the floor).");
                            continue;
                        }
                        // Single-entry eligibility — pins this op to exactly one machine,
                        // e.g. to force two jobs to compete for the same resource.
                        eligible[o][idList[idx]] = duration;
                    }
                }

                jobs[j] = new FJSSPJobDefinition
                {
                    JobId = jobId,
                    ArrivalTime = arrivalTime,
                    OperationSequence = opSequence,
                    EligibleMachinesPerOp = eligible,
                };
            }

            SimLogger.Low($"[ScenarioLoader] Loaded {jobs.Length} hand-crafted jobs.");
            return jobs;
        }
    }
}
