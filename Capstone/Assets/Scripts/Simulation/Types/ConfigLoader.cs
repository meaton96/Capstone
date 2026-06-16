using System;
using System.IO;
using System.Linq;
using UnityEngine;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Logging;
using System.Collections.Generic;

namespace Assets.Scripts.Simulation.Types
{
    /// @brief Loads FJSSPConfig instances from JSON files on disk.
    ///
    /// @details Supports single-config files and batch arrays for headless runs.
    ///          Machine types are specified by name strings in the JSON and mapped
    ///          to MachineType enum values at load time.
    ///
    /// JSON format (single config):
    /// @code
    /// {
    ///     "seed": 42,
    ///     "jobCount": 20,
    ///     "machinesPerType": 3,
    ///     "machineTypes": ["Mill","Lathe","Weld","Inspect","Assemble"],
    ///     "minProcTime": 15.0,
    ///     "maxProcTime": 90.0,
    ///     "minOpsPerJob": 3,
    ///     "maxOpsPerJob": 7,
    ///     "maxArrivalTime": 0.0,
    ///     "name": "baseline_20j_15m"
    /// }
    /// @endcode
    ///
    /// Batch format (array wrapper):
    /// @code
    /// { "configs": [ { ... }, { ... } ] }
    /// @endcode
    public static class ConfigLoader
    {
        // ── Public API ──

        /// @brief Loads a single FJSSPConfig from a JSON file path.
        /// @param path  Absolute or relative path to the .json file.
        /// @returns     Parsed FJSSPConfig, or null on failure.
        public static FJSSPConfig LoadSingle(string path)
        {
            if (!File.Exists(path))
            {
                SimLogger.LogError($"[ConfigLoader] File not found: {path}");
                return null;
            }

            string json = File.ReadAllText(path);
            return ParseSingle(json);
        }

        /// @brief Loads an array of FJSSPConfig instances from a batch JSON file.
        /// @details Falls back to single-config parsing if the "configs" array is empty.
        /// @param path  Path to a JSON file containing a "configs" array.
        /// @returns     Array of parsed FJSSPConfig objects, or empty array on failure.
        public static FJSSPConfig[] LoadBatch(string path)
        {
            if (!File.Exists(path))
            {
                SimLogger.LogError($"[ConfigLoader] File not found: {path}");
                return Array.Empty<FJSSPConfig>();
            }

            string json = File.ReadAllText(path);
            return ParseBatch(json);
        }

        /// @brief Parses a single FJSSPConfig from a JSON string.
        /// @param json  JSON-formatted configuration string.
        /// @returns     Parsed FJSSPConfig, or null on failure.
        public static FJSSPConfig ParseSingle(string json)
        {
            try
            {
                JsonConfig raw = JsonUtility.FromJson<JsonConfig>(json);
                return Convert(raw);
            }
            catch (Exception ex)
            {
                SimLogger.LogError($"[ConfigLoader] Parse error: {ex.Message}");
                return null;
            }
        }

        /// @brief Parses a batch of FJSSPConfig instances from a JSON string.
        /// @details Expects a "configs" array; falls back to single-config parsing if absent.
        /// @param json  JSON-formatted string containing a "configs" array or single config object.
        /// @returns     Array of parsed FJSSPConfig objects, or empty array on failure.
        public static FJSSPConfig[] ParseBatch(string json)
        {
            try
            {
                JsonBatchWrapper wrapper = JsonUtility.FromJson<JsonBatchWrapper>(json);
                if (wrapper?.configs == null || wrapper.configs.Length == 0)
                {
                    // Try single-config fallback
                    var single = ParseSingle(json);
                    return single != null ? new[] { single } : Array.Empty<FJSSPConfig>();
                }
                return wrapper.configs.Select(Convert).Where(c => c != null).ToArray();
            }
            catch (Exception ex)
            {
                SimLogger.LogError($"[ConfigLoader] Batch parse error: {ex.Message}");
                return Array.Empty<FJSSPConfig>();
            }
        }

        // ── Internal JSON data classes (JsonUtility-compatible) ──

        /// @brief Wrapper for batch config JSON with a "configs" array field.
        [Serializable]
        private class JsonBatchWrapper
        {
            public JsonConfig[] configs;
        }

        /// @brief Serializable representation of a single FJSSPConfig for JSON deserialization.
        /// @details Field names and defaults mirror FJSSPConfig to enable direct JsonUtility mapping.
        [Serializable]
        private class JsonConfig
        {
            public string name = "";
            public int seed = 42;
            public int jobCount = 20;
            public int machinesPerType = 3;
            public string[] machineTypes;
            public float minProcTime = 15f;
            public float maxProcTime = 90f;
            public int minOpsPerJob = 3;
            public int maxOpsPerJob = 7;
            public float maxArrivalTime = 0f;
            public int agvCount = 3;
            public float machineFlexibilityProbability = 0f;
            public string parkingMethod = "single";
            public string preDispatchingMethod = "fixed";
            public JsonStochasticConfig stochastic = null;

            /// @brief Optional per-type normal distribution parameters for processing time sampling.
            /// @details If provided, overrides the uniform minProcTime/maxProcTime fallback for
            ///          each named type. Types not listed here fall back to the uniform derivation.
            /// JSON example:
            /// @code
            /// "procTimeParams": [
            ///   { "machineType": "Mill",     "mu":  90.0, "sigma": 10.0 },
            ///   { "machineType": "Lathe",    "mu":  75.0, "sigma": 10.0 },
            ///   { "machineType": "Weld",     "mu": 150.0, "sigma": 25.0 },
            ///   { "machineType": "Inspect",  "mu":  60.0, "sigma": 10.0 },
            ///   { "machineType": "Assemble", "mu": 240.0, "sigma": 40.0 }
            /// ]
            /// @endcode
            public JsonProcTimeParam[] procTimeParams = null;
        }

        /// @brief Per-machine-type normal distribution override for JSON deserialization.
        [Serializable]
        private class JsonProcTimeParam
        {
            public string machineType = "";
            public float mu = 52.5f;    // matches midpoint of default minProcTime=15/maxProcTime=90
            public float sigma = 12.5f; // matches (90-15)/6
        }
        [Serializable]
        private class JsonStochasticConfig
        {
            public bool machineFailuresEnabled = false;
            public float weibullK = 1.5f;
            public float weibullLambda = 2700f;
            public float repairLogMu = 4.0f;
            public float repairLogSigma = 0.5f;
            public bool agvFailuresEnabled = false;
            public float agvWeibullLambda = 700f;
            public float agvRepairLogMu = 3.4f;
            public float agvRepairLogSigma = 0.4f;
            public bool dynamicArrivalsEnabled = false;
            public float arrivalLambda = 0.003f;
            public int dynamicJobCap = 0;
        }

        // ── Conversion ──

        /// @brief Converts a JsonConfig to a runtime FJSSPConfig.
        /// @details Maps string-based machine type names to MachineType enum values and expands
        ///          the layout by repeating each type machinesPerType times.
        /// @param raw  Parsed JSON config data.
        /// @returns     Runtime FJSSPConfig, or null if raw is null.
        private static FJSSPConfig Convert(JsonConfig raw)
        {
            if (raw == null) return null;

            // Build MachineTypeLayout: machinesPerType copies of each named type
            MachineType[] baseTypes;
            if (raw.machineTypes != null && raw.machineTypes.Length > 0)
            {
                baseTypes = raw.machineTypes
                    .Select(ParseMachineType)
                    .Where(t => t.HasValue)
                    .Select(t => t.Value)
                    .ToArray();
            }
            else
            {
                // Default: all enum values
                baseTypes = (MachineType[])Enum.GetValues(typeof(MachineType));
            }

            // Expand: machinesPerType copies of each base type, grouped
            MachineType[] layout = new MachineType[baseTypes.Length * raw.machinesPerType];
            for (int t = 0; t < baseTypes.Length; t++)
                for (int m = 0; m < raw.machinesPerType; m++)
                    layout[t * raw.machinesPerType + m] = baseTypes[t];

            float mu = (raw.minProcTime + raw.maxProcTime) * 0.5f;
            float sigma = (raw.maxProcTime - raw.minProcTime) / 6f;
            var procTimeParams = new Dictionary<MachineType, (float mu, float sigma)>();

            // Build a lookup from any explicit per-type overrides in the JSON
            var explicitParams = new Dictionary<MachineType, (float mu, float sigma)>();
            if (raw.procTimeParams != null)
            {
                foreach (var p in raw.procTimeParams)
                {
                    var parsed = ParseMachineType(p.machineType);
                    if (parsed.HasValue)
                        explicitParams[parsed.Value] = (p.mu, p.sigma);
                }
            }

            // Per type: use explicit override if present, else fall back to uniform derivation
            foreach (MachineType t in baseTypes)
                procTimeParams[t] = explicitParams.TryGetValue(t, out var ep) ? ep : (mu, sigma);

            FJSSPConfig cfg = new FJSSPConfig
            {
                Name = string.IsNullOrEmpty(raw.name) ? "unnamed" : raw.name,
                Seed = raw.seed,
                JobCount = raw.jobCount,
                MachinesPerType = raw.machinesPerType,
                MachineTypeLayout = layout,
                MinProcTime = raw.minProcTime,
                MaxProcTime = raw.maxProcTime,
                MinOpsPerJob = raw.minOpsPerJob,
                MaxOpsPerJob = raw.maxOpsPerJob,
                AGVCount = raw.agvCount,
                MachineFlexibilityProbability = raw.machineFlexibilityProbability,
                ProcTimeParams = procTimeParams,
                parkingMethod = raw.parkingMethod,
                preDispatchingMethod = raw.preDispatchingMethod,
            };

            if (raw.stochastic != null)
            {
                cfg.Stochastic = new StochasticConfig
                {
                    MachineFailuresEnabled = raw.stochastic.machineFailuresEnabled,
                    WeibullK = raw.stochastic.weibullK,
                    WeibullLambda = raw.stochastic.weibullLambda,
                    RepairLogMu = raw.stochastic.repairLogMu,
                    RepairLogSigma = raw.stochastic.repairLogSigma,
                    AGVFailuresEnabled = raw.stochastic.agvFailuresEnabled,
                    AGVWeibullLambda = raw.stochastic.agvWeibullLambda,
                    AGVRepairLogMu = raw.stochastic.agvRepairLogMu,
                    AGVRepairLogSigma = raw.stochastic.agvRepairLogSigma,
                    DynamicArrivalsEnabled = raw.stochastic.dynamicArrivalsEnabled,
                    ArrivalLambda = raw.stochastic.arrivalLambda,
                    DynamicJobCap = raw.stochastic.dynamicJobCap,
                };
            }

            return cfg;
        }

        /// @brief Parses a machine type name string to its corresponding MachineType enum value.
        /// @param name  Case-insensitive machine type name from JSON (e.g. "Mill", "Lathe").
        /// @returns     Parsed MachineType enum value, or null if unrecognised (with warning).
        private static MachineType? ParseMachineType(string name)
        {
            if (Enum.TryParse<MachineType>(name, ignoreCase: true, out var result))
                return result;

            SimLogger.LogWarning($"[ConfigLoader] Unknown machine type '{name}', skipping.");
            return null;
        }
    }
}