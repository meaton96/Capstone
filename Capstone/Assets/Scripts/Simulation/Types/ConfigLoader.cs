using System;
using System.IO;
using System.Linq;
using UnityEngine;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Logging;

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
        // ─────────────────────────────────────────────────────────
        //  Public API
        // ─────────────────────────────────────────────────────────

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

        /// @brief Loads an array of configs from a batch JSON file.
        /// @param path  Path to a JSON file containing a "configs" array.
        /// @returns     Array of parsed FJSSPConfig objects, or empty on failure.
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

        /// @brief Parses a single config from a JSON string.
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

        /// @brief Parses a batch of configs from a JSON string.
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

        // ─────────────────────────────────────────────────────────
        //  JSON Data Classes  (JsonUtility-compatible)
        // ─────────────────────────────────────────────────────────

        [Serializable]
        private class JsonBatchWrapper
        {
            public JsonConfig[] configs;
        }

        [Serializable]
        private class JsonConfig
        {
            public string name = "";
            public int seed = 42;
            public int jobCount = 20;
            public int machinesPerType = 3;
            public string[] machineTypes;     // e.g. ["Mill","Lathe","Weld","Inspect","Assemble"]
            public float minProcTime = 15f;
            public float maxProcTime = 90f;
            public int minOpsPerJob = 3;
            public int maxOpsPerJob = 7;
            public float maxArrivalTime = 0f;
        }

        // ─────────────────────────────────────────────────────────
        //  Conversion
        // ─────────────────────────────────────────────────────────

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

            return new FJSSPConfig
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
                MaxArrivalTime = raw.maxArrivalTime,
            };
        }

        private static MachineType? ParseMachineType(string name)
        {
            if (Enum.TryParse<MachineType>(name, ignoreCase: true, out var result))
                return result;

            SimLogger.LogWarning($"[ConfigLoader] Unknown machine type '{name}', skipping.");
            return null;
        }
    }
}