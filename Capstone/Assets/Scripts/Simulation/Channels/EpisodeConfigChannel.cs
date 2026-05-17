using System;
using Unity.MLAgents.SideChannels;
using Assets.Scripts.Logging;
using Assets.Scripts.Simulation.Types;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Assets.Scripts.Simulation.Machines;
using System.Collections.Generic;

namespace Assets.Scripts.Simulation.Channels
{
    /// <summary>
    /// Receives a full FJSSPConfig JSON blob from Python each episode reset.
    /// Python sends this via the matching EpisodeConfigChannel on the Python side.
    ///
    /// Channel GUID must match exactly between C# and Python.
    ///
    /// Usage in SimulationBridge.OnEpisodeBegin():
    ///   var cfg = EpisodeConfigChannel.Instance.ConsumeConfig();
    ///   if (cfg != null) ApplyConfig(cfg);  // override current config
    ///   else             UseDefaultConfig(); // no Python override this episode
    /// </summary>
    public class EpisodeConfigChannel : SideChannel
    {
        public static readonly Guid ChannelGuid =
            new Guid("b1e2c3d4-f5a6-7890-bcde-f01234567891");

        public static EpisodeConfigChannel Instance { get; private set; }

        private FJSSPConfig _pendingConfig = null;
        private readonly object _lock = new object();

        public EpisodeConfigChannel()
        {
            Instance = this;
            ChannelId = ChannelGuid;
        }

        /// <summary>
        /// Called by ML-Agents when Python sends a message on this channel.
        /// Deserialises the JSON payload into a FJSSPConfig and holds it
        /// until SimulationBridge calls ConsumeConfig().
        /// </summary>
        protected override void OnMessageReceived(IncomingMessage msg)
        {
            string json = msg.ReadString();
            try
            {
                FJSSPConfig cfg = DeserialiseConfig(json);
                lock (_lock) { _pendingConfig = cfg; }
                SimLogger.Low($"[ConfigChannel] Received config: {cfg.Name} " +
                              $"jobs={cfg.JobCount} machines={cfg.TotalMachines} " +
                              $"agvs={cfg.AGVCount} " +
                              $"stochastic={cfg.Stochastic?.Tag ?? "none"}");
            }
            catch (Exception ex)
            {
                SimLogger.LogError($"[ConfigChannel] Failed to parse config JSON: {ex.Message}");
            }
        }

        /// <summary>
        /// Returns and clears the pending config. Returns null if Python
        /// has not sent a config for this episode (use default/previous).
        /// </summary>
        public FJSSPConfig ConsumeConfig()
        {
            lock (_lock)
            {
                var cfg = _pendingConfig;
                _pendingConfig = null;
                return cfg;
            }
        }

        // ── JSON deserialisation ─────────────────────────────────────────────

        private static FJSSPConfig DeserialiseConfig(string json)
        {
            JObject root = JObject.Parse(json);

            var cfg = new FJSSPConfig
            {
                Name = root["name"]?.Value<string>() ?? "python_config",
                Seed = root["seed"]?.Value<int>() ?? 42,
                JobCount = root["jobCount"].Value<int>(),
                MachinesPerType = root["machinesPerType"].Value<int>(),
                MinProcTime = root["minProcTime"]?.Value<float>() ?? 15f,
                MaxProcTime = root["maxProcTime"]?.Value<float>() ?? 60f,
                MinOpsPerJob = root["minOpsPerJob"]?.Value<int>() ?? 3,
                MaxOpsPerJob = root["maxOpsPerJob"]?.Value<int>() ?? 6,
                MaxArrivalTime = root["maxArrivalTime"]?.Value<float>() ?? 0f,
                AGVCount = root["agvCount"]?.Value<int>() ?? 5,
                MachineFlexibilityProbability = root["machineFlexibilityProbability"]?.Value<float>() ?? 0f,

            };

            // MachineTypeLayout from machineTypes string array
            if (root["machineTypes"] is JArray typeArray)
            {
                var layout = new MachineType[typeArray.Count * cfg.MachinesPerType];
                for (int i = 0; i < layout.Length; i++)
                {
                    string typeName = typeArray[i % typeArray.Count].Value<string>();
                    layout[i] = Enum.Parse<MachineType>(typeName);
                }
                cfg.MachineTypeLayout = layout;
            }

            // Optional stochastic block
            if (root["stochastic"] is JObject s)
            {
                cfg.Stochastic = new StochasticConfig
                {
                    MachineFailuresEnabled = s["machineFailuresEnabled"]?.Value<bool>() ?? false,
                    WeibullK = s["weibullK"]?.Value<float>() ?? 1.5f,
                    WeibullLambda = s["weibullLambda"]?.Value<float>() ?? 900f,
                    RepairLogMu = s["repairLogMu"]?.Value<float>() ?? 4.0f,
                    RepairLogSigma = s["repairLogSigma"]?.Value<float>() ?? 0.5f,
                    AGVFailuresEnabled = s["agvFailuresEnabled"]?.Value<bool>() ?? false,
                    AGVWeibullLambda = s["agvWeibullLambda"]?.Value<float>() ?? 700f,
                    AGVRepairLogMu = s["agvRepairLogMu"]?.Value<float>() ?? 3.4f,
                    AGVRepairLogSigma = s["agvRepairLogSigma"]?.Value<float>() ?? 0.4f,
                    DynamicArrivalsEnabled = s["dynamicArrivalsEnabled"]?.Value<bool>() ?? false,
                    ArrivalLambda = s["arrivalLambda"]?.Value<float>() ?? 0.005f,
                };
            }

            // Optional per-type proc time params
            if (root["procTimeParams"] is JObject ptp)
            {
                foreach (var kvp in ptp)
                {
                    if (Enum.TryParse<MachineType>(kvp.Key, out var mt) &&
                        kvp.Value is JObject p)
                    {
                        cfg.ProcTimeParams[mt] = (
                            p["mu"].Value<float>(),
                            p["sigma"].Value<float>()
                        );
                    }
                }
            }

            return cfg;
        }
    }
}