using System;
using System.Collections.Generic;
using System.IO;
using Unity.MLAgents.SideChannels;
using Assets.Scripts.Simulation.Logging;
using Assets.Scripts.Simulation.Types;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Assets.Scripts.Simulation.Machines;

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
    ///
    /// A message with "scenarios" instead carries scripted ScenarioLoader scenarios to queue
    /// (one item consumed per StartEpisode, same buffered-queue pattern as EpisodeSeedChannel —
    /// Python stays ahead of episodes that auto-restart inside Unity). Each item is either a
    /// JSON object (an inline scenario) or a string (a path to a scenario file). "clear" empties
    /// the queue first. When the queue runs dry, FactoryOrchestrator keeps replaying the last
    /// scenario it consumed, so a single one-shot item behaves like the old sticky single-slot API.
    /// </summary>
    public class EpisodeConfigChannel : SideChannel
    {
        public static readonly Guid ChannelGuid =
            new Guid("b1e2c3d4-f5a6-7890-bcde-f01234567891");

        public static EpisodeConfigChannel Instance { get; private set; }

        /// <summary>A scripted scenario received from Python, parsed at the next episode start.</summary>
        public class PendingScenario
        {
            public string Name;
            public string Json;
        }

        private FJSSPConfig _pendingConfig = null;
        private readonly Queue<PendingScenario> _scenarioQueue = new Queue<PendingScenario>();
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
                JObject root = JObject.Parse(json);
                if (root["scenarios"] is JArray scenarios)
                {
                    ReadScenarioQueue(scenarios, root["clear"]?.Value<bool>() ?? false);
                    return;
                }

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

        /// <summary>
        /// Pops the next queued scenario, or null if the queue is empty (FactoryOrchestrator then
        /// keeps replaying whichever scenario it last consumed).
        /// </summary>
        public PendingScenario ConsumeScenario()
        {
            lock (_lock)
            {
                return _scenarioQueue.Count > 0 ? _scenarioQueue.Dequeue() : null;
            }
        }

        private void ReadScenarioQueue(JArray items, bool clear)
        {
            var incoming = new List<PendingScenario>();
            foreach (JToken item in items)
            {
                try
                {
                    incoming.Add(item.Type == JTokenType.String
                        ? ReadScenarioFromPath((string)item)
                        : ReadScenarioInline((JObject)item));
                }
                catch (Exception ex)
                {
                    SimLogger.LogError($"[ConfigChannel] Skipping one queued scenario: {ex.Message}");
                }
            }

            lock (_lock)
            {
                if (clear) _scenarioQueue.Clear();
                foreach (PendingScenario scenario in incoming)
                    _scenarioQueue.Enqueue(scenario);
            }
            SimLogger.Low($"[ConfigChannel] Queued {incoming.Count} scenario(s) (clear={clear}).");
        }

        private static PendingScenario ReadScenarioInline(JObject scenario) => new PendingScenario
        {
            Name = scenario["name"]?.Value<string>() ?? "python_scenario",
            Json = scenario.ToString(Formatting.None),
        };

        private static PendingScenario ReadScenarioFromPath(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"Scenario file not found: {path}");
            return new PendingScenario
            {
                Name = Path.GetFileNameWithoutExtension(path),
                Json = File.ReadAllText(path),
            };
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
                AGVCount = root["agvCount"]?.Value<int>() ?? 5,
                AGVMoveSpeed = root["agvMoveSpeed"]?.Value<float>(),
                AGVHandshakeDuration = root["agvHandshakeDuration"]?.Value<float>(),
                MachineFlexibilityProbability = root["machineFlexibilityProbability"]?.Value<float>() ?? 0f,
                parkingMethod = root["parkingMethod"]?.Value<string>() ?? "single",
                reservationProtocol = ReservationProtocolParser.Validated(root["reservationProtocol"]?.Value<string>()),
                preDispatchingMethod = root["preDispatchingMethod"]?.Value<string>() ?? "fixed",
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