using System;
using System.Collections.Generic;
using Unity.MLAgents.SideChannels;
using Assets.Scripts.Logging;
using Newtonsoft.Json;
using UnityEngine;

namespace Assets.Scripts.Simulation.Channels
{
    /// <summary>
    /// Collects per-episode events (failures, repairs, arrivals, final result)
    /// and flushes them to Python as a JSON string at episode end.
    ///
    /// PhysicalMachine calls RecordFailure/RecordRepair directly.
    /// SimulationBridge calls RecordEpisodeResult then Flush at episode end.
    ///
    /// Python receives a single JSON blob per episode containing all events.
    /// This is used for:
    ///   (a) Stochastic distribution validation tests
    ///   (b) Curriculum training diagnostics
    ///   (c) Eventually: rich per-episode logging without touching ResultsLogger
    /// </summary>
    public class EpisodeTelemetryChannel : SideChannel
    {
        public static readonly Guid ChannelGuid =
            new Guid("c2f3d4e5-a6b7-8901-cdef-012345678902");

        public static EpisodeTelemetryChannel Instance { get; private set; }

        private readonly List<TelemetryEvent> _events = new List<TelemetryEvent>();
        private EpisodeResult _result;

        public EpisodeTelemetryChannel()
        {
            Instance = this;
            ChannelId = ChannelGuid;
        }

        // Side channel is send-only from Unity side — we never receive from Python.
        protected override void OnMessageReceived(IncomingMessage msg) { }

        // ── Recording API (called by PhysicalMachine, AGVController, SimulationBridge) ──

        /// <summary>Called by PhysicalMachine.TickTTF when FailedFlag is set.</summary>
        public void RecordMachineFailure(int machineId, float observedTtf, float repairDuration)
        {
            _events.Add(new TelemetryEvent
            {
                type = "machine_failure",
                machineId = machineId,
                ttf = observedTtf,
                repairDuration = repairDuration,
                simTime = Time.time,
            });
        }

        /// <summary>Called by PhysicalMachine.AcknowledgeRepairComplete.</summary>
        public void RecordMachineRepairComplete(int machineId, float actualRepairDuration)
        {
            _events.Add(new TelemetryEvent
            {
                type = "machine_repair_complete",
                machineId = machineId,
                repairDuration = actualRepairDuration,
                simTime = Time.time,
            });
        }

        /// <summary>Called by AGVController when Weibull failure fires (Phase 3).</summary>
        public void RecordAGVFailure(int agvId, float observedTtf, float repairDuration)
        {
            _events.Add(new TelemetryEvent
            {
                type = "agv_failure",
                agvId = agvId,
                ttf = observedTtf,
                repairDuration = repairDuration,
                simTime = Time.time,
            });
        }

        /// <summary>Called by PoissonClock when a dynamic job arrives (Phase 4).</summary>
        public void RecordJobArrival(int jobId, float interArrivalTime)
        {
            _events.Add(new TelemetryEvent
            {
                type = "job_arrival",
                jobId = jobId,
                interArrivalTime = interArrivalTime,
                simTime = Time.time,
            });
        }

        /// <summary>
        /// Called by SimulationBridge at episode end. Stores the final result
        /// so it is included in the flushed payload alongside event data.
        /// </summary>
        public void RecordEpisodeResult(double makespan, int jobCount, int machineCount,
                                        int totalOps, int decisions, double totalReward,
                                        string ruleName, string stochasticTag)
        {
            _result = new EpisodeResult
            {
                makespan = makespan,
                jobCount = jobCount,
                machineCount = machineCount,
                totalOps = totalOps,
                decisions = decisions,
                totalReward = totalReward,
                ruleName = ruleName,
                stochasticTag = stochasticTag,
            };
        }

        /// <summary>
        /// Serialises all collected events + episode result and sends to Python.
        /// Called by SimulationBridge just before EndEpisode().
        /// Clears the event list for the next episode.
        /// </summary>
        public void Flush()
        {
            var payload = new TelemetryPayload
            {
                events = _events,
                result = _result,
            };

            string json = JsonConvert.SerializeObject(payload);

            using (var msg = new OutgoingMessage())
            {
                msg.WriteString(json);
                QueueMessageToSend(msg);
            }

            SimLogger.Low($"[TelemetryChannel] Flushed {_events.Count} events to Python.");
            _events.Clear();
            _result = null;
        }

        // ── Data structures ──────────────────────────────────────────────────

        [Serializable]
        private class TelemetryEvent
        {
            public string type;
            public int machineId = -1;
            public int agvId = -1;
            public int jobId = -1;
            public float ttf = 0f;
            public float repairDuration = 0f;
            public float interArrivalTime = 0f;
            public float simTime = 0f;
        }

        [Serializable]
        private class EpisodeResult
        {
            public double makespan;
            public int jobCount;
            public int machineCount;
            public int totalOps;
            public int decisions;
            public double totalReward;
            public string ruleName;
            public string stochasticTag;
        }

        [Serializable]
        private class TelemetryPayload
        {
            public List<TelemetryEvent> events;
            public EpisodeResult result;
        }
    }
}