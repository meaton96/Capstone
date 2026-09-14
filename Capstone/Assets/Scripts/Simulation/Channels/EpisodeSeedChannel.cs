using System;
using System.Collections.Generic;
using Assets.Scripts.Simulation.Logging;
using Newtonsoft.Json.Linq;
using Unity.MLAgents.SideChannels;

namespace Assets.Scripts.Simulation.Channels
{
    /// <summary>
    /// Receives per-episode instance seeds from Python. Seeds queue up and each StartEpisode
    /// consumes one, so Python can line up seeds for episodes that Unity starts before Python
    /// has even seen the previous episode end (episodes roll over inside Unity).
    ///
    /// Message JSON: {"seeds": [int, ...], "clear": bool}. "clear" empties the queue and resets
    /// the consumed-seed counter before appending, so an evaluation run can index episodes from 0.
    /// The seed and its queue index are reported back through RewardMetrics (episode_seed,
    /// episode_seed_index); both are -1 for episodes started with an empty queue.
    ///
    /// Channel GUID must match env/channels/channels.py exactly.
    /// </summary>
    public class EpisodeSeedChannel : SideChannel
    {
        public static readonly Guid ChannelGuid =
            new Guid("d3a4b5c6-e7f8-9012-def0-123456789013");

        public static EpisodeSeedChannel Instance { get; private set; }

        /// <summary>
        /// Seeds must be below this: they are reported to Python as float32 metrics, which hold
        /// integers exactly only up to 2^24.
        /// </summary>
        public const int MaxSeed = 1 << 24;

        private readonly Queue<int> _seeds = new Queue<int>();
        private readonly object _lock = new object();
        private int _consumed;

        public EpisodeSeedChannel()
        {
            Instance = this;
            ChannelId = ChannelGuid;
        }

        protected override void OnMessageReceived(IncomingMessage msg)
        {
            string json = msg.ReadString();
            try
            {
                JObject root = JObject.Parse(json);
                var incoming = new List<int>();
                if (root["seeds"] is JArray seeds)
                {
                    foreach (JToken token in seeds)
                    {
                        int seed = token.Value<int>();
                        if (seed < 0 || seed >= MaxSeed)
                            throw new ArgumentOutOfRangeException(nameof(seed), $"seed {seed} is outside [0, {MaxSeed})");
                        incoming.Add(seed);
                    }
                }

                bool clear = root["clear"]?.Value<bool>() ?? false;
                lock (_lock)
                {
                    if (clear)
                    {
                        _seeds.Clear();
                        _consumed = 0;
                    }
                    foreach (int seed in incoming)
                        _seeds.Enqueue(seed);
                }
                SimLogger.Low($"[SeedChannel] Queued {incoming.Count} seeds (clear={clear}).");
            }
            catch (Exception ex)
            {
                SimLogger.LogError($"[SeedChannel] Failed to parse seed message: {ex.Message}");
            }
        }

        /// <summary>
        /// Pops the next seed and its position since the last clear. Returns false, with both
        /// set to -1, when the queue is empty.
        /// </summary>
        public bool TryDequeue(out int seed, out int index)
        {
            lock (_lock)
            {
                if (_seeds.Count == 0)
                {
                    seed = -1;
                    index = -1;
                    return false;
                }
                seed = _seeds.Dequeue();
                index = _consumed++;
                return true;
            }
        }
    }
}
