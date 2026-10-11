using System;
using Unity.MLAgents.SideChannels;

namespace Assets.Scripts.Simulation.Channels
{
    /// <summary>
    /// Unity -> Python: the raw candidate view of each agent decision, as JSON (Python rule mode, PyRules; 2026-10-11).
    /// Registered only with -pyrules. SchedulingAgent.CollectObservations queues one message per decision; ML-Agents
    /// sends queued side-channel messages with the step being sent, so Python reads the view of exactly the decision
    /// whose observation it receives (env/channels/channels.py DecisionViewChannel, env/rules_api/unity_adapter.py).
    /// Python never sends on this channel.
    /// </summary>
    public class DecisionViewChannel : SideChannel
    {
        public static readonly Guid ChannelGuid = new Guid("d4a5b6c7-e8f9-4a0b-9c1d-2e3f4a5b6c7d");

        public static DecisionViewChannel Instance { get; private set; }

        public DecisionViewChannel()
        {
            Instance = this;
            ChannelId = ChannelGuid;
        }

        protected override void OnMessageReceived(IncomingMessage msg) { }

        public void SendView(string json)
        {
            using (var msg = new OutgoingMessage())
            {
                msg.WriteString(json);
                QueueMessageToSend(msg);
            }
        }
    }
}
