using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

namespace Assets.Scripts.Simulation.Channels
{
    public class SideChannelRegistrar : MonoBehaviour
    {
        private EpisodeConfigChannel _configChannel;
        private EpisodeTelemetryChannel _telemetryChannel;

        private void Awake()
        {
            _configChannel = new EpisodeConfigChannel();
            _telemetryChannel = new EpisodeTelemetryChannel();

            SideChannelManager.RegisterSideChannel(_configChannel);
            SideChannelManager.RegisterSideChannel(_telemetryChannel);
        }

        private void OnDestroy()
        {
            SideChannelManager.UnregisterSideChannel(_configChannel);
            SideChannelManager.UnregisterSideChannel(_telemetryChannel);
        }
    }
}