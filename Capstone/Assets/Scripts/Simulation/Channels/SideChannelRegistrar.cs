using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

namespace Assets.Scripts.Simulation.Channels
{
    public class SideChannelRegistrar : MonoBehaviour
    {
        private EpisodeConfigChannel _configChannel;
        private EpisodeTelemetryChannel _telemetryChannel;
        private EpisodeSeedChannel _seedChannel;

        private void Awake()
        {
            _configChannel = new EpisodeConfigChannel();
            _telemetryChannel = new EpisodeTelemetryChannel();
            _seedChannel = new EpisodeSeedChannel();

            SideChannelManager.RegisterSideChannel(_configChannel);
            SideChannelManager.RegisterSideChannel(_telemetryChannel);
            SideChannelManager.RegisterSideChannel(_seedChannel);
        }

        private void OnDestroy()
        {
            SideChannelManager.UnregisterSideChannel(_configChannel);
            SideChannelManager.UnregisterSideChannel(_telemetryChannel);
            SideChannelManager.UnregisterSideChannel(_seedChannel);
        }
    }
}
