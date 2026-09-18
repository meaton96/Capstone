using Assets.Scripts.Simulation.Logging;
using Unity.MLAgents.Sensors;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// Registers <see cref="RewardMetricsSensor"/> with the agent. Added at runtime by
    /// SchedulingAgent.Awake, so the scene needs no changes; Agent.OnEnable collects it.
    /// </summary>
    public class RewardMetricsSensorComponent : SensorComponent
    {
        public override ISensor[] CreateSensors()
        {
            SimLogger.Low($"[RewardMetrics] Creating sensor for component {GetInstanceID()} " +
                          $"on '{gameObject.name}'.");
            return new ISensor[] { new RewardMetricsSensor() };
        }
    }
}
