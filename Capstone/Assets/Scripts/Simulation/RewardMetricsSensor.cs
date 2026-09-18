using Unity.MLAgents.Sensors;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// Second observation stream carrying a <see cref="RewardMetrics"/> snapshot alongside the
    /// policy's vector observation. Sending it as an observation (rather than a side-channel
    /// message) guarantees each snapshot arrives attached to exactly the agent step it describes,
    /// including the terminal step. The Python env wrapper strips it from the policy input and
    /// hands it only to the reward function.
    /// </summary>
    public class RewardMetricsSensor : ISensor
    {
        /// <summary>
        /// ML-Agents orders an agent's sensors by name. This must sort after the default
        /// "VectorSensor_size…" name so the policy observation stays at index 0.
        /// </summary>
        public const string SensorName = "Z_RewardMetrics";

        private readonly float[] _values = new float[RewardMetrics.Count];
        private readonly ObservationSpec _spec = ObservationSpec.Vector(RewardMetrics.Count);

        /// <summary>
        /// Captures the snapshot at serialization time. ML-Agents calls Write only on the sensor
        /// instance it registered, right after CollectObservations for a decision and inside
        /// EndEpisode for the terminal step, so the snapshot always matches the step being sent.
        /// (Capturing from SchedulingAgent.CollectObservations via the component's sensor reference
        /// silently never reached the registered instance in the player build.)
        /// </summary>
        public int Write(ObservationWriter writer)
        {
            FactoryOrchestrator.Instance?.WriteRewardMetrics(_values);
            writer.AddList(_values);
            return _values.Length;
        }

        public ObservationSpec GetObservationSpec() => _spec;
        public byte[] GetCompressedObservation() => null;
        public CompressionSpec GetCompressionSpec() => CompressionSpec.Default();
        public string GetName() => SensorName;
        public void Update() { }
        public void Reset() { }
    }
}
