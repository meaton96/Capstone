using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace Assets.Scripts.Simulation
{
    public class SchedulingAgent : Agent
    {
        [Header("References")]
        [SerializeField] private SimulationBridge bridge;
        [SerializeField] private TextAsset instanceJson;

        [Header("Observation Config")]
        [SerializeField] private int maxQueueSlots = 10;

        /// <summary>
        /// True when the bridge failed to start and we need to
        /// end the episode on the next Academy step (not inside
        /// OnEpisodeBegin, which would cause recursion).
        /// </summary>
        private bool needsDeferredReset;

        // ─── ML-Agents lifecycle ─────────────────────────────

        public override void Initialize()
        {
            // Turn off visuals for training throughput.
            // bridge.EnableVisuals = false;  // uncomment once you add the setter
        }

        public override void OnEpisodeBegin()
        {
            // The Academy may call this before SimulationBridge.Awake()
            // has initialized the simulator. Guard against that.
            if (bridge == null || bridge.Simulator == null)
            {
                Debug.LogWarning("[Agent] Bridge not ready yet. Deferring.");
                needsDeferredReset = true;
                return;
            }

            bridge.StartEpisode(instanceJson);

            if (bridge.IsDone)
            {
                // Can't call EndEpisode() here — ML-Agents forbids it
                // inside OnEpisodeBegin. Defer to next FixedUpdate.
                Debug.LogWarning("[Agent] Episode had no decisions. Deferring reset.");
                needsDeferredReset = true;
                return;
            }

            needsDeferredReset = false;
            RequestDecision();
        }

        private void FixedUpdate()
        {
            // Handle the deferred reset outside of OnEpisodeBegin.
            if (needsDeferredReset)
            {
                needsDeferredReset = false;
                EndEpisode(); // safe here — not inside OnEpisodeBegin
            }
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            DecisionRequest req = bridge.CurrentDecision;

            // Guard: if the bridge hasn't produced a decision yet,
            // write zeros so the observation size still matches.
            bool valid = bridge.IsEpisodeActive
                         && req.QueuedJobIds != null
                         && req.QueuedDurations != null;

            if (!valid)
            {
                // Write the correct number of zeros.
                for (int i = 0; i < ObservationSize; i++)
                    sensor.AddObservation(0f);
                return;
            }

            // ── Scalar context ───────────────────────────────
            sensor.AddObservation(req.MachineId);
            sensor.AddObservation((float)req.SimTime);
            sensor.AddObservation(req.DecisionIndex);
            sensor.AddObservation(req.TotalJobs);
            sensor.AddObservation(req.CompletedJobs);
            sensor.AddObservation(req.QueuedJobIds.Length);

            // ── Queue contents (fixed-width, zero-padded) ────
            for (int i = 0; i < maxQueueSlots; i++)
            {
                if (i < req.QueuedJobIds.Length)
                {
                    sensor.AddObservation(req.QueuedJobIds[i]);
                    sensor.AddObservation((float)req.QueuedDurations[i]);
                }
                else
                {
                    sensor.AddObservation(0);
                    sensor.AddObservation(0f);
                }
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            if (!bridge.IsWaitingForAction)
            {
                return;
            }

            int pdrIndex = actions.DiscreteActions[0];
            Debug.Log($"[Agent] Decision #{bridge.DecisionCount}: chose action {pdrIndex}");

            StepResult result = bridge.Step(pdrIndex);
            AddReward(result.Reward);

            if (result.Done)
                EndEpisode();
            else
                RequestDecision();
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            actionsOut.DiscreteActions.Array[0] = 0;
        }

        public int ObservationSize => 6 + (maxQueueSlots * 2);
    }
}