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
        private TextAsset instanceJson;

        [Header("Observation Config")]
        [SerializeField] private int maxQueueSlots = 10;

        // ─── Unity Lifecycle ─────────────────────────────────

        protected override void OnEnable()
        {
            base.OnEnable();
            if (bridge != null)
            {
                // Listen for the Bridge to tell us a machine is physically ready
                bridge.OnDecisionRequired.AddListener(HandleDecisionRequired);
            }
        }

        protected override void OnDisable()
        {
            base.OnDisable();
            if (bridge != null)
            {
                bridge.OnDecisionRequired.RemoveListener(HandleDecisionRequired);
            }
        }

        // ─── ML-Agents Lifecycle ─────────────────────────────

        public override void Initialize()
        {
            // You can safely manage your time scale from here or a global manager
            // Time.timeScale = 1.0f; // Set to 100f for fast training later
        }

        public override void OnEpisodeBegin()
        {
            if (bridge == null)
            {
                Debug.LogWarning("[Agent] Bridge not assigned.");
                return;
            }

            // Start the physics episode. 
            // 1. The Bridge will spawn jobs.
            // 2. The jobs will physically drop into the Machine trigger colliders.
            // 3. The PhysicalMachine will tell the Bridge.
            // 4. The Bridge will fire OnDecisionRequired to wake this agent up.
            //bridge.TaillardJson = instanceJson;
            instanceJson = bridge.TaillardJson;
            bridge.StartEpisode();
        }

        // ─── Event Handlers ──────────────────────────────────

        private void HandleDecisionRequired(DecisionRequest req)
        {
            // This wakes up the ML-Agent, forcing it to CollectObservations 
            // and output an action to OnActionReceived.
            RequestDecision();
        }

        // ─── Agent Logic ─────────────────────────────────────

        public override void CollectObservations(VectorSensor sensor)
        {
            DecisionRequest req = bridge.CurrentDecision;

            // Guard: if the bridge isn't waiting for us, write zeros
            bool valid = bridge.IsEpisodeActive
                         && req.QueuedJobIds != null
                         && req.QueuedDurations != null;

            if (!valid)
            {
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
            // Double check that the bridge actually wants an action
            if (!bridge.IsWaitingForAction) return;

            int pdrIndex = actions.DiscreteActions[0];

            // Apply the step to the physical machine
            StepResult result = bridge.Step(pdrIndex);

            // Log the reward from the elapsed physics time
            AddReward(result.Reward);

            if (result.Done)
            {
                EndEpisode();
            }
            // Notice there is no RequestDecision() here anymore!
            // We patiently wait for the next HandleDecisionRequired event.
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            actionsOut.DiscreteActions.Array[0] = 0;
        }

        public int ObservationSize => 6 + (maxQueueSlots * 2);
    }
}