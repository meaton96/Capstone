using Assets.Scripts.Logging;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation
{
    /// @brief ML-Agents @c Agent subclass that drives job-shop scheduling decisions.
    ///
    /// @details Listens for @c DecisionRequest events from the @c SimulationBridge,
    /// collects a fixed-width observation vector describing the current machine queue,
    /// and maps a discrete action index to a dispatching rule applied via
    /// @c SimulationBridge.Step().
    public class SchedulingAgent : Agent
    {
        [Header("References")]
        [SerializeField] private SimulationBridge bridge;
        //  private TextAsset instanceJson;

        [Header("Observation Config")]
        [SerializeField] private int maxQueueSlots = 10;

        // ─────────────────────────────────────────────────────────
        //  Unity Lifecycle
        // ─────────────────────────────────────────────────────────

        /// @brief Subscribes to the bridge's decision-required event.
        protected override void OnEnable()
        {
            base.OnEnable();
            if (bridge != null)
            {
                bridge.OnDecisionRequired.AddListener(HandleDecisionRequired);
            }
        }

        /// @brief Unsubscribes from the bridge's decision-required event.
        protected override void OnDisable()
        {
            base.OnDisable();
            if (bridge != null)
            {
                bridge.OnDecisionRequired.RemoveListener(HandleDecisionRequired);
            }
        }

        // ─────────────────────────────────────────────────────────
        //  ML-Agents Lifecycle
        // ─────────────────────────────────────────────────────────

        /// @brief One-time agent setup called by ML-Agents on first activation.
        public override void Initialize()
        {
        }

        /// @brief Resets the simulation and begins a new physics episode.
        /// @details Delegates full episode setup to @c SimulationBridge.StartEpisode().
        ///          The bridge spawns jobs which physically enter machine trigger
        ///          colliders, causing @c HandleDecisionRequired to fire and wake
        ///          this agent when the first scheduling decision is needed.
        public override void OnEpisodeBegin()
        {
            if (bridge == null)
            {
                SimLogger.Error("[Agent] Bridge not assigned.");
                return;
            }

            if (bridge.TaillardJson == null)
                return;

            bridge.StartEpisode();
        }

        // ─────────────────────────────────────────────────────────
        //  Event Handlers
        // ─────────────────────────────────────────────────────────

        /// @brief Invoked by the bridge when a machine needs a scheduling decision.
        /// @param req  Contextual data describing the waiting machine and its queue.
        private void HandleDecisionRequired(DecisionRequest req)
        {
            RequestDecision();
        }

        // ─────────────────────────────────────────────────────────
        //  Agent Logic
        // ─────────────────────────────────────────────────────────

        /// @brief Builds the observation vector sent to the neural network.
        ///
        /// @details Writes six scalar context values followed by a fixed-width
        ///          padded block of @c maxQueueSlots (job-id, duration) pairs.
        ///          All slots beyond the actual queue length are zero-padded so
        ///          the vector size remains constant across episodes.
        ///
        /// @param sensor  The @c VectorSensor provided by ML-Agents.
        public override void CollectObservations(VectorSensor sensor)
        {
            DecisionRequest req = bridge.CurrentDecision;

            bool valid = bridge.IsEpisodeActive
                         && req.QueuedJobIds != null
                         && req.QueuedDurations != null;

            if (!valid)
            {
                for (int i = 0; i < ObservationSize; i++)
                    sensor.AddObservation(0f);
                return;
            }

            sensor.AddObservation(req.MachineId);
            sensor.AddObservation((float)req.SimTime);
            sensor.AddObservation(req.DecisionIndex);
            sensor.AddObservation(req.TotalJobs);
            sensor.AddObservation(req.CompletedJobs);
            sensor.AddObservation(req.QueuedJobIds.Length);

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

        /// @brief Receives the network's chosen action and applies it to the simulation.
        ///
        /// @details Maps the discrete action index to a dispatching rule via
        ///          @c SimulationBridge.Step(), accumulates the returned reward,
        ///          and ends the episode if the bridge signals completion.
        ///          The next @c RequestDecision() call is driven by the
        ///          @c HandleDecisionRequired event, not called here directly.
        ///
        /// @param actions  Action buffers provided by ML-Agents.
        public override void OnActionReceived(ActionBuffers actions)
        {
            if (!bridge.IsWaitingForAction) return;

            int pdrIndex = actions.DiscreteActions[0];

            StepResult result = bridge.Step(pdrIndex);

            AddReward(result.Reward);

            if (result.Done)
            {
                EndEpisode();
            }
        }

        /// @brief Heuristic fallback: always selects action 0 (first dispatching rule).
        /// @param actionsOut  Action buffer to write the heuristic choice into.
        public override void Heuristic(in ActionBuffers actionsOut)
        {
            actionsOut.DiscreteActions.Array[0] = 0;
        }

        /// @brief Total number of floats in one observation vector.
        public int ObservationSize => 6 + (maxQueueSlots * 2);
    }
}