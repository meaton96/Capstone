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
        [SerializeField] private int maxCandidateSlots = 3;
        //  private TextAsset instanceJson;

        [Header("Observation Config")]
        [SerializeField] private int maxQueueSlots = 10;
        public int ObservationSize => 5 + 2 + (maxQueueSlots * 2) + 2 + (maxCandidateSlots * 3);
        [Header("Heuristic / Baseline Config")]
        [SerializeField] private DispatchingRule heuristicRule = DispatchingRule.SPT_SMPT;

        [SerializeField] private bool logDecisions = true;

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            actionsOut.DiscreteActions.Array[0] = SimulationBridge.Instance.GetRuleIndex(heuristicRule);

            if (logDecisions)
            {
                string ruleName = heuristicRule.ToString();
                string decType = bridge.CurrentDecision.Type.ToString();
                //  Debug.Log($"[Heuristic] {decType} → rule {ruleName}");
            }
        }

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

            // if (bridge.TaillardJson == null)
            //     return;

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
            if (!bridge.IsEpisodeActive) { PadZeros(sensor); return; }

            sensor.AddObservation((float)req.Type);   // 0=Dispatch, 1=Routing
            sensor.AddObservation((float)req.SimTime);
            sensor.AddObservation(req.DecisionIndex);
            sensor.AddObservation(req.TotalJobs);
            sensor.AddObservation(req.CompletedJobs);

            if (req.Type == DecisionType.Dispatch)
            {
                sensor.AddObservation(req.MachineId);
                sensor.AddObservation(req.QueuedJobIds?.Length ?? 0);
                for (int i = 0; i < maxQueueSlots; i++)
                {
                    bool valid = req.QueuedJobIds != null && i < req.QueuedJobIds.Length;
                    sensor.AddObservation(valid ? req.QueuedJobIds[i] : 0);
                    sensor.AddObservation(valid ? (float)req.QueuedDurations[i] : 0f);
                }
                // pad routing slots — must match routing branch exactly
                sensor.AddObservation(0); // jobId placeholder
                sensor.AddObservation(0); // requiredType placeholder
                for (int i = 0; i < maxCandidateSlots; i++)
                {
                    sensor.AddObservation(0); // machineId
                    sensor.AddObservation(0f); // jobTime
                    sensor.AddObservation(0f); // queueLength
                }
            }
            else // Routing
            {
                sensor.AddObservation(req.JobId);
                sensor.AddObservation((float)req.RequiredType);
                // pad dispatch slots — must match dispatch branch exactly
                sensor.AddObservation(0); // machineId placeholder
                sensor.AddObservation(0); // queueLength placeholder
                for (int i = 0; i < maxQueueSlots; i++)
                {
                    sensor.AddObservation(0); // jobId
                    sensor.AddObservation(0f); // duration
                }
                for (int i = 0; i < maxCandidateSlots; i++)
                {
                    bool valid = req.CandidateMachineIds != null && i < req.CandidateMachineIds.Length;
                    sensor.AddObservation(valid ? req.CandidateMachineIds[i] : 0);
                    sensor.AddObservation(valid ? req.CandidateJobTimes[i] : 0f);
                    sensor.AddObservation(valid ? req.CandidateQueueLengths[i] : 0f);
                }
            }
        }

        private void PadZeros(VectorSensor sensor)
        {
            for (int i = 0; i < ObservationSize; i++) sensor.AddObservation(0f);
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
        // public override void Heuristic(in ActionBuffers actionsOut)
        // {
        //     actionsOut.DiscreteActions.Array[0] = 0;
        // }

        /// @brief Total number of floats in one observation vector.
        //public int ObservationSize => 6 + (maxQueueSlots * 2);
    }
}