using Assets.Scripts.Logging;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation
{
    /// @brief ML-Agents Agent subclass that drives job-shop scheduling decisions.
    ///
    /// @details Listens for DecisionRequest events from the @c SimulationBridge, 
    /// collects fixed-width observation vectors, and maps discrete action indices 
    /// to dispatching rules.
    public class SchedulingAgent : Agent
    {
        [Header("References")]
        [SerializeField] private SimulationBridge bridge;
        //  [SerializeField] private int maxCandidateSlots = 3;

        private ObservationBuilder _obsBuilder;

        [Header("Observation Config")]
        // [SerializeField] private int maxQueueSlots = 10;

        /// @brief The calculated size of the observation vector for ML-Agents.
        public int ObservationSize => ObservationBuilder.TotalObservationSize;

        [Header("Heuristic / Baseline Config")]
        [SerializeField] private DispatchingRule heuristicRule = DispatchingRule.SPT_SMPT;

        [SerializeField] private bool logDecisions = true;

        /// @brief Gating property to control when an episode is allowed to start.
        public bool IsArmed { get; set; }

        /// @brief Forces the agent into an active state and ends any current episode to trigger a reset.
        public void ArmAndStart()
        {
            IsArmed = true;
            EndEpisode();
        }


        /// @brief Sets the rule used when the agent is running in Heuristic mode.
        ///
        /// @param rule The @c DispatchingRule to apply.
        public void SetHeuristicRule(DispatchingRule rule)
        {
            heuristicRule = rule;
        }

        /// @brief Provides a baseline action based on a hardcoded dispatching rule.
        ///
        /// @param actionsOut The action buffer to be populated by the heuristic.
        public override void Heuristic(in ActionBuffers actionsOut)
        {
            actionsOut.DiscreteActions.Array[0] = SimulationBridge.Instance.GetRuleIndex(heuristicRule);
        }

        /// @brief Subscribes to simulation events when the component is enabled.
        protected override void OnEnable()
        {
            base.OnEnable();
            if (bridge != null)
            {
                bridge.OnDecisionRequired.AddListener(HandleDecisionRequired);
                bridge.OnEpisodeFinished.AddListener(HandleEpisodeFinished);
            }
        }

        /// @brief Unsubscribes from simulation events when the component is disabled.
        protected override void OnDisable()
        {
            base.OnDisable();
            if (bridge != null)
            {
                bridge.OnDecisionRequired.RemoveListener(HandleDecisionRequired);
                bridge.OnEpisodeFinished.RemoveListener(HandleEpisodeFinished);
            }
        }

        public override void Initialize()
        {
            _obsBuilder = new ObservationBuilder(bridge);
        }

        /// @brief Prepares the simulation and internal state for a new episode.
        ///
        /// @details Consumes the "armed" ticket to prevent runaway looping in batch 
        /// modes and triggers @c SimulationBridge.StartEpisode.
        public override void OnEpisodeBegin()
        {
            Initialize();
            if (bridge != null && bridge.AutoStartOnPlay)
            {
                IsArmed = true;
            }

            if (!IsArmed)
            {
                SimLogger.Low("[Agent] OnEpisodeBegin skipped — waiting for UI to arm.");
                return;
            }

            if (bridge == null) return;

            if (!bridge.AutoStartOnPlay)
            {
                IsArmed = false;
            }

            bridge.StartEpisode();
        }

        /// @brief Relays the decision requirement from the bridge to ML-Agents.
        ///
        /// @param req The @c DecisionRequest context.
        private void HandleDecisionRequired(DecisionRequest req)
        {
            RequestDecision();
        }

        /// @brief Handles the termination of a simulation run.
        ///
        /// @param result The final metrics of the completed episode.
        ///
        /// @details Only calls @c EndEpisode directly if in @c AutoStartOnPlay mode 
        /// to allow external runners to process results before resetting.
        private void HandleEpisodeFinished(EpisodeResult result)
        {
            SimLogger.Low("[Agent] End Episode");
            if (bridge != null && bridge.AutoStartOnPlay)
                EndEpisode();
        }

        /// @brief Populates the ML-Agents observation vector with environment state data.
        ///
        /// @param sensor The vector sensor to write observations into.
        ///
        /// @details Observations include simulation time, job progress, and context-specific 
        /// data for either @c Dispatch or @c Routing decision types.
        public override void CollectObservations(VectorSensor sensor)
        {
            DecisionRequest req = bridge.CurrentDecision;
            if (!bridge.IsEpisodeActive || req == null)
            {
                PadZeros(sensor);
                return;
            }

            // Get the massive 1D array containing all 5 streams
            float[] snapshot = _obsBuilder.BuildCompleteSnapshot(req);

            // Feed it to the ML-Agents sensor
            foreach (float val in snapshot)
            {
                sensor.AddObservation(val);
            }
        }

        /// @brief Fills the observation vector with zeros if the agent is inactive.
        private void PadZeros(VectorSensor sensor)
        {
            for (int i = 0; i < ObservationSize; i++) sensor.AddObservation(0f);
        }

        /// @brief Processes the discrete action index returned by the neural network.
        ///
        /// @param actions The buffer containing the predicted actions.
        ///
        /// @details Maps the action to a dispatching rule, steps the simulation via 
        /// @c bridge.Step, and applies the resulting reward to the agent.
        public override void OnActionReceived(ActionBuffers actions)
        {
            if (!bridge.IsWaitingForAction) return;

            int pdrIndex = actions.DiscreteActions[0];
            StepResult result = bridge.Step(pdrIndex);
            AddReward(result.Reward);
        }
    }
}