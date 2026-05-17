using Assets.Scripts.Simulation.Logging;
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
        //[SerializeField] private FactoryOrchestrator orchestrator;
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
            if (heuristicRule == DispatchingRule.Random)
            {
                actionsOut.DiscreteActions.Array[0] = Random.Range(0, FactoryOrchestrator.ActionCount);
            }
            else
            {
                actionsOut.DiscreteActions.Array[0] = FactoryOrchestrator.Instance.GetRuleIndex(heuristicRule);
            }
        }

        /// @brief Subscribes to simulation events when the component is enabled.
        protected override void OnEnable()
        {
            base.OnEnable();
            if (FactoryOrchestrator.Instance != null)
            {
                FactoryOrchestrator.Instance.OnDecisionRequired.AddListener(HandleDecisionRequired);
                FactoryOrchestrator.Instance.OnEpisodeFinished.AddListener(HandleEpisodeFinished);
            }
        }

        /// @brief Unsubscribes from simulation events when the component is disabled.
        protected override void OnDisable()
        {
            base.OnDisable();
            if (FactoryOrchestrator.Instance != null)
            {
                FactoryOrchestrator.Instance.OnDecisionRequired.RemoveListener(HandleDecisionRequired);
                FactoryOrchestrator.Instance.OnEpisodeFinished.RemoveListener(HandleEpisodeFinished);
            }
        }

        public override void Initialize()
        {
            _obsBuilder = new ObservationBuilder(FactoryOrchestrator.Instance);
        }

        /// @brief Prepares the simulation and internal state for a new episode.
        ///
        /// @details Consumes the "armed" ticket to prevent runaway looping in batch 
        /// modes and triggers @c SimulationBridge.StartEpisode.
        public override void OnEpisodeBegin()
        {
            Initialize();
            if (FactoryOrchestrator.Instance != null && FactoryOrchestrator.Instance.AutoStartOnPlay)
            {
                IsArmed = true;
            }

            if (!IsArmed)
            {
                SimLogger.Low("[Agent] OnEpisodeBegin skipped — waiting for UI to arm.");
                return;
            }

            if (FactoryOrchestrator.Instance == null) return;

            if (!FactoryOrchestrator.Instance.AutoStartOnPlay)
            {
                IsArmed = false;
            }

            FactoryOrchestrator.Instance.StartEpisode();
        }

        /// @brief Relays the decision requirement from the bridge to ML-Agents.
        ///
        /// @param req The @c DecisionRequest context.
        private void HandleDecisionRequired(DecisionRequest req)
        {
            SimLogger.High($"[Agent] RequestDecision called — communicator={Academy.Instance.IsCommunicatorOn}");
            RequestDecision();
        }

        /// @brief Handles the termination of a simulation run.
        ///
        /// @param result The final metrics of the completed episode.
        ///
        /// @details Only calls @c EndEpisode directly if in @c AutoStartOnPlay mode 
        /// to allow external runners to process results before resetting.
        private void HandleEpisodeFinished(EpisodeRecord record)
        {
            SimLogger.Low("[Agent] End Episode");
            if (FactoryOrchestrator.Instance != null && FactoryOrchestrator.Instance.AutoStartOnPlay)
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
            DecisionRequest req = FactoryOrchestrator.Instance.CurrentDecision;
            if (!FactoryOrchestrator.Instance.IsEpisodeActive || req == null)
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
            if (!FactoryOrchestrator.Instance.IsWaitingForAction) return;

            int pdrIndex = actions.DiscreteActions[0];
            StepResult result = FactoryOrchestrator.Instance.Step(pdrIndex);
            AddReward(result.Reward);
        }
    }
}