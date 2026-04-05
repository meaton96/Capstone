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
    ///
    /// The agent is gated by @c IsArmed — ML-Agents' Academy will call
    /// @c OnEpisodeBegin() immediately on FixedUpdate, but we only proceed
    /// if something (UI button, batch runner, or autoStartOnPlay) has armed us.
    public class SchedulingAgent : Agent
    {
        [Header("References")]
        [SerializeField] private SimulationBridge bridge;
        [SerializeField] private int maxCandidateSlots = 3;

        [Header("Observation Config")]
        [SerializeField] private int maxQueueSlots = 10;
        public int ObservationSize => 5 + 2 + (maxQueueSlots * 2) + 2 + (maxCandidateSlots * 3);

        [Header("Heuristic / Baseline Config")]
        [SerializeField] private DispatchingRule heuristicRule = DispatchingRule.SPT_SMPT;

        [SerializeField] private bool logDecisions = true;

        // ─────────────────────────────────────────────────────────
        //  Episode Gating
        // ─────────────────────────────────────────────────────────

        /// @brief When false, OnEpisodeBegin() is a no-op.
        ///        Set to true by the UI "Start" button, batch runner,
        ///        or SimulationBridge.autoStartOnPlay before the first
        ///        Academy step fires.
        public bool IsArmed { get; set; }

        /// @brief Arms the agent and immediately requests an episode reset.
        ///        Call this from UI buttons or the batch runner.
        public void ArmAndStart()
        {
            IsArmed = true;

            // If Academy already ran its first step and the agent is idle,
            // we need to manually trigger a new episode.
            // EndEpisode() → Academy calls OnEpisodeBegin() next step.
            EndEpisode();
        }

        /// @brief Public setter so the batch runner can swap rules without reflection.
        public void SetHeuristicRule(DispatchingRule rule)
        {
            heuristicRule = rule;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            actionsOut.DiscreteActions.Array[0] = SimulationBridge.Instance.GetRuleIndex(heuristicRule);

            if (logDecisions)
            {
                string ruleName = heuristicRule.ToString();
                string decType = bridge.CurrentDecision.Type.ToString();
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Unity Lifecycle
        // ─────────────────────────────────────────────────────────

        protected override void OnEnable()
        {
            base.OnEnable();
            if (bridge != null)
                bridge.OnDecisionRequired.AddListener(HandleDecisionRequired);
        }

        protected override void OnDisable()
        {
            base.OnDisable();
            if (bridge != null)
                bridge.OnDecisionRequired.RemoveListener(HandleDecisionRequired);
        }

        // ─────────────────────────────────────────────────────────
        //  ML-Agents Lifecycle
        // ─────────────────────────────────────────────────────────

        public override void Initialize() { }

        /// @brief Called by Academy every time it wants a new episode.
        ///
        /// @details ML-Agents fires this on the very first FixedUpdate AND
        ///          after every EndEpisode() call. We gate it with IsArmed
        ///          so the factory only spawns when something has explicitly
        ///          asked for it (UI click, batch runner, or autoStartOnPlay).
        public override void OnEpisodeBegin()
        {
            if (!IsArmed)
            {
                SimLogger.Low("[Agent] OnEpisodeBegin skipped — not armed.");
                return;
            }

            if (bridge == null)
            {
                SimLogger.Error("[Agent] Bridge not assigned.");
                return;
            }

            bridge.StartEpisode();
        }

        // ─────────────────────────────────────────────────────────
        //  Event Handlers
        // ─────────────────────────────────────────────────────────

        private void HandleDecisionRequired(DecisionRequest req)
        {
            RequestDecision();
        }

        // ─────────────────────────────────────────────────────────
        //  Observations
        // ─────────────────────────────────────────────────────────

        public override void CollectObservations(VectorSensor sensor)
        {
            DecisionRequest req = bridge.CurrentDecision;
            if (!bridge.IsEpisodeActive) { PadZeros(sensor); return; }

            sensor.AddObservation((float)req.Type);
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
                // pad routing slots
                sensor.AddObservation(0);
                sensor.AddObservation(0);
                for (int i = 0; i < maxCandidateSlots; i++)
                {
                    sensor.AddObservation(0);
                    sensor.AddObservation(0f);
                    sensor.AddObservation(0f);
                }
            }
            else // Routing
            {
                sensor.AddObservation(req.JobId);
                sensor.AddObservation((float)req.RequiredType);
                // pad dispatch slots
                sensor.AddObservation(0);
                sensor.AddObservation(0);
                for (int i = 0; i < maxQueueSlots; i++)
                {
                    sensor.AddObservation(0);
                    sensor.AddObservation(0f);
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

        // ─────────────────────────────────────────────────────────
        //  Actions
        // ─────────────────────────────────────────────────────────

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
    }
}