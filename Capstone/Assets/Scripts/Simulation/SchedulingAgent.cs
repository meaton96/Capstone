using Assets.Scripts.Logging;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// ML-Agents Agent subclass that drives job-shop scheduling decisions.
    /// 
    /// Listens for DecisionRequest events from the SimulationBridge,
    /// collects a fixed-width observation vector, and maps a discrete action 
    /// index to a dispatching rule applied via SimulationBridge.Step().
    /// </summary>
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

        public bool IsArmed { get; set; }

        public void ArmAndStart()
        {
            IsArmed = true;
            EndEpisode();
        }

        public void SetHeuristicRule(DispatchingRule rule)
        {
            heuristicRule = rule;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            actionsOut.DiscreteActions.Array[0] = SimulationBridge.Instance.GetRuleIndex(heuristicRule);

            if (logDecisions && bridge.CurrentDecision != null)
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
            {
                bridge.OnDecisionRequired.AddListener(HandleDecisionRequired);
                bridge.OnEpisodeFinished.AddListener(HandleEpisodeFinished); // NEW: Listen for natural completion
            }
        }

        protected override void OnDisable()
        {
            base.OnDisable();
            if (bridge != null)
            {
                bridge.OnDecisionRequired.RemoveListener(HandleDecisionRequired);
                bridge.OnEpisodeFinished.RemoveListener(HandleEpisodeFinished);
            }
        }

        // ─────────────────────────────────────────────────────────
        //  ML-Agents Lifecycle
        // ─────────────────────────────────────────────────────────

        public override void Initialize() { }

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

        private void HandleEpisodeFinished(EpisodeResult result)
        {
            // The simulation has naturally reached the end (all jobs exited).
            // Tell ML-Agents to finalize the episode and loop back to OnEpisodeBegin.
            EndEpisode();
        }

        // ─────────────────────────────────────────────────────────
        //  Observations
        // ─────────────────────────────────────────────────────────

        public override void CollectObservations(VectorSensor sensor)
        {
            DecisionRequest req = bridge.CurrentDecision;
            if (!bridge.IsEpisodeActive || req == null) { PadZeros(sensor); return; }

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

        }
    }
}