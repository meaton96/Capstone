using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Events;
using System.Collections.Generic;
using Assets.Scripts.Simulation.Logging;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.AGV;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Stochastic;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Channels;
using Unity.MLAgents;

namespace Assets.Scripts.Simulation
{
    public class FactoryOrchestrator : MonoBehaviour
    {
        public static FactoryOrchestrator Instance;

        [Header("Scene References")]
        [SerializeField] private FactoryLayoutManager layoutManager;
        [SerializeField] private TrafficZoneManager trafficZoneManager;
        [SerializeField] private AGVPool agvPool;
        [SerializeField] private SchedulingAgent agent;
        public JobStore Jobs;

        [Header("Episode Configuration")]
        public int PreDispatchLeadTime = 15;
        public bool AutoStartOnPlay = false;

        private FJSSPJobDefinition[] prebuiltJobs;

        private FJSSPConfig currentConfig;
        private Dictionary<MachineType, List<int>> cachedMachinesByType;
        public Dictionary<MachineType, List<int>> CachedMachinesByType => cachedMachinesByType;

        private bool episodeActive;
        private int decisionCount;
        public int DecisionCount => decisionCount;
        private double totalReward;
        private double previousMakespan;
        private float startTime;

        public bool IsEpisodeActive => episodeActive;
        public bool IsFactoryReady { get; set; }
        public double SimTime => Time.time - startTime;
        public FJSSPConfig CurrentConfig => currentConfig;

        public DecisionRequest CurrentDecision { get; private set; }
        public bool IsWaitingForAction { get; private set; }
        public string LastAppliedRule { get; private set; } = "Waiting...";

        private readonly EpisodeTracker _tracker = new();
        private readonly Dictionary<int, double> _machineProcessingStartTime = new();

        private FlagHarvester _flags;
        private FailureCoordinator _failures;
        private DecisionCoordinator _decisions;

        [Header("Events")]
        public UnityEvent<DecisionRequest> OnDecisionRequired;
        public UnityEvent<StepResult> OnStepCompleted;
        public UnityEvent<EpisodeRecord> OnEpisodeFinished;
        public UnityEvent OnFactorySpawned;

        public static int ActionCount => DispatchingEngine.ActionCount;
        public int GetRuleIndex(DispatchingRule rule) => DispatchingEngine.IndexForRule(rule);

        private void Awake()
        {
            if (Instance != null) { Destroy(this); return; }
            Instance = this;
        }

        private void Start()
        {
            if (AutoStartOnPlay && agent != null)
                agent.IsArmed = true;
        }

        public void LoadConfig(FJSSPConfig config)
        {
            currentConfig = config;
            IsFactoryReady = false;
            StochasticEventManager.Instance?.Initialize(config);
        }

        public void LoadPrebuiltJobs(FJSSPJobDefinition[] jobs)
        {
            prebuiltJobs = jobs;
        }

        public void SpawnFactory()
        {
            if (currentConfig == null) return;
            if (IsFactoryReady || episodeActive) StopEpisode();

            UnityEngine.Random.InitState(currentConfig.Seed);
            cachedMachinesByType = layoutManager.BuildFloor(currentConfig);
            trafficZoneManager.BuildZoneGraph();
            agvPool.InitializeFleet(currentConfig.AGVCount);

            IsFactoryReady = true;
            OnFactorySpawned?.Invoke();
        }

        public void StartEpisode()
        {
            var pythonConfig = EpisodeConfigChannel.Instance?.ConsumeConfig();
            if (pythonConfig != null)
            {
                currentConfig = pythonConfig;
                IsFactoryReady = false;
                SimLogger.Low($"[Bridge] Applied Python config: {currentConfig.Name}");
            }

            currentConfig ??= DefaultConfigFactory.BuildDefault();//DefaultConfigFactory.BuildDefaultStochastic();

            if (!IsFactoryReady)
                SpawnFactory();

            trafficZoneManager.ResetEpisodeStats();
            foreach (var agv in agvPool.AllAGVs)
                agv.ResetEpisodeStats();


            agent.SetHeuristicRule(currentConfig.dispatchingRule);

            FJSSPJobDefinition[] jobDefs;
            if (prebuiltJobs != null)
            {
                jobDefs = prebuiltJobs;
                prebuiltJobs = null;
                SimLogger.Low("[Orchestrator] Using prebuilt benchmark jobs");
            }
            else
            {
                jobDefs = FJSSPJobGenerator.Generate(currentConfig, cachedMachinesByType);
            }

            Jobs.Initialize(jobDefs, spawnVisuals: true);

            if (currentConfig.Stochastic != null && currentConfig.Stochastic.AnyEnabled)
            {
                StochasticEventManager.Instance?.Initialize(currentConfig);
                foreach (var machine in layoutManager.Machines)
                    machine.InitializeStochastic();
            }

            _tracker.Reset();
            _machineProcessingStartTime.Clear();

            _flags = new FlagHarvester();
            _flags.Initialize(Jobs, agvPool, layoutManager, _tracker, _machineProcessingStartTime);

            _failures = new FailureCoordinator();
            _failures.Initialize(
                Jobs, agvPool, layoutManager, _tracker, _machineProcessingStartTime,
                onMachineFailedInvalidateDecision: (machineId) =>
                {
                    if (IsWaitingForAction &&
                        CurrentDecision.Type == DecisionType.Dispatch &&
                        CurrentDecision.MachineId == machineId)
                    {
                        IsWaitingForAction = false;
                        SimLogger.Low($"[Orchestrator] Pending dispatch decision for machine " +
                                      $"{machineId} invalidated (machine failed).");
                    }
                },
                refreshLabels: _flags.RefreshMachineLabels
            );

            _decisions = new DecisionCoordinator();
            _decisions.Initialize(
                Jobs, layoutManager,
                getSimTime: () => SimTime,
                getDecisionCount: () => decisionCount,
                incrementDecisionCount: () => decisionCount++
            );

            episodeActive = true;
            decisionCount = 0;
            totalReward = 0;
            previousMakespan = 0;
            IsWaitingForAction = false;
            startTime = Time.time;

            SimLogger.Low($"[Orchestrator] Episode started: {currentConfig.JobCount} jobs, " +
                          $"{layoutManager.MachineCount} machines, " +
                          $"stochastic={StochasticEventManager.Instance?.IsActive}");
        }

        public void StopEpisode()
        {
            episodeActive = false;
            IsWaitingForAction = false;
            IsFactoryReady = false;
            layoutManager.ClearFloor();
            Jobs.Cleanup();
            agvPool.ClearFleet();
        }

        private void Update()
        {
            if (!episodeActive) return;

            _failures.SetSimTime(SimTime);
            _flags.SetSimTime(SimTime);

            _failures.HarvestFailureFlags();
            _flags.HarvestMachineFlags();
            _flags.HarvestAGVFlags();
            _flags.HarvestAlmostDoneFlags(PreDispatchLeadTime);
            _flags.AssignAGVs();

            if (!IsWaitingForAction)
            {
                var req = _decisions.FindNextDecision();
                if (req != null) // Standard null check for reference types
                {
                    CurrentDecision = req; // No .Value needed
                    IsWaitingForAction = true;
                    OnDecisionRequired?.Invoke(CurrentDecision);
                }
            }

            if (Jobs.AreAllExited())
                FinaliseEpisode();
        }

        public StepResult Step(int actionIndex)
        {
            IsWaitingForAction = false;

            if (CurrentDecision.Type == DecisionType.Routing)
                ExecuteRoutingDecision(actionIndex);
            else if (CurrentDecision.Type == DecisionType.Dispatch)
                ExecuteDispatchDecision(actionIndex);

            float reward = CalculateReward();
            totalReward += reward;

            return new StepResult { Reward = reward, Done = false, CurrentMakespan = SimTime };
        }

        private void ExecuteRoutingDecision(int actionIndex)
        {
            int chosenMachineId = DispatchingEngine.SelectMachine(actionIndex, CurrentDecision);
            JobData job = Jobs.Get(CurrentDecision.JobId);
            if (job == null) return;

            job.TargetMachineId = chosenMachineId;
            job.State = JobState.WaitingForPickup;
            job.StateEntryTime = SimTime;

            if (job.PreDispatchedAgvId >= 0)
            {
                AGVController preAgv = agvPool.GetPreDispatchedAGV(job.JobId);
                if (preAgv != null)
                {
                    PhysicalMachine targetMachine = layoutManager.GetMachine(chosenMachineId);
                    Vector3 dropoffPos = targetMachine != null
                        ? targetMachine.GetDropoffPosition() : layoutManager.OutgoingBeltPosition;
                    preAgv.FinalizePreDispatch(job.JobId, dropoffPos, targetMachine, job.Visual);
                    job.AssignedAgvId = preAgv.AgvId;
                    job.PreDispatchedAgvId = -1;
                    return;
                }
                job.PreDispatchedAgvId = -1;
            }
        }

        private void ExecuteDispatchDecision(int actionIndex)
        {
            int machineId = CurrentDecision.MachineId;
            int chosenJobId = DispatchingEngine.SelectJob(actionIndex, machineId, Jobs, SimTime);

            JobData job = Jobs.Get(chosenJobId);
            if (job == null || job.State != JobState.Queued || job.LocationMachineId != machineId) return;

            float duration = job.GetProcessingTime(machineId);
            job.State = JobState.Processing;
            job.TotalWaitTime += (SimTime - job.StateEntryTime);
            job.StateEntryTime = SimTime;

            PhysicalMachine machine = layoutManager.GetMachine(machineId);
            machine.StartJob(chosenJobId, duration, job.Visual);

            _machineProcessingStartTime[machineId] = SimTime;

            _flags.RefreshMachineLabels(machineId);
            LastAppliedRule = DispatchingEngine.RuleForIndex(actionIndex).ToString();
        }

        private void FinaliseEpisode()
        {
            episodeActive = false;

            var telemetry = EpisodeTelemetryChannel.Instance;
            if (telemetry != null)
            {
                telemetry.RecordEpisodeResult(
                    makespan: SimTime,
                    jobCount: currentConfig.JobCount,
                    machineCount: layoutManager.MachineCount,
                    totalOps: Jobs.AllJobs.Sum(j => j.TotalOperations),
                    decisions: decisionCount,
                    totalReward: totalReward,
                    ruleName: LastAppliedRule,
                    stochasticTag: currentConfig.Stochastic?.Tag ?? "none"
                );
                telemetry.Flush();
            }

            EpisodeRecord record = _tracker.Build(
                config: currentConfig,
                simTime: SimTime,
                ruleName: LastAppliedRule,
                completedJobs: Jobs.CountInState(JobState.Exited),
                totalOps: Jobs.AllJobs.Sum(j => j.TotalOperations),
                decisionPoints: decisionCount,
                totalReward: totalReward,
                agvCount: agvPool.AllAGVs.Count,
                machines: layoutManager.Machines,
                averageTimeScale: Time.timeScale
            );
            // ── Collect AGV performance records ──────────────────────────────────────
            foreach (var agv in agvPool.AllAGVs)
                record.AGVRecords.Add(agv.GetRecord(record.Makespan));

            // ── Collect segment congestion records ───────────────────────────────────
            // Skip parking alcove (Capacity=64) — it's intentionally unconstrained and
            // would inflate the zone count without diagnostic value.
            foreach (TrafficZone zone in trafficZoneManager.Zones)
            {
                if (zone.Name == "Parking_Alcove") continue;
                record.SegmentRecords.Add(new SegmentRecord
                {
                    ZoneId = zone.ZoneId,
                    ZoneName = zone.Name,
                    AisleType = zone.AisleType.ToString(),
                    FlowDirection = zone.Flow.ToString(),
                    TraversalCount = zone.TraversalCount,
                    BlockEvents = zone.BlockEvents,
                    TotalBlockTime = zone.TotalBlockTime,
                });
            }


            if (!Academy.Instance.IsCommunicatorOn)
            {
                SimLogger.Low($"[Orchestrator] Logging Academy Epiosde");
                //TODO: uncomment
                //ResultsLogger.LogAll(record);
            }


            if (record.MachineFailureCount > 0)
            {
                float theoreticalMeanTtf = currentConfig.Stochastic != null
                    ? EpisodeTracker.TheoreticalMeanTTF(currentConfig.Stochastic.WeibullLambda)
                    : 0f;
                SimLogger.Low($"[StochasticSummary] Failures={record.MachineFailureCount} " +
                              $"TotalRepairTime={record.MachineRepairTime:F1}s " +
                              $"MeanTTF_theory={theoreticalMeanTtf:F1}s");
            }

            OnEpisodeFinished?.Invoke(record);
        }

        private float CalculateReward()
        {
            float current = (float)SimTime;
            float delta = current - (float)previousMakespan;
            previousMakespan = current;
            int totalOps = Jobs.AllJobs.Sum(j => j.TotalOperations);
            return -delta / (Mathf.Max(totalOps, 1) * Time.timeScale);
        }

        public float GetFractionMachinesFailed()
        {
            int total = layoutManager.MachineCount;
            if (total == 0) return 0f;
            return (float)layoutManager.Machines.Count(m => m.HealthState == MachineHealthState.Failed) / total;
        }

        public float GetFractionMachinesRepairing()
        {
            int total = layoutManager.MachineCount;
            if (total == 0) return 0f;
            return (float)layoutManager.Machines.Count(m => m.HealthState == MachineHealthState.Repairing) / total;
        }

        public float GetMeanNormalisedRepairTime()
        {
            var repairing = layoutManager.Machines
                .Where(m => m.HealthState == MachineHealthState.Repairing && m.SampledRepairDuration > 0f)
                .ToList();

            if (repairing.Count == 0) return 0f;
            return repairing.Sum(m => m.RemainingRepairTime / m.SampledRepairDuration) / repairing.Count;
        }
    }
}