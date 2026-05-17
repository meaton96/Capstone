// using System;
// using System.Linq;
// using UnityEngine;
// using UnityEngine.Events;
// using System.Collections.Generic;
// using Assets.Scripts.Simulation.Logging;
// using Assets.Scripts.Simulation.Machines;
// using Assets.Scripts.Simulation.AGV;
// using Assets.Scripts.Simulation.FactoryLayout;
// using Assets.Scripts.Simulation.Jobs;
// using Assets.Scripts.Simulation.Stochastic;
// using Assets.Scripts.Simulation.Types;
// using Assets.Scripts.Simulation.Channels;
// using Unity.MLAgents;

// namespace Assets.Scripts.Simulation
// {
//     /// @brief The central orchestrator responsible for driving the factory simulation.
//     ///
//     /// @details SimulationBridge implements a strictly centralized state machine.
//     /// In a single @c Update tick, it harvests status flags from physical components
//     /// (Machines and AGVs), manages job transitions, resolves AGV assignments,
//     /// and interfaces with the @c SchedulingAgent to resolve scheduling conflicts.
//     /// No other component is permitted to mutate @c JobData state.
//     ///
//     /// Tracking refactor (EpisodeTracker):
//     ///   All per-episode and per-machine statistics are now accumulated by
//     ///   @c _tracker (an EpisodeTracker). FinaliseEpisode calls tracker.Build()
//     ///   to produce a single EpisodeRecord that flows to ResultsLogger and
//     ///   the OnEpisodeFinished event. Adding a new stochastic event only
//     ///   requires changes to EpisodeTracker + EpisodeRecord + ResultsLogger.
//     public class SimulationBridge : MonoBehaviour
//     {
//         public static SimulationBridge Instance;

//         [Header("Scene References")]
//         [SerializeField] private FactoryLayoutManager layoutManager;
//         [SerializeField] private TrafficZoneManager trafficZoneManager;
//         [SerializeField] private AGVPool agvPool;
//         [SerializeField] private SchedulingAgent agent;
//         public JobStore Jobs;

//         [Header("Episode Configuration")]
//         public int PreDispatchLeadTime = 15;
//         public bool AutoStartOnPlay = false;

//         private FJSSPJobDefinition[] prebuiltJobs;

//         private FJSSPConfig currentConfig;
//         private Dictionary<MachineType, List<int>> cachedMachinesByType;
//         public Dictionary<MachineType, List<int>> CachedMachinesByType => cachedMachinesByType;

//         private bool episodeActive;
//         private int decisionCount;
//         public int DecisionCount => decisionCount;
//         private double totalReward;
//         private double previousMakespan;
//         private float startTime;

//         public bool IsEpisodeActive => episodeActive;
//         public bool IsFactoryReady { get; set; }
//         public double SimTime => Time.time - startTime;
//         public FJSSPConfig CurrentConfig => currentConfig;

//         public DecisionRequest CurrentDecision { get; private set; }
//         public bool IsWaitingForAction { get; private set; }
//         public string LastAppliedRule { get; private set; } = "Waiting...";

//         // ── EpisodeTracker — replaces all scattered _episode* and _machine* dicts ──
//         // Single source of truth for all per-episode and per-machine statistics.
//         // Add new stochastic event tracking to EpisodeTracker, not here.
//         private readonly EpisodeTracker _tracker = new();

//         // Processing start timestamp — transient bridge-local timing helper.
//         // Not stats; just records when a machine began its current operation so
//         // HarvestMachineFlags can compute elapsed SimTime when FinishedFlag fires.
//         private readonly Dictionary<int, double> _machineProcessingStartTime = new();

//         [Header("Events")]
//         public UnityEvent<DecisionRequest> OnDecisionRequired;
//         public UnityEvent<StepResult> OnStepCompleted;
//         public UnityEvent<EpisodeRecord> OnEpisodeFinished;   // was UnityEvent<EpisodeResult>
//         public UnityEvent OnFactorySpawned;

//         private static readonly DispatchingRule[] ActionToRule = new DispatchingRule[]
//         {
//             DispatchingRule.SPT_SMPT,
//             DispatchingRule.SPT_SRWT,
//             DispatchingRule.LPT_MMUR,
//             DispatchingRule.LPT_SMPT,
//             DispatchingRule.SRT_SRWT,
//             DispatchingRule.SRT_SMPT,
//             DispatchingRule.LRT_MMUR,
//             DispatchingRule.SDT_SRWT
//         };

//         public static int ActionCount => ActionToRule.Length;
//         public int GetRuleIndex(DispatchingRule rule) => Array.IndexOf(ActionToRule, rule);

//         private void Awake()
//         {
//             if (Instance != null) { Destroy(this); return; }
//             Instance = this;
//         }

//         private void Start()
//         {
//             if (AutoStartOnPlay && agent != null)
//                 agent.IsArmed = true;
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  Config / Factory lifecycle
//         // ─────────────────────────────────────────────────────────────────────

//         public void LoadConfig(FJSSPConfig config)
//         {
//             currentConfig = config;
//             IsFactoryReady = false;
//             StochasticEventManager.Instance?.Initialize(config);
//         }

//         public void LoadPrebuiltJobs(FJSSPJobDefinition[] jobs)
//         {
//             prebuiltJobs = jobs;
//         }

//         public void SpawnFactory()
//         {
//             if (currentConfig == null) return;
//             if (IsFactoryReady || episodeActive) StopEpisode();

//             UnityEngine.Random.InitState(currentConfig.Seed);
//             cachedMachinesByType = layoutManager.BuildFloor(currentConfig);
//             trafficZoneManager.BuildZoneGraph();
//             agvPool.InitializeFleet(currentConfig.AGVCount);

//             IsFactoryReady = true;
//             OnFactorySpawned?.Invoke();
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  Episode start
//         // ─────────────────────────────────────────────────────────────────────

//         public void StartEpisode()
//         {
//             // ── Consume Python-sent config if available ───────────────────────
//             var pythonConfig = EpisodeConfigChannel.Instance?.ConsumeConfig();
//             if (pythonConfig != null)
//             {
//                 currentConfig = pythonConfig;
//                 IsFactoryReady = false;
//                 SimLogger.Low($"[Bridge] Applied Python config: {currentConfig.Name}");
//             }

//             if (currentConfig == null)
//                 currentConfig = BuildDefaultStochasticConfig();

//             if (!IsFactoryReady)
//                 SpawnFactory();

//             agent.SetHeuristicRule(currentConfig.dispatchingRule);

//             // ── Jobs ──────────────────────────────────────────────────────────
//             FJSSPJobDefinition[] jobDefs;
//             if (prebuiltJobs != null)
//             {
//                 jobDefs = prebuiltJobs;
//                 prebuiltJobs = null;
//                 SimLogger.Low("[Orchestrator] Using prebuilt benchmark jobs");
//             }
//             else
//             {
//                 jobDefs = FJSSPJobGenerator.Generate(currentConfig, cachedMachinesByType);
//             }

//             Jobs.Initialize(jobDefs, spawnVisuals: true);

//             // ── Stochastic init ───────────────────────────────────────────────
//             if (currentConfig.Stochastic != null && currentConfig.Stochastic.AnyEnabled)
//             {
//                 StochasticEventManager.Instance?.Initialize(currentConfig);
//                 foreach (var machine in layoutManager.Machines)
//                     machine.InitializeStochastic();
//             }

//             // ── Tracker reset — replaces all dict.Clear() calls ───────────────
//             _tracker.Reset();
//             _machineProcessingStartTime.Clear();

//             // ── Episode state ─────────────────────────────────────────────────
//             episodeActive = true;
//             decisionCount = 0;
//             totalReward = 0;
//             previousMakespan = 0;
//             IsWaitingForAction = false;
//             startTime = Time.time;

//             SimLogger.Low($"[Orchestrator] Episode started: {currentConfig.JobCount} jobs, " +
//                           $"{layoutManager.MachineCount} machines, " +
//                           $"stochastic={StochasticEventManager.Instance?.IsActive}");
//         }

//         public void StopEpisode()
//         {
//             episodeActive = false;
//             IsWaitingForAction = false;
//             IsFactoryReady = false;
//             layoutManager.ClearFloor();
//             Jobs.Cleanup();
//             agvPool.ClearFleet();
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  Update loop
//         // ─────────────────────────────────────────────────────────────────────

//         private void Update()
//         {
//             if (!episodeActive) return;

//             HarvestMachineFailureFlags();
//             HarvestMachineFlags();
//             HarvestAGVFlags();
//             HarvestAlmostDoneFlags();
//             AssignAGVs();

//             if (!IsWaitingForAction)
//                 FindNextDecision();

//             if (Jobs.AreAllExited())
//                 FinaliseEpisode();
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  Phase 2: Machine failure harvesting
//         // ─────────────────────────────────────────────────────────────────────

//         private void HarvestMachineFailureFlags()
//         {
//             if (StochasticEventManager.Instance == null ||
//                 !StochasticEventManager.Instance.MachineFailuresEnabled)
//                 return;

//             foreach (var machine in layoutManager.Machines)
//             {
//                 if (machine.FailedFlag)
//                     HandleMachineFailure(machine);
//                 else if (machine.RepairCompleteFlag)
//                     HandleMachineRepairComplete(machine);
//             }
//         }

//         private void HandleMachineFailure(PhysicalMachine machine)
//         {
//             int machineId = machine.MachineId;
//             SimLogger.Low($"[Orchestrator] Machine {machineId} FAILED. " +
//                           $"RepairTime={machine.SampledRepairDuration:F1}s");

//             // Discard any partial processing interval — operation restarts in full.
//             _machineProcessingStartTime.Remove(machineId);

//             // Delegate all stat accumulation to tracker.
//             _tracker.RecordMachineFailure(machineId, machine.SampledRepairDuration, SimTime);

//             // 1. Return the actively processing job to NeedsRouting.
//             if (machine.ActiveJobId >= 0)
//             {
//                 JobData processingJob = Jobs.Get(machine.ActiveJobId);
//                 if (processingJob != null && processingJob.State == JobState.Processing)
//                 {
//                     processingJob.State = JobState.NeedsRouting;
//                     processingJob.LocationMachineId = machineId;
//                     processingJob.StateEntryTime = SimTime;
//                     SimLogger.Low($"[Orchestrator] Job {processingJob.JobId} returned to " +
//                                   $"NeedsRouting (was Processing on failed machine {machineId}).");
//                 }
//             }

//             // 2. Return jobs Queued at this machine to NeedsRouting.
//             foreach (var job in Jobs.AllJobs)
//             {
//                 if (job.LocationMachineId == machineId && job.State == JobState.Queued)
//                 {
//                     job.State = JobState.NeedsRouting;
//                     job.StateEntryTime = SimTime;
//                     SimLogger.Low($"[Orchestrator] Queued job {job.JobId} re-routed " +
//                                   $"from failed machine {machineId}.");
//                 }
//             }

//             // 3. Re-route AGVs carrying jobs destined for this machine.
//             foreach (var agv in agvPool.AllAGVs)
//             {
//                 int agvJobId = agv.CurrentJobId;
//                 if (agvJobId < 0) continue;

//                 JobData transitJob = Jobs.Get(agvJobId);
//                 if (transitJob == null) continue;
//                 if (transitJob.State != JobState.InTransit) continue;
//                 if (transitJob.TargetMachineId != machineId) continue;

//                 transitJob.State = JobState.NeedsRouting;
//                 transitJob.TargetMachineId = -1;
//                 transitJob.AssignedAgvId = -1;
//                 transitJob.StateEntryTime = SimTime;

//                 SimLogger.Low($"[Orchestrator] AGV {agv.AgvId} carrying job {agvJobId} " +
//                               $"re-routed: destination machine {machineId} has failed.");
//             }

//             // 4. Cancel any pre-dispatched AGV headed for this machine.
//             foreach (var job in Jobs.AllJobs)
//             {
//                 if (job.PreDispatchedAgvId < 0) continue;
//                 if (job.TargetMachineId != machineId) continue;

//                 SimLogger.Low($"[Orchestrator] Pre-dispatch for job {job.JobId} to " +
//                               $"machine {machineId} cancelled.");
//                 job.PreDispatchedAgvId = -1;
//             }

//             // 5. Transition machine to Repairing.
//             machine.AcknowledgeFailure();
//             RefreshMachineLabels(machineId);

//             // 6. Invalidate any pending dispatch decision for this machine.
//             if (IsWaitingForAction &&
//                 CurrentDecision.Type == DecisionType.Dispatch &&
//                 CurrentDecision.MachineId == machineId)
//             {
//                 IsWaitingForAction = false;
//                 SimLogger.Low($"[Orchestrator] Pending dispatch decision for machine " +
//                               $"{machineId} invalidated (machine failed).");
//             }
//         }

//         private void HandleMachineRepairComplete(PhysicalMachine machine)
//         {
//             SimLogger.Low($"[Orchestrator] Machine {machine.MachineId} repair complete — " +
//                           $"returning to OPERATIONAL.");

//             machine.AcknowledgeRepairComplete();

//             // Delegate downtime interval close to tracker.
//             _tracker.RecordRepairComplete(machine.MachineId, SimTime);

//             RefreshMachineLabels(machine.MachineId);
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  Observation helpers (Global Scalars — Phase 2)
//         // ─────────────────────────────────────────────────────────────────────

//         public float GetFractionMachinesFailed()
//         {
//             int total = layoutManager.MachineCount;
//             if (total == 0) return 0f;
//             return (float)layoutManager.Machines.Count(m => m.HealthState == MachineHealthState.Failed) / total;
//         }

//         public float GetFractionMachinesRepairing()
//         {
//             int total = layoutManager.MachineCount;
//             if (total == 0) return 0f;
//             return (float)layoutManager.Machines.Count(m => m.HealthState == MachineHealthState.Repairing) / total;
//         }

//         public float GetMeanNormalisedRepairTime()
//         {
//             var repairing = layoutManager.Machines
//                 .Where(m => m.HealthState == MachineHealthState.Repairing && m.SampledRepairDuration > 0f)
//                 .ToList();

//             if (repairing.Count == 0) return 0f;
//             return repairing.Sum(m => m.RemainingRepairTime / m.SampledRepairDuration) / repairing.Count;
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  HarvestMachineFlags
//         // ─────────────────────────────────────────────────────────────────────

//         private void HarvestMachineFlags()
//         {
//             foreach (var machine in layoutManager.Machines)
//             {
//                 if (!machine.FinishedFlag) continue;

//                 int jobId = machine.ActiveJobId;
//                 int mid = machine.MachineId;
//                 machine.ClearFinished();

//                 // Accumulate processing time via tracker.
//                 if (_machineProcessingStartTime.TryGetValue(mid, out double procStart))
//                 {
//                     _tracker.AddProcessingTime(mid, SimTime - procStart);
//                     _machineProcessingStartTime.Remove(mid);
//                 }
//                 _tracker.RecordOperationComplete(mid);

//                 JobData job = Jobs.Get(jobId);
//                 if (job == null) continue;

//                 job.CompletedOps++;
//                 if (job.CurrentOpIndex < job.TotalOperations)
//                     job.CurrentOpIndex++;

//                 machine.PlaceOnOutgoing(jobId, job.Visual);
//                 RefreshMachineLabels(mid);

//                 if (job.IsLastOperation)
//                 {
//                     job.State = JobState.WaitingForPickup;
//                     job.TargetMachineId = -1;
//                     job.LocationMachineId = mid;
//                     job.StateEntryTime = SimTime;

//                     if (job.PreDispatchedAgvId >= 0)
//                     {
//                         AGVController preAgv = agvPool.GetPreDispatchedAGV(job.JobId);
//                         if (preAgv != null)
//                         {
//                             preAgv.FinalizePreDispatch(job.JobId, layoutManager.OutgoingBeltPosition, null, job.Visual);
//                             job.AssignedAgvId = preAgv.AgvId;
//                         }
//                         job.PreDispatchedAgvId = -1;
//                     }
//                 }
//                 else
//                 {
//                     job.State = JobState.NeedsRouting;
//                     job.LocationMachineId = mid;
//                     job.StateEntryTime = SimTime;
//                 }
//             }
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  HarvestAlmostDoneFlags
//         // ─────────────────────────────────────────────────────────────────────

//         private void HarvestAlmostDoneFlags()
//         {
//             foreach (var machine in layoutManager.Machines)
//             {
//                 if (!machine.AlmostDoneFlag) continue;

//                 int jobId = machine.AlmostDoneJobId;
//                 machine.ClearAlmostDone();

//                 JobData job = Jobs.Get(jobId);
//                 if (job == null || job.State != JobState.Processing || job.PreDispatchedAgvId >= 0) continue;
//                 if (job.CompletedOps == job.TotalOperations - 1) continue;

//                 AGVController agv = agvPool.GetAvailableAGV();
//                 if (agv == null) continue;

//                 agv.PreDispatch(jobId, machine.GetPickupPosition(), machine);
//                 job.PreDispatchedAgvId = agv.AgvId;
//             }
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  HarvestAGVFlags
//         // ─────────────────────────────────────────────────────────────────────

//         private void HarvestAGVFlags()
//         {
//             foreach (var agv in agvPool.AllAGVs)
//             {
//                 if (agv.PickedUpFlag)
//                 {
//                     JobData job = Jobs.Get(agv.CurrentJobId);
//                     if (job != null && job.State == JobState.WaitingForPickup)
//                     {
//                         job.State = JobState.InTransit;
//                         job.StateEntryTime = SimTime;
//                     }
//                 }

//                 if (agv.DeliveredFlag)
//                 {
//                     int jobId = agv.DeliveredJobId;
//                     int machineId = agv.DeliveredMachineId;
//                     JobData job = Jobs.Get(jobId);

//                     if (job != null)
//                     {
//                         if (machineId < 0)
//                         {
//                             job.State = JobState.Exited;
//                             job.LocationMachineId = -1;
//                             job.StateEntryTime = SimTime;
//                             if (job.Visual != null) job.Visual.gameObject.SetActive(false);
//                         }
//                         else
//                         {
//                             job.State = JobState.Queued;
//                             job.LocationMachineId = machineId;
//                             job.StateEntryTime = SimTime;

//                             PhysicalMachine targetMachine = layoutManager.GetMachine(machineId);
//                             targetMachine.PlaceOnIncoming(jobId, job.Visual);
//                             RefreshMachineLabels(machineId);
//                         }
//                         job.TotalTransitTime += (SimTime - job.StateEntryTime);
//                         job.AssignedAgvId = -1;
//                     }
//                 }

//                 if (agv.PickedUpFlag || agv.DeliveredFlag)
//                     agv.ClearFlags();
//             }
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  AssignAGVs
//         // ─────────────────────────────────────────────────────────────────────

//         private void AssignAGVs()
//         {
//             var candidates = new List<JobData>();
//             foreach (var job in Jobs.AllJobs)
//             {
//                 if (job.State == JobState.WaitingForPickup
//                     && job.AssignedAgvId == -1
//                     && job.PreDispatchedAgvId < 0)
//                     candidates.Add(job);
//             }

//             foreach (var job in candidates)
//             {
//                 AGVController agv = agvPool.GetAvailableAGV();
//                 if (agv == null) break;

//                 PhysicalMachine src = job.LocationMachineId >= 0
//                     ? layoutManager.GetMachine(job.LocationMachineId) : null;
//                 Vector3 pickupPos = src != null
//                     ? src.GetPickupPosition() : layoutManager.IncomingBeltPosition;

//                 PhysicalMachine dst = job.TargetMachineId >= 0
//                     ? layoutManager.GetMachine(job.TargetMachineId) : null;
//                 Vector3 dropoffPos = dst != null
//                     ? dst.GetDropoffPosition() : layoutManager.OutgoingBeltPosition;

//                 job.AssignedAgvId = agv.AgvId;
//                 agv.Dispatch(job.JobId, pickupPos, dropoffPos, src, dst, job.Visual);
//                 agv.SetCarryVisual(job.Visual);
//             }
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  FindNextDecision
//         // ─────────────────────────────────────────────────────────────────────

//         private void FindNextDecision()
//         {
//             JobData routingJob = Jobs.GetNextNeedsRouting();
//             if (routingJob != null)
//             {
//                 var eligibleIds = new HashSet<int>(
//                     routingJob.EligibleMachinesPerOp[routingJob.CurrentOpIndex].Keys);

//                 bool anyAvailable = layoutManager.Machines
//                     .Any(m => eligibleIds.Contains(m.MachineId) && m.IsAvailableForWork);

//                 if (!anyAvailable)
//                 {
//                     SimLogger.Low($"[Orchestrator] Job {routingJob.JobId}: all eligible machines " +
//                                   $"are Failed/Repairing. Deferring routing decision.");
//                 }
//                 else
//                 {
//                     CurrentDecision = BuildRoutingDecision(routingJob);
//                     IsWaitingForAction = true;
//                     OnDecisionRequired?.Invoke(CurrentDecision);
//                     return;
//                 }
//             }

//             foreach (var machine in layoutManager.Machines)
//             {
//                 if (machine.IsIdle && machine.IsAvailableForWork && Jobs.HasDispatchableJob(machine.MachineId))
//                 {
//                     CurrentDecision = BuildDispatchDecision(machine.MachineId);
//                     IsWaitingForAction = true;
//                     OnDecisionRequired?.Invoke(CurrentDecision);
//                     return;
//                 }
//             }
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  Step / Execute
//         // ─────────────────────────────────────────────────────────────────────

//         public StepResult Step(int actionIndex)
//         {
//             IsWaitingForAction = false;

//             if (CurrentDecision.Type == DecisionType.Routing)
//                 ExecuteRoutingDecision(actionIndex);
//             else if (CurrentDecision.Type == DecisionType.Dispatch)
//                 ExecuteDispatchDecision(actionIndex);

//             float reward = CalculateReward();
//             totalReward += reward;

//             return new StepResult { Reward = reward, Done = false, CurrentMakespan = SimTime };
//         }

//         private void ExecuteRoutingDecision(int actionIndex)
//         {
//             int chosenMachineId = ApplyMachineSelectionRule(actionIndex, CurrentDecision);
//             JobData job = Jobs.Get(CurrentDecision.JobId);
//             if (job == null) return;

//             job.TargetMachineId = chosenMachineId;
//             job.State = JobState.WaitingForPickup;
//             job.StateEntryTime = SimTime;

//             if (job.PreDispatchedAgvId >= 0)
//             {
//                 AGVController preAgv = agvPool.GetPreDispatchedAGV(job.JobId);
//                 if (preAgv != null)
//                 {
//                     PhysicalMachine targetMachine = layoutManager.GetMachine(chosenMachineId);
//                     Vector3 dropoffPos = targetMachine != null
//                         ? targetMachine.GetDropoffPosition() : layoutManager.OutgoingBeltPosition;
//                     preAgv.FinalizePreDispatch(job.JobId, dropoffPos, targetMachine, job.Visual);
//                     job.AssignedAgvId = preAgv.AgvId;
//                     job.PreDispatchedAgvId = -1;
//                     return;
//                 }
//                 job.PreDispatchedAgvId = -1;
//             }
//         }

//         private void ExecuteDispatchDecision(int actionIndex)
//         {
//             int machineId = CurrentDecision.MachineId;
//             int chosenJobId = ApplyDispatchingRule(actionIndex, machineId);

//             JobData job = Jobs.Get(chosenJobId);
//             if (job == null || job.State != JobState.Queued || job.LocationMachineId != machineId) return;

//             float duration = job.GetProcessingTime(machineId);
//             job.State = JobState.Processing;
//             job.TotalWaitTime += (SimTime - job.StateEntryTime);
//             job.StateEntryTime = SimTime;

//             PhysicalMachine machine = layoutManager.GetMachine(machineId);
//             machine.StartJob(chosenJobId, duration, job.Visual);

//             // Record processing start for elapsed-time accumulation in HarvestMachineFlags.
//             _machineProcessingStartTime[machineId] = SimTime;

//             RefreshMachineLabels(machineId);
//             LastAppliedRule = ActionToRule[actionIndex].ToString();
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  Decision builders
//         // ─────────────────────────────────────────────────────────────────────

//         private DecisionRequest BuildRoutingDecision(JobData job)
//         {
//             var eligibleIds = new HashSet<int>(
//                 job.EligibleMachinesPerOp[job.CurrentOpIndex].Keys);

//             var candidates = layoutManager.Machines
//                 .Where(m => eligibleIds.Contains(m.MachineId) && m.IsAvailableForWork)
//                 .Select(m => m.MachineId)
//                 .ToList();

//             return new DecisionRequest
//             {
//                 Type = DecisionType.Routing,
//                 SimTime = SimTime,
//                 DecisionIndex = decisionCount++,
//                 TotalJobs = Jobs.JobCount,
//                 CompletedJobs = Jobs.CountInState(JobState.Exited),
//                 JobId = job.JobId,
//                 SourceMachineId = job.LocationMachineId,
//                 RequiredType = job.NextRequiredType,
//                 CandidateMachineIds = candidates.ToArray(),
//                 CandidateQueueLengths = candidates.Select(id => Jobs.GetMachineLoad(id)).ToArray(),
//                 CandidateJobTimes = candidates.Select(id => job.GetProcessingTime(id)).ToArray(),
//             };
//         }

//         private DecisionRequest BuildDispatchDecision(int machineId)
//         {
//             List<int> queue = Jobs.GetDispatchableJobs(machineId);
//             return new DecisionRequest
//             {
//                 Type = DecisionType.Dispatch,
//                 MachineId = machineId,
//                 SimTime = SimTime,
//                 DecisionIndex = decisionCount++,
//                 TotalJobs = Jobs.JobCount,
//                 CompletedJobs = Jobs.CountInState(JobState.Exited),
//                 QueuedJobIds = queue.ToArray(),
//                 QueuedDurations = queue.Select(id => (double)Jobs.GetProcessingTime(id, machineId)).ToArray(),
//             };
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  FinaliseEpisode — builds EpisodeRecord, logs, fires event
//         // ─────────────────────────────────────────────────────────────────────

//         private void FinaliseEpisode()
//         {
//             episodeActive = false;

//             // ── Telemetry channel (ML-Agents training mode only) ──────────────
//             var telemetry = EpisodeTelemetryChannel.Instance;
//             if (telemetry != null)
//             {
//                 telemetry.RecordEpisodeResult(
//                     makespan: SimTime,
//                     jobCount: currentConfig.JobCount,
//                     machineCount: layoutManager.MachineCount,
//                     totalOps: Jobs.AllJobs.Sum(j => j.TotalOperations),
//                     decisions: decisionCount,
//                     totalReward: totalReward,
//                     ruleName: LastAppliedRule,
//                     stochasticTag: currentConfig.Stochastic?.Tag ?? "none"
//                 );
//                 telemetry.Flush();
//             }

//             // ── Build EpisodeRecord (tracker closes open downtime intervals) ──
//             EpisodeRecord record = _tracker.Build(
//                 config: currentConfig,
//                 simTime: SimTime,
//                 ruleName: LastAppliedRule,
//                 completedJobs: Jobs.CountInState(JobState.Exited),
//                 totalOps: Jobs.AllJobs.Sum(j => j.TotalOperations),
//                 decisionPoints: decisionCount,
//                 totalReward: totalReward,
//                 agvCount: agvPool.AllAGVs.Count,
//                 machines: layoutManager.Machines,
//                 averageTimeScale: Time.timeScale
//             );

//             // ── Log to CSV (both episode and machine utilization in one call) ──
//             // Skip during ML-Agents training — headless batch runner handles logging.
//             if (!Academy.Instance.IsCommunicatorOn)
//                 ResultsLogger.LogAll(record);

//             // ── Debug summary ─────────────────────────────────────────────────
//             if (record.MachineFailureCount > 0)
//             {
//                 float theoreticalMeanTtf = currentConfig.Stochastic != null
//                     ? EpisodeTracker.TheoreticalMeanTTF(currentConfig.Stochastic.WeibullLambda)
//                     : 0f;
//                 SimLogger.Low($"[StochasticSummary] Failures={record.MachineFailureCount} " +
//                               $"TotalRepairTime={record.MachineRepairTime:F1}s " +
//                               $"MeanTTF_theory={theoreticalMeanTtf:F1}s");
//             }

//             // ── Fire event — HeadlessBatchRunner listens here ─────────────────
//             OnEpisodeFinished?.Invoke(record);
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  Reward
//         // ─────────────────────────────────────────────────────────────────────

//         private float CalculateReward()
//         {
//             float current = (float)SimTime;
//             float delta = current - (float)previousMakespan;
//             previousMakespan = current;
//             int totalOps = Jobs.AllJobs.Sum(j => j.TotalOperations);
//             return -delta / (Mathf.Max(totalOps, 1) * Time.timeScale);
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  Dispatching rule helpers
//         // ─────────────────────────────────────────────────────────────────────

//         private int ApplyDispatchingRule(int actionIndex, int machineId)
//         {
//             DispatchingRule rule = ActionToRule[actionIndex];
//             List<int> queue = Jobs.GetDispatchableJobs(machineId);

//             if (queue.Count == 0) return -1;
//             if (queue.Count == 1) return queue[0];

//             return rule switch
//             {
//                 DispatchingRule.SPT_SMPT or DispatchingRule.SPT_SRWT
//                     => ArgMin(queue, id => Jobs.Get(id).GetProcessingTime(machineId)),
//                 DispatchingRule.LPT_MMUR or DispatchingRule.LPT_SMPT
//                     => ArgMax(queue, id => Jobs.Get(id).GetProcessingTime(machineId)),
//                 DispatchingRule.SRT_SRWT or DispatchingRule.SRT_SMPT
//                     => ArgMin(queue, id => GetRemainingWork(id)),
//                 DispatchingRule.LRT_MMUR
//                     => ArgMax(queue, id => GetRemainingWork(id)),
//                 DispatchingRule.SDT_SRWT
//                     => ArgMin(queue, id => (float)(SimTime - Jobs.Get(id).ArrivalTime)),
//                 _ => queue[0]
//             };
//         }

//         private int ApplyMachineSelectionRule(int actionIndex, DecisionRequest req)
//         {
//             DispatchingRule rule = ActionToRule[actionIndex];
//             int[] candidates = req.CandidateMachineIds;
//             if (candidates.Length == 1) return candidates[0];

//             return rule switch
//             {
//                 DispatchingRule.SPT_SMPT or DispatchingRule.LPT_SMPT or DispatchingRule.SRT_SMPT
//                     => candidates[ArgMinIdx(req.CandidateJobTimes)],
//                 DispatchingRule.SPT_SRWT or DispatchingRule.SRT_SRWT or DispatchingRule.SDT_SRWT
//                     => candidates[ArgMinIdx(req.CandidateQueueLengths)],
//                 DispatchingRule.LPT_MMUR or DispatchingRule.LRT_MMUR
//                     => candidates[ArgMaxIdx(req.CandidateQueueLengths)],
//                 _ => candidates[0]
//             };
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  Utilities
//         // ─────────────────────────────────────────────────────────────────────

//         private void RefreshMachineLabels(int machineId)
//         {
//             PhysicalMachine machine = layoutManager.GetMachine(machineId);
//             if (machine == null) return;
//             int inCount = Jobs.GetDispatchableJobs(machineId).Count;
//             int outCount = Jobs.AllJobs.Count(j =>
//                 j.LocationMachineId == machineId && j.State == JobState.WaitingForPickup);
//             machine.RefreshQueueLabels(inCount, outCount);
//         }

//         private float GetRemainingWork(int jobId)
//         {
//             JobData j = Jobs.Get(jobId);
//             if (j == null) return 0f;
//             float total = 0f;
//             for (int o = j.CurrentOpIndex; o < j.TotalOperations; o++)
//                 total += j.EligibleMachinesPerOp[o].Values.Min();
//             return total;
//         }

//         private int ArgMin(List<int> ids, Func<int, float> score)
//         {
//             int best = ids[0]; float bestS = float.MaxValue;
//             foreach (int id in ids) { float s = score(id); if (s < bestS) { bestS = s; best = id; } }
//             return best;
//         }

//         private int ArgMax(List<int> ids, Func<int, float> score)
//         {
//             int best = ids[0]; float bestS = float.MinValue;
//             foreach (int id in ids) { float s = score(id); if (s > bestS) { bestS = s; best = id; } }
//             return best;
//         }

//         private int ArgMinIdx(float[] v)
//         {
//             int b = 0;
//             for (int i = 1; i < v.Length; i++) if (v[i] < v[b]) b = i;
//             return b;
//         }

//         private int ArgMaxIdx(float[] v)
//         {
//             int b = 0;
//             for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
//             return b;
//         }

//         // ─────────────────────────────────────────────────────────────────────
//         //  Default configs (editor / Python training fallback)
//         // ─────────────────────────────────────────────────────────────────────

//         private FJSSPConfig BuildDefaultStochasticConfig()
//         {
//             MachineType[] types = (MachineType[])Enum.GetValues(typeof(MachineType));
//             var layout = new MachineType[types.Length];
//             for (int i = 0; i < types.Length; i++) layout[i] = types[i];

//             return new FJSSPConfig
//             {
//                 Seed = 42,
//                 JobCount = 5,
//                 MachinesPerType = 1,
//                 MachineTypeLayout = layout,
//                 MinProcTime = 1f,
//                 MaxProcTime = 30f,
//                 MinOpsPerJob = 2,
//                 MaxOpsPerJob = 4,
//                 MaxArrivalTime = 0f,
//                 AGVCount = 3,
//                 ProcTimeParams = new Dictionary<MachineType, (float mu, float sigma)>
//                 {
//                     { MachineType.Mill,     (9f,  1f)  },
//                     { MachineType.Lathe,    (7f,  1f)  },
//                     { MachineType.Weld,     (15f, 2f)  },
//                     { MachineType.Inspect,  (6f,  1f)  },
//                     { MachineType.Assemble, (24f, 4f)  },
//                 },
//                 dispatchingRule = DispatchingRule.SRT_SRWT,
//                 MachineFlexibilityProbability = 0f,
//                 Stochastic = new StochasticConfig
//                 {
//                     MachineFailuresEnabled = true,
//                     WeibullK = 1.5f,
//                     WeibullLambda = 2000f,
//                     RepairLogMu = 2.0f,
//                     RepairLogSigma = 0.5f,
//                     DynamicArrivalsEnabled = false,
//                 }
//             };
//         }

//         private FJSSPConfig BuildDefaultConfig()
//         {
//             MachineType[] types = (MachineType[])Enum.GetValues(typeof(MachineType));
//             var layout = new MachineType[types.Length];
//             for (int i = 0; i < types.Length; i++) layout[i] = types[i];

//             return new FJSSPConfig
//             {
//                 Seed = 42,
//                 JobCount = 15,
//                 MachinesPerType = 1,
//                 MachineTypeLayout = layout,
//                 MinProcTime = 1f,
//                 MaxProcTime = 30f,
//                 MinOpsPerJob = 2,
//                 MaxOpsPerJob = 4,
//                 MaxArrivalTime = 0f,
//                 AGVCount = 3,
//                 ProcTimeParams = new Dictionary<MachineType, (float mu, float sigma)>
//                 {
//                     { MachineType.Mill,     (9f,  1f)  },
//                     { MachineType.Lathe,    (7f,  1f)  },
//                     { MachineType.Weld,     (15f, 2f)  },
//                     { MachineType.Inspect,  (6f,  1f)  },
//                     { MachineType.Assemble, (24f, 4f)  },
//                 },
//                 dispatchingRule = DispatchingRule.SRT_SRWT,
//                 MachineFlexibilityProbability = 0f,
//             };
//         }
//     }
// }