using System;
using System.Linq;
using UnityEngine;
using UnityEngine.Events;
using System.Collections.Generic;
using Assets.Scripts.Logging;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.AGV;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Types;
using System.IO;
using Unity.MLAgents;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// THE single orchestrator. One Update() tick drives the entire simulation.
    /// 
    /// Nothing else mutates job state. PhysicalMachine sets flags. AGVController
    /// sets flags. This class reads those flags and makes ALL state transitions.
    /// 
    /// Decision flow:
    ///   1. Harvest flags from machines and AGVs
    ///   2. Assign AGVs to jobs waiting for pickup
    ///   3. If not waiting for agent: find next decision (routing or dispatch)
    ///   4. Agent responds via Step() → execute the decision
    /// </summary>
    public class SimulationBridge : MonoBehaviour
    {
        public static SimulationBridge Instance;

        // ─────────────────────────────────────────────────────────
        //  Scene References
        // ─────────────────────────────────────────────────────────

        [Header("Scene References")]
        [SerializeField] private FactoryLayoutManager layoutManager;
        [SerializeField] private TrafficZoneManager trafficZoneManager;
        [SerializeField] private AGVPool agvPool;
        [SerializeField] private SchedulingAgent agent;
        public JobStore Jobs;



        // ─────────────────────────────────────────────────────────
        //  Config & Episode State
        // ─────────────────────────────────────────────────────────

        [Header("Episode Configuration")]
        [SerializeField] private bool autoStartOnPlay = false;
        public bool AutoStartOnPlay => autoStartOnPlay;
        //[SerializeField] private LogLevel logLevel = LogLevel.Low;

        private FJSSPConfig currentConfig;
        private Dictionary<MachineType, List<int>> cachedMachinesByType;

        private bool episodeActive;
        private int decisionCount;
        public int DecisionCount => decisionCount;
        private double totalReward;
        private double previousMakespan;
        private float startTime;

        public bool IsEpisodeActive => episodeActive;
        public bool IsFactoryReady { get; private set; }
        public double SimTime => Time.time - startTime;
        public FJSSPConfig CurrentConfig => currentConfig;

        // ─────────────────────────────────────────────────────────
        //  Agent Interface
        // ─────────────────────────────────────────────────────────

        public DecisionRequest CurrentDecision { get; private set; }
        public bool IsWaitingForAction { get; private set; }
        public string LastAppliedRule { get; private set; } = "Waiting...";

        [Header("Events")]
        public UnityEvent<DecisionRequest> OnDecisionRequired;
        public UnityEvent<StepResult> OnStepCompleted;
        public UnityEvent<EpisodeResult> OnEpisodeFinished;
        public UnityEvent OnFactorySpawned;

        // ─────────────────────────────────────────────────────────
        //  Dispatching Rules
        // ─────────────────────────────────────────────────────────

        private static readonly DispatchingRule[] ActionToRule = new DispatchingRule[]
        {
            DispatchingRule.SPT_SMPT,
            DispatchingRule.SPT_SRWT,
            DispatchingRule.LPT_MMUR,
            DispatchingRule.LPT_SMPT,
            DispatchingRule.SRT_SRWT,
            DispatchingRule.SRT_SMPT,
            DispatchingRule.LRT_MMUR,
            DispatchingRule.SDT_SRWT
        };

        public static int ActionCount => ActionToRule.Length;
        public int GetRuleIndex(DispatchingRule rule) => Array.IndexOf(ActionToRule, rule);

        // ═════════════════════════════════════════════════════════
        //  UNITY LIFECYCLE
        // ═════════════════════════════════════════════════════════

        private void Awake()
        {
            if (Instance != null) { Destroy(this); return; }
            Instance = this;
            //SimLogger.ActiveLevel = logLevel;
        }

        private void Start()
        {
            if (autoStartOnPlay && agent != null)
                agent.IsArmed = true;
        }

        // ═════════════════════════════════════════════════════════
        //  EPISODE MANAGEMENT
        // ═════════════════════════════════════════════════════════

        public void LoadConfig(FJSSPConfig config)
        {
            currentConfig = config;
            IsFactoryReady = false;
        }

        public void SpawnFactory()
        {
            if (currentConfig == null) return;
            if (IsFactoryReady || episodeActive) StopEpisode();

            UnityEngine.Random.InitState(currentConfig.Seed);
            cachedMachinesByType = layoutManager.BuildFloor(currentConfig);
            trafficZoneManager.BuildZoneGraph();
            agvPool.InitializeFleet();

            IsFactoryReady = true;
            OnFactorySpawned?.Invoke();
        }

        public void StartEpisode()
        {
            if (currentConfig == null)
            {
                currentConfig = BuildDefaultConfig();
                // currentConfig = BuildTestConfig();
            }


            if (!IsFactoryReady)
            {
                SpawnFactory();
            }

            var jobDefs = FJSSPJobGenerator.Generate(currentConfig, cachedMachinesByType);
            Jobs.Initialize(jobDefs, spawnVisuals: true);

            episodeActive = true;
            decisionCount = 0;
            totalReward = 0;
            previousMakespan = 0;
            IsWaitingForAction = false;
            startTime = Time.time;

            // All jobs start in NeedsRouting — the tick loop will pick them up.
            SimLogger.Low($"[Orchestrator] Episode started: {currentConfig.JobCount} jobs, " +
                           $"{layoutManager.MachineCount} machines");
        }

        public void StopEpisode()
        {
            episodeActive = false;
            IsWaitingForAction = false;
            layoutManager.ClearFloor();
            Jobs.Cleanup();
            // agent?.EndEpisode();
        }

        // ═════════════════════════════════════════════════════════
        //  THE TICK — one pass per frame, deterministic order
        // ═════════════════════════════════════════════════════════

        private void Update()
        {
            if (!episodeActive) return;

            // ── Phase 1: Harvest machine completion flags ─────────
            HarvestMachineFlags();

            // ── Phase 2: Harvest AGV delivery flags ──────────────
            HarvestAGVFlags();

            // ── Phase 3: Assign idle AGVs to WaitingForPickup jobs
            AssignAGVs();

            // ── Phase 4: Feed next decision to agent ─────────────
            if (!IsWaitingForAction)
                FindNextDecision();

            // ── Phase 5: Check episode completion ────────────────
            if (Jobs.AreAllExited())
                FinaliseEpisode();
        }

        // ─────────────────────────────────────────────────────────
        //  Phase 1: Machine Flags
        // ─────────────────────────────────────────────────────────

        private void HarvestMachineFlags()
        {
            foreach (var machine in layoutManager.Machines)
            {
                if (!machine.FinishedFlag) continue;

                int jobId = machine.ActiveJobId;
                machine.ClearFinished();

                JobData job = Jobs.Get(jobId);
                if (job == null)
                {
                    SimLogger.Error($"[Orchestrator] Machine M{machine.MachineId} finished unknown job {jobId}");
                    continue;
                }

                // Advance operation
                job.CompletedOps++;

                if (job.CurrentOpIndex < job.TotalOperations)
                    job.CurrentOpIndex++;

                // Place visual on outgoing belt
                machine.PlaceOnOutgoing(jobId, job.Visual);
                RefreshMachineLabels(machine.MachineId);
                if (job.IsLastOperation)
                {
                    // All ops done → needs transport to exit
                    job.State = JobState.WaitingForPickup;
                    job.TargetMachineId = -1;  // -1 = exit
                    job.LocationMachineId = machine.MachineId;
                    job.StateEntryTime = SimTime;
                    SimLogger.High($"[Orchestrator] Job {jobId} complete → WaitingForPickup(exit)");
                }
                else
                {
                    // More ops → needs routing decision
                    job.State = JobState.NeedsRouting;
                    job.LocationMachineId = machine.MachineId;
                    job.StateEntryTime = SimTime;
                    SimLogger.High($"[Orchestrator] Job {jobId} op {job.CompletedOps}/{job.TotalOperations} done " +
                                   $"→ NeedsRouting (next={job.NextRequiredType})");
                }
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Phase 2: AGV Delivery Flags
        // ─────────────────────────────────────────────────────────

        private void HarvestAGVFlags()
        {
            foreach (var agv in agvPool.AllAGVs)
            {
                if (agv.PickedUpFlag)
                {
                    // Job is now physically on the AGV
                    int jobId = agv.CurrentJobId;
                    JobData job = Jobs.Get(jobId);
                    if (job != null && job.State == JobState.WaitingForPickup)
                    {
                        job.State = JobState.InTransit;
                        job.StateEntryTime = SimTime;
                        SimLogger.High($"[Orchestrator] Job {jobId} picked up by AGV {agv.AgvId} → InTransit");
                    }
                    // Don't clear yet — clear after checking delivered too
                }

                if (agv.DeliveredFlag)
                {
                    int jobId = agv.DeliveredJobId;
                    int machineId = agv.DeliveredMachineId;

                    JobData job = Jobs.Get(jobId);
                    if (job != null)
                    {
                        if (machineId < 0)
                        {
                            // Delivered to exit
                            job.State = JobState.Exited;
                            job.LocationMachineId = -1;
                            job.StateEntryTime = SimTime;
                            if (job.Visual != null) job.Visual.gameObject.SetActive(false);
                            SimLogger.High($"[Orchestrator] Job {jobId} → Exited");
                        }
                        else
                        {
                            // Delivered to machine queue
                            job.State = JobState.Queued;
                            job.LocationMachineId = machineId;
                            job.StateEntryTime = SimTime;
                            SimLogger.High($"[Orchestrator] Job {jobId} → Queued at M{machineId}");

                            PhysicalMachine targetMachine = layoutManager.GetMachine(machineId);
                            targetMachine.PlaceOnIncoming(jobId, job.Visual);
                            RefreshMachineLabels(machineId);
                        }

                        job.TotalTransitTime += (SimTime - job.StateEntryTime);
                        job.AssignedAgvId = -1;
                    }
                }

                // Clear both flags together
                if (agv.PickedUpFlag || agv.DeliveredFlag)
                    agv.ClearFlags();
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Phase 3: AGV Assignment
        // ─────────────────────────────────────────────────────────

        private void AssignAGVs()
        {
            // Assign available AGVs to unassigned WaitingForPickup jobs.
            // GetAvailableAGV() prefers idle AGVs but will also return an AGV
            // currently ReturningToParking — Dispatch() cancels the parking
            // route mid-trip so the AGV pivots directly to the new pickup.
            while (true)
            {
                JobData job = Jobs.GetNextUnassignedPickup();
                if (job == null) break;

                AGVController agv = agvPool.GetAvailableAGV();
                if (agv == null) break;

                // Resolve positions
                PhysicalMachine sourceMachine = null;
                Vector3 pickupPos;
                if (job.LocationMachineId >= 0)
                {
                    sourceMachine = layoutManager.GetMachine(job.LocationMachineId);
                    pickupPos = sourceMachine.GetPickupPosition();
                }
                else
                {
                    pickupPos = layoutManager.IncomingBeltPosition;
                }

                PhysicalMachine targetMachine = null;
                Vector3 dropoffPos;
                if (job.TargetMachineId >= 0)
                {
                    targetMachine = layoutManager.GetMachine(job.TargetMachineId);
                    dropoffPos = targetMachine.GetDropoffPosition();
                }
                else
                {
                    dropoffPos = layoutManager.OutgoingBeltPosition;
                }

                // Dispatch AGV
                job.AssignedAgvId = agv.AgvId;
                agv.Dispatch(job.JobId, pickupPos, dropoffPos, sourceMachine, targetMachine, job.Visual);
                agv.SetCarryVisual(job.Visual);

                SimLogger.High($"[Orchestrator] Assigned AGV {agv.AgvId} to job {job.JobId} " +
                               $"(M{job.LocationMachineId} → M{job.TargetMachineId})");
            }

        }

        // ─────────────────────────────────────────────────────────
        //  Phase 4: Find Next Decision
        // ─────────────────────────────────────────────────────────

        private void FindNextDecision()
        {
            // Priority 1: Routing decisions (jobs need a target machine)
            JobData routingJob = Jobs.GetNextNeedsRouting();
            if (routingJob != null)
            {
                CurrentDecision = BuildRoutingDecision(routingJob);
                IsWaitingForAction = true;
                OnDecisionRequired?.Invoke(CurrentDecision);
                return;
            }

            // Priority 2: Dispatch decisions (idle machine with queued jobs)
            foreach (var machine in layoutManager.Machines)
            {
                if (!machine.IsIdle) continue;
                if (!Jobs.HasDispatchableJob(machine.MachineId)) continue;

                CurrentDecision = BuildDispatchDecision(machine.MachineId);
                IsWaitingForAction = true;
                OnDecisionRequired?.Invoke(CurrentDecision);
                return;
            }

        }

        // ═════════════════════════════════════════════════════════
        //  STEP — called by SchedulingAgent.OnActionReceived
        // ═════════════════════════════════════════════════════════

        public StepResult Step(int actionIndex)
        {
            IsWaitingForAction = false;

            if (CurrentDecision.Type == DecisionType.Routing)
            {
                ExecuteRoutingDecision(actionIndex);
            }
            else if (CurrentDecision.Type == DecisionType.Dispatch)
            {
                ExecuteDispatchDecision(actionIndex);
            }

            float reward = CalculateReward();
            totalReward += reward;

            return new StepResult
            {
                Reward = reward,
                Done = false,
                CurrentMakespan = SimTime
            };
        }

        // ─────────────────────────────────────────────────────────
        //  Execute Routing: agent picked a target machine
        // ─────────────────────────────────────────────────────────

        private void ExecuteRoutingDecision(int actionIndex)
        {
            int chosenMachineId = ApplyMachineSelectionRule(actionIndex, CurrentDecision);

            JobData job = Jobs.Get(CurrentDecision.JobId);
            if (job == null) return;

            job.TargetMachineId = chosenMachineId;
            job.State = JobState.WaitingForPickup;
            job.StateEntryTime = SimTime;

            SimLogger.High($"[Orchestrator] Routed job {job.JobId} → M{chosenMachineId} " +
                           $"(type={job.NextRequiredType})");

            // AGV assignment happens in Phase 3 next frame.
            // No immediate dispatch — keeps the tick clean.
        }

        // ─────────────────────────────────────────────────────────
        //  Execute Dispatch: agent picked a job for the machine
        // ─────────────────────────────────────────────────────────

        private void ExecuteDispatchDecision(int actionIndex)
        {
            int machineId = CurrentDecision.MachineId;
            int chosenJobId = ApplyDispatchingRule(actionIndex, machineId);

            if (chosenJobId < 0)
            {
                SimLogger.LogWarning($"[Orchestrator] No dispatchable job at M{machineId}.");
                return;
            }

            JobData job = Jobs.Get(chosenJobId);
            if (job == null) return;

            // Guard: verify job is actually Queued at this machine
            if (job.State != JobState.Queued || job.LocationMachineId != machineId)
            {
                SimLogger.Error($"[Orchestrator] Job {chosenJobId} not Queued at M{machineId} " +
                                $"(state={job.State}, loc=M{job.LocationMachineId}). Skipping.");
                return;
            }

            // Guard: verify operation index is valid
            if (job.CurrentOpIndex >= job.TotalOperations)
            {
                SimLogger.Error($"[Orchestrator] Job {chosenJobId} has no more operations. Skipping.");
                return;
            }

            float duration = job.GetProcessingTime(machineId);
            if (duration <= 0f)
            {
                SimLogger.LogWarning($"[Orchestrator] Job {chosenJobId} has 0 processing time at M{machineId}. " +
                                     "Machine not eligible for this operation?");
                // Still process it — could be a valid 0-time op, but log it.
            }

            // Transition job
            job.State = JobState.Processing;
            job.TotalWaitTime += (SimTime - job.StateEntryTime);
            job.StateEntryTime = SimTime;

            // Tell machine to run the timer
            PhysicalMachine machine = layoutManager.GetMachine(machineId);
            machine.StartJob(chosenJobId, duration, job.Visual);
            RefreshMachineLabels(machineId);

            LastAppliedRule = ActionToRule[actionIndex].ToString();

            SimLogger.High($"[Orchestrator] M{machineId} processing job {chosenJobId} " +
                           $"(op {job.CurrentOpIndex}/{job.TotalOperations}, {duration:F1}s)");
        }

        // ═════════════════════════════════════════════════════════
        //  DECISION BUILDERS
        // ═════════════════════════════════════════════════════════

        private DecisionRequest BuildRoutingDecision(JobData job)
        {
            MachineType required = job.NextRequiredType;

            var candidates = new List<int>();
            foreach (var m in layoutManager.Machines)
                if (m.MachineType == required)
                    candidates.Add(m.MachineId);

            float[] queueLengths = new float[candidates.Count];
            float[] jobTimes = new float[candidates.Count];

            for (int i = 0; i < candidates.Count; i++)
            {
                queueLengths[i] = Jobs.GetMachineLoad(candidates[i]);
                jobTimes[i] = job.GetProcessingTime(candidates[i]);
            }

            return new DecisionRequest
            {
                Type = DecisionType.Routing,
                SimTime = SimTime,
                DecisionIndex = decisionCount++,
                TotalJobs = Jobs.JobCount,
                CompletedJobs = Jobs.CountInState(JobState.Exited),
                JobId = job.JobId,
                SourceMachineId = job.LocationMachineId,
                RequiredType = required,
                CandidateMachineIds = candidates.ToArray(),
                CandidateQueueLengths = queueLengths,
                CandidateJobTimes = jobTimes,
            };
        }

        private DecisionRequest BuildDispatchDecision(int machineId)
        {
            List<int> queue = Jobs.GetDispatchableJobs(machineId);
            int[] jobIds = queue.ToArray();
            double[] durations = new double[jobIds.Length];

            for (int i = 0; i < jobIds.Length; i++)
                durations[i] = Jobs.GetProcessingTime(jobIds[i], machineId);

            return new DecisionRequest
            {
                Type = DecisionType.Dispatch,
                MachineId = machineId,
                SimTime = SimTime,
                DecisionIndex = decisionCount++,
                TotalJobs = Jobs.JobCount,
                CompletedJobs = Jobs.CountInState(JobState.Exited),
                QueuedJobIds = jobIds,
                QueuedDurations = durations,
            };
        }

        // ═════════════════════════════════════════════════════════
        //  DISPATCHING RULES (same logic, just reads from JobStore)
        // ═════════════════════════════════════════════════════════

        private int ApplyDispatchingRule(int actionIndex, int machineId)
        {
            DispatchingRule rule = ActionToRule[actionIndex];
            List<int> queue = Jobs.GetDispatchableJobs(machineId);

            if (queue.Count == 0) return -1;
            if (queue.Count == 1) return queue[0];

            return rule switch
            {
                DispatchingRule.SPT_SMPT or
                DispatchingRule.SPT_SRWT =>
                    ArgMin(queue, id => Jobs.Get(id).GetProcessingTime(machineId)),

                DispatchingRule.LPT_MMUR or
                DispatchingRule.LPT_SMPT =>
                    ArgMax(queue, id => Jobs.Get(id).GetProcessingTime(machineId)),

                DispatchingRule.SRT_SRWT or
                DispatchingRule.SRT_SMPT =>
                    ArgMin(queue, id => GetRemainingWork(id)),

                DispatchingRule.LRT_MMUR =>
                    ArgMax(queue, id => GetRemainingWork(id)),

                DispatchingRule.SDT_SRWT =>
                    ArgMin(queue, id => (float)(SimTime - Jobs.Get(id).ArrivalTime)),

                _ => queue[0]
            };
        }

        private int ApplyMachineSelectionRule(int actionIndex, DecisionRequest req)
        {
            DispatchingRule rule = ActionToRule[actionIndex];
            int[] candidates = req.CandidateMachineIds;

            if (candidates.Length == 1) return candidates[0];

            return rule switch
            {
                DispatchingRule.SPT_SMPT or
                DispatchingRule.LPT_SMPT or
                DispatchingRule.SRT_SMPT =>
                    candidates[ArgMinIdx(req.CandidateJobTimes)],

                DispatchingRule.SPT_SRWT or
                DispatchingRule.SRT_SRWT or
                DispatchingRule.SDT_SRWT =>
                    candidates[ArgMinIdx(req.CandidateQueueLengths)],

                DispatchingRule.LPT_MMUR or
                DispatchingRule.LRT_MMUR =>
                    candidates[ArgMaxIdx(req.CandidateQueueLengths)],

                _ => candidates[0]
            };
        }

        // ─────────────────────────────────────────────────────────
        //  Helpers
        // ─────────────────────────────────────────────────────────

        private void RefreshMachineLabels(int machineId)
        {
            PhysicalMachine machine = layoutManager.GetMachine(machineId);
            if (machine == null) return;

            int inCount = Jobs.GetDispatchableJobs(machineId).Count;

            int outCount = 0;
            foreach (var j in Jobs.AllJobs)
                if (j.LocationMachineId == machineId && j.State == JobState.WaitingForPickup)
                    outCount++;

            machine.RefreshQueueLabels(inCount, outCount);
        }
        private float GetRemainingWork(int jobId)
        {
            JobData j = Jobs.Get(jobId);
            if (j == null) return 0f;
            float total = 0f;
            for (int o = j.CurrentOpIndex; o < j.TotalOperations; o++)
            {
                float min = float.MaxValue;
                foreach (float t in j.EligibleMachinesPerOp[o].Values)
                    if (t < min) min = t;
                if (min < float.MaxValue) total += min;
            }
            return total;
        }

        private float CalculateReward()
        {
            float current = (float)SimTime;
            float delta = current - (float)previousMakespan;
            previousMakespan = current;

            int totalOps = 0;
            foreach (var j in Jobs.AllJobs) totalOps += j.TotalOperations;
            return -delta / (Mathf.Max(totalOps, 1) * Time.timeScale);
        }

        private void FinaliseEpisode()
        {
            episodeActive = false;
            SimLogger.Low($"[Orchestrator] All jobs exited. Makespan={SimTime:F1}, decisions={decisionCount}");
            // Fire events, log results, etc.
            OnEpisodeFinished?.Invoke(new EpisodeResult
            {
                Makespan = SimTime,
                DecisionPoints = decisionCount,
                TotalReward = totalReward
            });
        }

        // ── Generic helpers ──────────────────────────────────────

        private int ArgMin(List<int> ids, Func<int, float> score)
        {
            int best = ids[0]; float bestS = float.MaxValue;
            foreach (int id in ids) { float s = score(id); if (s < bestS) { bestS = s; best = id; } }
            return best;
        }
        private int ArgMax(List<int> ids, Func<int, float> score)
        {
            int best = ids[0]; float bestS = float.MinValue;
            foreach (int id in ids) { float s = score(id); if (s > bestS) { bestS = s; best = id; } }
            return best;
        }
        private int ArgMinIdx(float[] v)
        {
            int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] < v[b]) b = i; return b;
        }
        private int ArgMaxIdx(float[] v)
        {
            int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i; return b;
        }

        private FJSSPConfig BuildTestConfig()
        {
            var layout = new MachineType[5];
            MachineType[] types = (MachineType[])Enum.GetValues(typeof(MachineType));
            for (int i = 0; i < 5; i++) layout[i] = types[i];

            return new FJSSPConfig
            {
                Seed = 42,
                JobCount = 3,
                MachinesPerType = 1,
                MachineTypeLayout = layout,
                MinProcTime = 15f,
                MaxProcTime = 30f,
                MinOpsPerJob = 2,
                MaxOpsPerJob = 5,
                MaxArrivalTime = 0f
            };
        }

        private FJSSPConfig BuildDefaultConfig()
        {
            var layout = new MachineType[15];
            MachineType[] types = (MachineType[])Enum.GetValues(typeof(MachineType));
            for (int i = 0; i < 15; i++) layout[i] = types[i / 3];

            return new FJSSPConfig
            {
                Seed = 42,
                JobCount = 20,
                MachinesPerType = 3,
                MachineTypeLayout = layout,
                MinProcTime = 15f,
                MaxProcTime = 90f,
                MinOpsPerJob = 5,
                MaxOpsPerJob = 8,
                MaxArrivalTime = 0f
            };
        }
    }
}