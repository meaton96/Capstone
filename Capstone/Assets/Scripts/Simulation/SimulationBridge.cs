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

namespace Assets.Scripts.Simulation
{
    /// @brief The central orchestrator responsible for driving the factory simulation.
    ///
    /// @details SimulationBridge implements a strictly centralized state machine. 
    /// In a single @c Update tick, it harvests status flags from physical components 
    /// (Machines and AGVs), manages job transitions, resolves AGV assignments, 
    /// and interfaces with the @c SchedulingAgent to resolve scheduling conflicts. 
    /// No other component is permitted to mutate @c JobData state.
    public class SimulationBridge : MonoBehaviour
    {
        public static SimulationBridge Instance;

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

        [Header("Events")]
        public UnityEvent<DecisionRequest> OnDecisionRequired;
        public UnityEvent<StepResult> OnStepCompleted;
        public UnityEvent<EpisodeResult> OnEpisodeFinished;
        public UnityEvent OnFactorySpawned;

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

        /// @brief Updates the active configuration for the next simulation run.
        public void LoadConfig(FJSSPConfig config)
        {
            currentConfig = config;
            IsFactoryReady = false;
        }
        /// @brief Injects pre-built job definitions (e.g. from Brandimarte benchmarks).
        public void LoadPrebuiltJobs(FJSSPJobDefinition[] jobs)
        {
            prebuiltJobs = jobs;
        }

        /// @brief Physically instantiates the factory floor, machines, and AGV fleet.
        ///
        /// @details Initializes the @c layoutManager, @c trafficZoneManager, and 
        /// @c agvPool based on the current @c FJSSPConfig. This must be called 
        /// before @c StartEpisode.
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

        /// @brief Initiates a new simulation episode.
        ///
        /// @details Generates a new set of job definitions, initializes the 
        /// @c JobStore, and resets all performance metrics (makespan, reward, 
        /// and decision counts).
        public void StartEpisode()
        {
            // ── Diagnostic: dump job states from the previous episode ──
            if (Jobs != null && Jobs.IsInitialized && Jobs.JobCount > 0)
            {
                var counts = new Dictionary<JobState, int>();
                foreach (var job in Jobs.AllJobs)
                {
                    if (!counts.ContainsKey(job.State)) counts[job.State] = 0;
                    counts[job.State]++;
                }
                string summary = string.Join(", ", counts.Select(
                    kvp => $"{kvp.Key}={kvp.Value}"));
                SimLogger.Low($"[Orchestrator] Previous episode state at reset: {summary}");

                // Log first stuck job details
                var stuck = Jobs.AllJobs.FirstOrDefault(j => j.State != JobState.Exited);
                if (stuck != null)
                {
                    SimLogger.Low($"[Orchestrator] Stuck job example: Job {stuck.JobId} " +
                        $"State={stuck.State} Op={stuck.CurrentOpIndex}/{stuck.TotalOperations} " +
                        $"Location=M{stuck.LocationMachineId} Target=M{stuck.TargetMachineId} " +
                        $"AGV={stuck.AssignedAgvId} PreAGV={stuck.PreDispatchedAgvId}");
                }
            }
            // if (Time.timeScale - 1 <= .001f)
            // {
            //     SimLogger.Low("Setting timescale to 100f");
            //     Time.timeScale = 100;
            // }
            if (currentConfig == null)
            {
                currentConfig = BuildDefaultConfig();
            }

            if (!IsFactoryReady)
            {
                SpawnFactory();
            }

            agent.SetHeuristicRule(currentConfig.dispatchingRule);

            // ── Use prebuilt jobs if injected, otherwise generate ──
            FJSSPJobDefinition[] jobDefs;
            if (prebuiltJobs != null)
            {
                jobDefs = prebuiltJobs;
                prebuiltJobs = null;   // single-use: clear after consumption
                SimLogger.Low("[Orchestrator] Using prebuilt benchmark jobs");
            }
            else
            {
                jobDefs = FJSSPJobGenerator.Generate(currentConfig, cachedMachinesByType);
            }

            Jobs.Initialize(jobDefs, spawnVisuals: true);

            episodeActive = true;
            decisionCount = 0;
            totalReward = 0;
            previousMakespan = 0;
            IsWaitingForAction = false;
            startTime = Time.time;

            SimLogger.Low($"[Orchestrator] Episode started: {currentConfig.JobCount} jobs, " +
                           $"{layoutManager.MachineCount} machines");
        }

        /// @brief Aborts the current episode and cleans up all runtime data.
        public void StopEpisode()
        {
            episodeActive = false;
            IsWaitingForAction = false;
            IsFactoryReady = false;
            layoutManager.ClearFloor();
            Jobs.Cleanup();
            agvPool.ClearFleet();
        }

        /// @brief The core execution loop of the simulation.
        ///
        /// @details Processes the simulation in five distinct phases:
        /// 1. Harvest completion flags from machines.
        /// 2. Harvest delivery/pickup flags from AGVs.
        /// 3. Predictive pre-dispatch of AGVs for near-complete operations.
        /// 4. Assignment of AGVs to jobs awaiting transport.
        /// 5. Identification and triggering of the next scheduling decision.
        private void Update()
        {
            if (!episodeActive) return;

            HarvestMachineFlags();
            HarvestAGVFlags();
            HarvestAlmostDoneFlags();
            AssignAGVs();

            if (!IsWaitingForAction)
                FindNextDecision();

            if (Jobs.AreAllExited())
                FinaliseEpisode();
        }

        /// @brief Processes machines that have finished their current processing timer.
        ///
        /// @details Advances the operation index of the associated job, updates 
        /// conveyor visuals, and transitions the job state to either @c NeedsRouting 
        /// or @c WaitingForPickup (if all operations are complete).
        private void HarvestMachineFlags()
        {
            foreach (var machine in layoutManager.Machines)
            {
                if (!machine.FinishedFlag) continue;

                int jobId = machine.ActiveJobId;
                machine.ClearFinished();

                JobData job = Jobs.Get(jobId);
                if (job == null) continue;

                job.CompletedOps++;
                if (job.CurrentOpIndex < job.TotalOperations)
                    job.CurrentOpIndex++;

                machine.PlaceOnOutgoing(jobId, job.Visual);
                RefreshMachineLabels(machine.MachineId);

                if (job.IsLastOperation)
                {
                    job.State = JobState.WaitingForPickup;
                    job.TargetMachineId = -1;
                    job.LocationMachineId = machine.MachineId;
                    job.StateEntryTime = SimTime;

                    if (job.PreDispatchedAgvId >= 0)
                    {
                        AGVController preAgv = agvPool.GetPreDispatchedAGV(job.JobId);
                        if (preAgv != null)
                        {
                            preAgv.FinalizePreDispatch(job.JobId, layoutManager.OutgoingBeltPosition, null, job.Visual);
                            job.AssignedAgvId = preAgv.AgvId;
                        }
                        job.PreDispatchedAgvId = -1;
                    }
                }
                else
                {
                    job.State = JobState.NeedsRouting;
                    job.LocationMachineId = machine.MachineId;
                    job.StateEntryTime = SimTime;
                }
            }
        }

        /// @brief Triggers predictive AGV movement for jobs nearing completion.
        ///
        /// @details Checks @c AlmostDoneFlag on all machines. If a machine is 
        /// within the @c PreDispatchLeadTime window, an AGV is dispatched to 
        /// its pickup dock ahead of the actual completion event.
        private void HarvestAlmostDoneFlags()
        {
            foreach (var machine in layoutManager.Machines)
            {
                if (!machine.AlmostDoneFlag) continue;

                int jobId = machine.AlmostDoneJobId;
                machine.ClearAlmostDone();

                JobData job = Jobs.Get(jobId);
                if (job == null || job.State != JobState.Processing || job.PreDispatchedAgvId >= 0) continue;
                if (job.CompletedOps == job.TotalOperations - 1) continue;

                AGVController agv = agvPool.GetAvailableAGV();
                if (agv == null) continue;

                agv.PreDispatch(jobId, machine.GetPickupPosition(), machine);
                job.PreDispatchedAgvId = agv.AgvId;
            }
        }

        /// @brief Processes AGV completion flags to transition job states.
        ///
        /// @details Handles @c PickedUpFlag (transitions job to @c InTransit) 
        /// and @c DeliveredFlag (transitions job to @c Queued or @c Exited).
        private void HarvestAGVFlags()
        {
            foreach (var agv in agvPool.AllAGVs)
            {
                if (agv.PickedUpFlag)
                {
                    int jobId = agv.CurrentJobId;
                    JobData job = Jobs.Get(jobId);
                    if (job != null && job.State == JobState.WaitingForPickup)
                    {
                        job.State = JobState.InTransit;
                        job.StateEntryTime = SimTime;
                    }
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
                            job.State = JobState.Exited;
                            job.LocationMachineId = -1;
                            job.StateEntryTime = SimTime;
                            if (job.Visual != null) job.Visual.gameObject.SetActive(false);
                        }
                        else
                        {
                            job.State = JobState.Queued;
                            job.LocationMachineId = machineId;
                            job.StateEntryTime = SimTime;

                            PhysicalMachine targetMachine = layoutManager.GetMachine(machineId);
                            targetMachine.PlaceOnIncoming(jobId, job.Visual);
                            RefreshMachineLabels(machineId);
                        }
                        job.TotalTransitTime += (SimTime - job.StateEntryTime);
                        job.AssignedAgvId = -1;
                    }
                }

                if (agv.PickedUpFlag || agv.DeliveredFlag)
                    agv.ClearFlags();
            }
        }

        /// @brief Pairs unassigned jobs with available AGV units.
        ///
        /// @details Iterates through jobs in @c WaitingForPickup and attempts 
        /// to dispatch idle or returning AGVs to fulfill the transport request.
        private void AssignAGVs()
        {
            // Collect candidates upfront to avoid the while/continue pattern
            // that can infinite-loop if GetNextUnassignedPickup returns the
            // same pre-dispatched job repeatedly.
            var candidates = new List<JobData>();
            foreach (var job in Jobs.AllJobs)
            {
                if (job.State == JobState.WaitingForPickup
                    && job.AssignedAgvId == -1
                    && job.PreDispatchedAgvId < 0)
                {
                    candidates.Add(job);
                }
            }

            foreach (var job in candidates)
            {
                AGVController agv = agvPool.GetAvailableAGV();
                if (agv == null) break;

                PhysicalMachine sourceMachine = job.LocationMachineId >= 0 ? layoutManager.GetMachine(job.LocationMachineId) : null;
                Vector3 pickupPos = sourceMachine != null ? sourceMachine.GetPickupPosition() : layoutManager.IncomingBeltPosition;

                PhysicalMachine targetMachine = job.TargetMachineId >= 0 ? layoutManager.GetMachine(job.TargetMachineId) : null;
                Vector3 dropoffPos = targetMachine != null ? targetMachine.GetDropoffPosition() : layoutManager.OutgoingBeltPosition;

                job.AssignedAgvId = agv.AgvId;
                agv.Dispatch(job.JobId, pickupPos, dropoffPos, sourceMachine, targetMachine, job.Visual);
                agv.SetCarryVisual(job.Visual);
            }
        }

        /// @brief Evaluates the factory state to determine if a new decision is required.
        ///
        /// @details Prioritizes @c Routing decisions (choosing machines for jobs) 
        /// over @c Dispatch decisions (choosing jobs for idle machines). Triggers 
        /// the @c OnDecisionRequired event for the agent.
        private void FindNextDecision()
        {
            JobData routingJob = Jobs.GetNextNeedsRouting();
            if (routingJob != null)
            {
                CurrentDecision = BuildRoutingDecision(routingJob);
                IsWaitingForAction = true;
                OnDecisionRequired?.Invoke(CurrentDecision);
                return;
            }

            foreach (var machine in layoutManager.Machines)
            {
                if (machine.IsIdle && Jobs.HasDispatchableJob(machine.MachineId))
                {
                    CurrentDecision = BuildDispatchDecision(machine.MachineId);
                    IsWaitingForAction = true;
                    OnDecisionRequired?.Invoke(CurrentDecision);
                    return;
                }
            }
        }

        /// @brief Applies an agent's chosen action to the simulation.
        ///
        /// @param actionIndex The discrete index of the dispatching rule to apply.
        /// @return A @c StepResult containing the immediate reward and episode status.
        public StepResult Step(int actionIndex)
        {
            IsWaitingForAction = false;

            if (CurrentDecision.Type == DecisionType.Routing)
                ExecuteRoutingDecision(actionIndex);
            else if (CurrentDecision.Type == DecisionType.Dispatch)
                ExecuteDispatchDecision(actionIndex);

            float reward = CalculateReward();
            totalReward += reward;

            return new StepResult
            {
                Reward = reward,
                Done = false,
                CurrentMakespan = SimTime
            };
        }

        /// @brief Finalizes a machine assignment for a job based on the chosen rule.
        private void ExecuteRoutingDecision(int actionIndex)
        {
            int chosenMachineId = ApplyMachineSelectionRule(actionIndex, CurrentDecision);
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
                    Vector3 dropoffPos = targetMachine != null ? targetMachine.GetDropoffPosition() : layoutManager.OutgoingBeltPosition;
                    preAgv.FinalizePreDispatch(job.JobId, dropoffPos, targetMachine, job.Visual);
                    job.AssignedAgvId = preAgv.AgvId;
                    job.PreDispatchedAgvId = -1;
                    return;
                }
                job.PreDispatchedAgvId = -1;
            }
        }

        /// @brief Commences machine processing for a job selected by the chosen rule.
        private void ExecuteDispatchDecision(int actionIndex)
        {
            int machineId = CurrentDecision.MachineId;
            int chosenJobId = ApplyDispatchingRule(actionIndex, machineId);

            JobData job = Jobs.Get(chosenJobId);
            if (job == null || job.State != JobState.Queued || job.LocationMachineId != machineId) return;

            float duration = job.GetProcessingTime(machineId);
            job.State = JobState.Processing;
            job.TotalWaitTime += (SimTime - job.StateEntryTime);
            job.StateEntryTime = SimTime;

            PhysicalMachine machine = layoutManager.GetMachine(machineId);
            machine.StartJob(chosenJobId, duration, job.Visual);
            RefreshMachineLabels(machineId);

            LastAppliedRule = ActionToRule[actionIndex].ToString();
        }

        /// @brief Constructs a request for a machine routing decision.
        /// 
        /// @details Identifies all machines capable of performing the job's next required 
        /// @c MachineType and gathers their current workloads (load balancing signal) 
        /// and expected processing times for the agent to evaluate.
        ///
        /// @param job The job requiring a target machine assignment.
        /// @return A @c DecisionRequest object of type @c DecisionType.Routing.
        private DecisionRequest BuildRoutingDecision(JobData job)
        {
            MachineType required = job.NextRequiredType;
            var candidates = layoutManager.Machines.Where(m => m.MachineType == required).Select(m => m.MachineId).ToList();

            float[] queueLengths = candidates.Select(id => Jobs.GetMachineLoad(id)).ToArray();
            float[] jobTimes = candidates.Select(id => job.GetProcessingTime(id)).ToArray();

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

        /// @brief Constructs a request for a job dispatch decision.
        /// 
        /// @details Aggregates all jobs currently in the @c Queued state at a specific 
        /// machine. This allows the agent to select which job should be processed next 
        /// based on the machine's local queue pressure.
        ///
        /// @param machineId The ID of the idle machine requesting a job.
        /// @return A @c DecisionRequest object of type @c DecisionType.Dispatch.
        private DecisionRequest BuildDispatchDecision(int machineId)
        {
            List<int> queue = Jobs.GetDispatchableJobs(machineId);
            double[] durations = queue.Select(id => (double)Jobs.GetProcessingTime(id, machineId)).ToArray();

            return new DecisionRequest
            {
                Type = DecisionType.Dispatch,
                MachineId = machineId,
                SimTime = SimTime,
                DecisionIndex = decisionCount++,
                TotalJobs = Jobs.JobCount,
                CompletedJobs = Jobs.CountInState(JobState.Exited),
                QueuedJobIds = queue.ToArray(),
                QueuedDurations = durations,
            };
        }

        /// @brief Resolves a dispatch decision using a specific heuristic rule.
        /// 
        /// @details Maps the @c actionIndex to a @c DispatchingRule and applies the 
        /// logic (e.g., Shortest Processing Time) to the machine's current queue.
        ///
        /// @param actionIndex The discrete action index provided by the agent.
        /// @param machineId The ID of the machine where the rule is being applied.
        /// @return The ID of the job selected for processing, or -1 if the queue is empty.
        private int ApplyDispatchingRule(int actionIndex, int machineId)
        {
            DispatchingRule rule = ActionToRule[actionIndex];
            List<int> queue = Jobs.GetDispatchableJobs(machineId);

            if (queue.Count == 0) return -1;
            if (queue.Count == 1) return queue[0];

            return rule switch
            {
                DispatchingRule.SPT_SMPT or DispatchingRule.SPT_SRWT => ArgMin(queue, id => Jobs.Get(id).GetProcessingTime(machineId)),
                DispatchingRule.LPT_MMUR or DispatchingRule.LPT_SMPT => ArgMax(queue, id => Jobs.Get(id).GetProcessingTime(machineId)),
                DispatchingRule.SRT_SRWT or DispatchingRule.SRT_SMPT => ArgMin(queue, id => GetRemainingWork(id)),
                DispatchingRule.LRT_MMUR => ArgMax(queue, id => GetRemainingWork(id)),
                DispatchingRule.SDT_SRWT => ArgMin(queue, id => (float)(SimTime - Jobs.Get(id).ArrivalTime)),
                _ => queue[0]
            };
        }

        /// @brief Resolves a routing decision using a specific heuristic rule.
        /// 
        /// @details Selects a target machine from the available candidates by applying 
        /// the chosen rule to signals like queue length or processing time.
        ///
        /// @param actionIndex The discrete action index provided by the agent.
        /// @param req The @c DecisionRequest context containing candidate machine data.
        /// @return The ID of the machine selected as the job's destination.
        private int ApplyMachineSelectionRule(int actionIndex, DecisionRequest req)
        {
            DispatchingRule rule = ActionToRule[actionIndex];
            int[] candidates = req.CandidateMachineIds;

            if (candidates.Length == 1) return candidates[0];

            return rule switch
            {
                DispatchingRule.SPT_SMPT or DispatchingRule.LPT_SMPT or DispatchingRule.SRT_SMPT => candidates[ArgMinIdx(req.CandidateJobTimes)],
                DispatchingRule.SPT_SRWT or DispatchingRule.SRT_SRWT or DispatchingRule.SDT_SRWT => candidates[ArgMinIdx(req.CandidateQueueLengths)],
                DispatchingRule.LPT_MMUR or DispatchingRule.LRT_MMUR => candidates[ArgMaxIdx(req.CandidateQueueLengths)],
                _ => candidates[0]
            };
        }

        /// @brief Updates the physical UI labels for a specific machine.
        /// 
        /// @details Forces the @c PhysicalMachine to refresh its HUD based on the 
        /// number of jobs currently in its incoming @c Queued state and its 
        /// outgoing @c WaitingForPickup state.
        private void RefreshMachineLabels(int machineId)
        {
            PhysicalMachine machine = layoutManager.GetMachine(machineId);
            if (machine == null) return;

            int inCount = Jobs.GetDispatchableJobs(machineId).Count;
            int outCount = Jobs.AllJobs.Count(j => j.LocationMachineId == machineId && j.State == JobState.WaitingForPickup);
            machine.RefreshQueueLabels(inCount, outCount);
        }

        /// @brief Estimates the total remaining processing time for a job.
        /// 
        /// @details Iterates through all remaining operations and sums the 
        /// minimum possible processing time for each step.
        ///
        /// @param jobId The ID of the job to evaluate.
        /// @return The sum of minimum processing times for all pending operations.
        private float GetRemainingWork(int jobId)
        {
            JobData j = Jobs.Get(jobId);
            if (j == null) return 0f;
            float total = 0f;
            for (int o = j.CurrentOpIndex; o < j.TotalOperations; o++)
            {
                float min = j.EligibleMachinesPerOp[o].Values.Min();
                total += min;
            }
            return total;
        }

        /// @brief Computes the reward signal for the reinforcement learning agent.
        /// 
        /// @details Calculates a negative reward based on the incremental makespan 
        /// increase since the last decision, normalized by the total workload:
        /// $$R = -\frac{\Delta makespan}{TotalOps \times TimeScale}$$
        ///
        /// @return A float representing the step reward.
        private float CalculateReward()
        {
            float current = (float)SimTime;
            float delta = current - (float)previousMakespan;
            previousMakespan = current;

            int totalOps = Jobs.AllJobs.Sum(j => j.TotalOperations);
            return -delta / (Mathf.Max(totalOps, 1) * Time.timeScale);
        }

        /// @brief Finalizes the simulation episode once all jobs have exited.
        /// 
        /// @details Deactivates the simulation loop and invokes the @c OnEpisodeFinished 
        /// event with summarized performance data.
        private void FinaliseEpisode()
        {
            episodeActive = false;
            OnEpisodeFinished?.Invoke(new EpisodeResult
            {
                Makespan = SimTime,
                DecisionPoints = decisionCount,
                TotalReward = totalReward,
                AGVCount = agvPool.AllAGVs.Count,
            });
        }

        /// @brief Utility to find the ID with the minimum score in a list.
        private int ArgMin(List<int> ids, Func<int, float> score)
        {
            int best = ids[0]; float bestS = float.MaxValue;
            foreach (int id in ids) { float s = score(id); if (s < bestS) { bestS = s; best = id; } }
            return best;
        }

        /// @brief Utility to find the ID with the maximum score in a list.
        private int ArgMax(List<int> ids, Func<int, float> score)
        {
            int best = ids[0]; float bestS = float.MinValue;
            foreach (int id in ids) { float s = score(id); if (s > bestS) { bestS = s; best = id; } }
            return best;
        }

        /// @brief Utility to find the index of the minimum value in a float array.
        private int ArgMinIdx(float[] v)
        {
            int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] < v[b]) b = i; return b;
        }

        /// @brief Utility to find the index of the maximum value in a float array.
        private int ArgMaxIdx(float[] v)
        {
            int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i; return b;
        }

        // private FJSSPConfig BuildDefaultConfig()
        // {
        //     var layout = new MachineType[5];
        //     MachineType[] types = (MachineType[])Enum.GetValues(typeof(MachineType));
        //     for (int i = 0; i < 5; i++) layout[i] = types[i];

        //     return new FJSSPConfig
        //     {
        //         Seed = 42,
        //         JobCount = 5,
        //         MachinesPerType = 1,
        //         MachineTypeLayout = layout,
        //         MinProcTime = 5f,
        //         MaxProcTime = 20f,
        //         MinOpsPerJob = 2,
        //         MaxOpsPerJob = 4,
        //         MaxArrivalTime = 0f
        //     };
        // }

        private FJSSPConfig BuildDefaultConfig()
        {
            MachineType[] types = (MachineType[])Enum.GetValues(typeof(MachineType));
            var layout = new MachineType[types.Length];
            for (int i = 0; i < types.Length; i++) layout[i] = types[i];

            var procParams = new Dictionary<MachineType, (float mu, float sigma)>
            {
                { MachineType.Mill,     (mu:  90f, sigma: 10f) },
                { MachineType.Lathe,    (mu:  75f, sigma: 10f) },
                { MachineType.Weld,     (mu: 150f, sigma: 25f) },
                { MachineType.Inspect,  (mu:  60f, sigma: 10f) },
                { MachineType.Assemble, (mu: 240f, sigma: 40f) },
            };

            return new FJSSPConfig
            {
                Seed = 42,
                JobCount = 5,
                MachinesPerType = 1,
                MachineTypeLayout = layout,
                MinProcTime = 1f,
                MaxProcTime = 30f,
                MinOpsPerJob = 2,
                MaxOpsPerJob = 4,
                MaxArrivalTime = 0f,
                ProcTimeParams = procParams,
                dispatchingRule = DispatchingRule.SRT_SRWT,
            };
        }
    }
}