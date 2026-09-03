using System;
using System.Collections.Generic;
using System.Linq;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Logging;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// Central coordinator for making routing and dispatching decisions in the factory simulation.
    /// Manages the lifecycle of decision requests, including job routing to eligible machines
    /// and dispatching jobs to available machines.
    /// </summary>
    public class DecisionCoordinator
    {
        /// <summary>
        /// Reference to the job store for retrieving and managing job data.
        /// </summary>
        private JobStore _jobs;

        /// <summary>
        /// Reference to the factory layout manager for accessing machine information.
        /// </summary>
        private FactoryLayoutManager _layout;

        /// <summary>
        /// Delegate for retrieving the current simulation time.
        /// </summary>
        private Func<double> _getSimTime;

        /// <summary>
        /// Delegate for retrieving the current decision count.
        /// </summary>
        private Func<int> _getDecisionCount;

        /// <summary>
        /// Delegate for incrementing the decision counter.
        /// </summary>
        private Action _incrementDecisionCount;

        /// <summary>
        /// Delegate returning the active baseline PDR's action index (0-8), or -1 if no fixed
        /// rule applies (RL-agent mode, where the agent hasn't acted yet at decision-assembly
        /// time, so pre-selecting a job by rule wouldn't make sense). Only set by baseline/
        /// heuristic headless runs (BaselineDrainMode) — null in interactive/RL mode.
        /// </summary>
        private Func<int> _getBaselineActionIndex;

        /// <summary>
        /// Initializes the coordinator with required dependencies via dependency injection.
        /// </summary>
        /// <param name="jobs">The job store providing job data and state information.</param>
        /// <param name="layout">The factory layout manager providing machine information.</param>
        /// <param name="getSimTime">Delegate to retrieve the current simulation time.</param>
        /// <param name="getDecisionCount">Delegate to retrieve the current decision count.</param>
        /// <param name="incrementDecisionCount">Delegate to increment the decision counter.</param>
        /// <param name="getBaselineActionIndex">Delegate returning the active baseline rule's
        /// action index, or -1/null if none (RL-agent mode) — used to apply job-priority
        /// selection among simultaneously-ready routing jobs. Optional.</param>
        public void Initialize(
            JobStore jobs,
            FactoryLayoutManager layout,
            Func<double> getSimTime,
            Func<int> getDecisionCount,
            Action incrementDecisionCount,
            Func<int> getBaselineActionIndex = null)
        {
            _jobs = jobs;
            _layout = layout;
            _getSimTime = getSimTime;
            _getDecisionCount = getDecisionCount;
            _incrementDecisionCount = incrementDecisionCount;
            _getBaselineActionIndex = getBaselineActionIndex;
        }

        /// <summary>
        /// Determines the next decision to be made by the simulation. Prioritizes routing decisions
        /// for jobs needing machine assignment, then checks for dispatch decisions for idle machines
        /// with dispatchable jobs.
        /// </summary>
        /// <returns>
        /// A DecisionRequest if a decision is needed, or null if no decisions are required
        /// (e.g., all eligible machines are unavailable and no idle machines have dispatchable jobs).
        /// </returns>
        public DecisionRequest FindNextDecision()
        {
            List<int> readyIds = _jobs.GetAllNeedingRouting();
            if (readyIds.Count > 0)
            {
                var routableIds = new List<int>();
                foreach (int jobId in readyIds)
                {
                    JobData job = _jobs.Get(jobId);
                    var eligibleIds = new HashSet<int>(job.EligibleMachinesPerOp[job.CurrentOpIndex].Keys);

                    bool anyAvailable = _layout.Machines
                        .Any(m => eligibleIds.Contains(m.MachineId) && m.IsAvailableForWork);

                    if (!anyAvailable)
                    {
                        _jobs.DeferredJobIds.Add(jobId);
                        SimLogger.Low($"[Orchestrator] Job {jobId}: all eligible machines " +
                                      $"are Failed/Repairing. Deferring routing decision.");
                    }
                    else
                    {
                        routableIds.Add(jobId);
                    }
                }

                if (routableIds.Count > 0)
                {
                    int chosenJobId = SelectRoutingJobId(routableIds);
                    DecisionRequest decision = BuildRoutingDecision(_jobs.Get(chosenJobId));
                    decision.JobCandidateIds = routableIds.ToArray();
                    return decision;
                }
            }

            foreach (var machine in _layout.Machines)
            {
                if (machine.IsIdle && machine.IsAvailableForWork && _jobs.HasDispatchableJob(machine.MachineId))
                {
                    return BuildDispatchDecision(machine.MachineId);
                }
            }

            return null;
        }

        /// <summary>
        /// Picks which of several simultaneously-routable jobs gets this routing decision, using
        /// job-priority scoring (DispatchingEngine.SelectRoutingJob) when a baseline rule is
        /// known; falls back to first-in-list (previous FIFO behaviour, unchanged) in RL-agent
        /// mode or when only one job is routable (no real choice either way).
        /// </summary>
        private int SelectRoutingJobId(List<int> routableIds)
        {
            if (routableIds.Count == 1) return routableIds[0];

            int actionIndex = _getBaselineActionIndex?.Invoke() ?? -1;
            if (actionIndex < 0) return routableIds[0];

            return DispatchingEngine.SelectRoutingJob(actionIndex, routableIds, _jobs, _getSimTime());
        }

        /// <summary>
        /// Builds a routing decision request for a job that needs to be assigned to an eligible machine.
        /// Filters machines based on eligibility and availability, and populates the decision request
        /// with candidate information including queue lengths and processing times.
        /// </summary>
        /// <param name="job">The job data requiring routing decision.</param>
        /// <returns>
        /// A DecisionRequest containing routing options with candidate machine IDs, queue lengths,
        /// and processing times for the specified job.
        /// </returns>
        public DecisionRequest BuildRoutingDecision(JobData job)
        {
            var eligibleIds = new HashSet<int>(
                job.EligibleMachinesPerOp[job.CurrentOpIndex].Keys);

            var candidates = _layout.Machines
                .Where(m => eligibleIds.Contains(m.MachineId) && m.IsAvailableForWork)
                .Select(m => m.MachineId)
                .ToList();

            int currentDecisionCount = _getDecisionCount();
            _incrementDecisionCount();

            return new DecisionRequest
            {
                Type = DecisionType.Routing,
                SimTime = _getSimTime(),
                DecisionIndex = currentDecisionCount,
                TotalJobs = _jobs.JobCount,
                CompletedJobs = _jobs.CountInState(JobState.Exited),
                JobId = job.JobId,
                SourceMachineId = job.LocationMachineId,
                RequiredType = job.NextRequiredType,
                CandidateMachineIds = candidates.ToArray(),
                CandidateQueueLengths = candidates.Select(id => _jobs.GetMachineLoad(id)).ToArray(),
                CandidateJobTimes = candidates.Select(id => job.GetProcessingTime(id)).ToArray(),
            };
        }

        /// <summary>
        /// Builds a dispatch decision request for a machine that has jobs waiting in its queue.
        /// Retrieves all dispatchable jobs for the specified machine and populates the decision
        /// request with job IDs and their corresponding processing durations.
        /// </summary>
        /// <param name="machineId">The ID of the machine requiring a dispatch decision.</param>
        /// <returns>
        /// A DecisionRequest containing the machine ID, queued job IDs, and their processing
        /// durations for dispatch priority determination.
        /// </returns>
        public DecisionRequest BuildDispatchDecision(int machineId)
        {
            List<int> queue = _jobs.GetDispatchableJobs(machineId);

            int currentDecisionCount = _getDecisionCount();
            _incrementDecisionCount();

            return new DecisionRequest
            {
                Type = DecisionType.Dispatch,
                MachineId = machineId,
                SimTime = _getSimTime(),
                DecisionIndex = currentDecisionCount,
                TotalJobs = _jobs.JobCount,
                CompletedJobs = _jobs.CountInState(JobState.Exited),
                QueuedJobIds = queue.ToArray(),
                QueuedDurations = queue.Select(id => (double)_jobs.GetProcessingTime(id, machineId)).ToArray(),
            };
        }
    }
}