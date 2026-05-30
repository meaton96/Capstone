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
        /// Initializes the coordinator with required dependencies via dependency injection.
        /// </summary>
        /// <param name="jobs">The job store providing job data and state information.</param>
        /// <param name="layout">The factory layout manager providing machine information.</param>
        /// <param name="getSimTime">Delegate to retrieve the current simulation time.</param>
        /// <param name="getDecisionCount">Delegate to retrieve the current decision count.</param>
        /// <param name="incrementDecisionCount">Delegate to increment the decision counter.</param>
        public void Initialize(
            JobStore jobs,
            FactoryLayoutManager layout,
            Func<double> getSimTime,
            Func<int> getDecisionCount,
            Action incrementDecisionCount)
        {
            _jobs = jobs;
            _layout = layout;
            _getSimTime = getSimTime;
            _getDecisionCount = getDecisionCount;
            _incrementDecisionCount = incrementDecisionCount;
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
            JobData routingJob = _jobs.GetNextNeedsRouting();
            if (routingJob != null)
            {
                var eligibleIds = new HashSet<int>(
                    routingJob.EligibleMachinesPerOp[routingJob.CurrentOpIndex].Keys);

                bool anyAvailable = _layout.Machines
                    .Any(m => eligibleIds.Contains(m.MachineId) && m.IsAvailableForWork);

                if (!anyAvailable)
                {
                    _jobs.DeferredJobIds.Add(routingJob.JobId);
                    SimLogger.Low($"[Orchestrator] Job {routingJob.JobId}: all eligible machines " +
                                  $"are Failed/Repairing. Deferring routing decision.");
                }
                else
                {
                    return BuildRoutingDecision(routingJob);
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