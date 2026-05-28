using System;
using System.Collections.Generic;
using System.Linq;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Logging;

namespace Assets.Scripts.Simulation
{
    public class DecisionCoordinator
    {
        private JobStore _jobs;
        private FactoryLayoutManager _layout;
        private Func<double> _getSimTime;
        private Func<int> _getDecisionCount;
        private Action _incrementDecisionCount;

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