using System;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.AGV;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Simulation.Stochastic;
using Assets.Scripts.Simulation.Logging;
using System.Collections.Generic;
using System.Linq;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// Coordinates failure events across the simulation, handling machine failures,
    /// repairs, and the resulting job re-routing decisions.
    /// </summary>
    public class FailureCoordinator
    {
        /// <summary>Job store for accessing and managing job data.</summary>
        private JobStore _jobs;
        /// <summary>Pool of AGVs used for dispatching and transit operations.</summary>
        private AGVPool _agvPool;
        /// <summary>Factory layout manager for machine access and health state.</summary>
        private FactoryLayoutManager _layout;
        /// <summary>Episode tracker for recording simulation events.</summary>
        private EpisodeTracker _tracker;
        /// <summary>Mapping of machine IDs to their processing start times.</summary>
        private Dictionary<int, double> _machineProcessingStartTime;
        /// <summary>Callback invoked when a machine failure requires invalidating pending decisions.</summary>
        private Action<int> _onMachineFailedInvalidateDecision;
        /// <summary>Callback invoked to refresh visual labels for a specified machine.</summary>
        private Action<int> _refreshLabels;
        /// <summary>Current simulation time reference used for event tracking.</summary>
        private double _simTimeRef;

        /// <summary>
        /// Initializes the FailureCoordinator with required dependencies and callbacks.
        /// </summary>
        /// <param name="jobs">Job store for accessing and managing job data.</param>
        /// <param name="agvPool">Pool of AGVs used for dispatching and transit operations.</param>
        /// <param name="layout">Factory layout manager for machine access and health state.</param>
        /// <param name="tracker">Episode tracker for recording simulation events.</param>
        /// <param name="machineProcessingStartTime">Mapping of machine IDs to their processing start times.</param>
        /// <param name="onMachineFailedInvalidateDecision">Callback invoked when a machine failure requires invalidating pending decisions.</param>
        /// <param name="refreshLabels">Callback invoked to refresh visual labels for a specified machine.</param>
        public void Initialize(
            JobStore jobs,
            AGVPool agvPool,
            FactoryLayoutManager layout,
            EpisodeTracker tracker,
            Dictionary<int, double> machineProcessingStartTime,
            Action<int> onMachineFailedInvalidateDecision,
            Action<int> refreshLabels)
        {
            _jobs = jobs;
            _agvPool = agvPool;
            _layout = layout;
            _tracker = tracker;
            _machineProcessingStartTime = machineProcessingStartTime;
            _onMachineFailedInvalidateDecision = onMachineFailedInvalidateDecision;
            _refreshLabels = refreshLabels;
        }

        /// <summary>
        /// Sets the current simulation time reference used for event tracking.
        /// </summary>
        /// <param name="simTime">The current simulation time.</param>
        public void SetSimTime(double simTime)
        {
            _simTimeRef = simTime;
        }

        /// <summary>
        /// Checks all machines for failure or repair-complete flags and handles them accordingly.
        /// Only processes events when stochastic event management is enabled.
        /// </summary>
        public void HarvestFailureFlags()
        {
            if (StochasticEventManager.Instance == null ||
                !StochasticEventManager.Instance.MachineFailuresEnabled)
                return;

            foreach (var machine in _layout.Machines)
            {
                if (machine.FailedFlag)
                    HandleMachineFailure(machine);
                else if (machine.RepairCompleteFlag)
                {
                    HandleMachineRepairComplete(machine);
                    RetryDeferredJobs();
                }

            }
        }
        /// <summary>
        /// Re-evaluates deferred jobs and re-queues those whose eligible machines are now operational
        /// after a repair completion.
        /// </summary>
        private void RetryDeferredJobs()
        {
            foreach (int jobId in _jobs.DeferredJobIds.ToList())
            {
                JobData job = _jobs.Get(jobId);
                if (job == null) { _jobs.DeferredJobIds.Remove(jobId); continue; }

                bool anyOperational = job.EligibleMachinesPerOp[job.CurrentOpIndex]
                    .Keys.Any(mid => _layout.GetMachine(mid)?.HealthState
                                     == MachineHealthState.Operational);
                if (anyOperational)
                {
                    _jobs.DeferredJobIds.Remove(jobId);
                    // Re-queue as NeedsRouting so the next Update() tick picks it up
                    job.State = JobState.NeedsRouting;
                    SimLogger.Low($"[Failure] Deferred job {jobId} re-queued after repair.");
                }
            }
        }

        /// <summary>
        /// Handles all consequences of a machine failure, including re-routing active jobs,
        /// canceling pickups and transits, and redirecting or aborting in-flight AGVs.
        /// </summary>
        /// <param name="machine">The physical machine that has failed.</param>
        private void HandleMachineFailure(PhysicalMachine machine)
        {
            int machineId = machine.MachineId;
            SimLogger.Medium($"[Orchestrator] Machine {machineId} FAILED. RepairTime={machine.SampledRepairDuration:F1}s");

            _machineProcessingStartTime.Remove(machineId);
            _tracker.RecordMachineFailure(machineId, machine.SampledRepairDuration, _simTimeRef);

            // Step 1: Handle job actively being processed on this machine
            if (machine.ActiveJobId >= 0)
            {
                JobData processingJob = _jobs.Get(machine.ActiveJobId);
                if (processingJob != null && processingJob.State == JobState.Processing)
                {
                    processingJob.TransitionTo(JobState.NeedsRouting, _simTimeRef);
                    processingJob.LocationMachineId = machineId;
                    SimLogger.Medium($"[Failure] Job {processingJob.JobId} returned to NeedsRouting " +
                                  $"(was Processing on failed machine {machineId}).");
                }
            }

            // Step 2: Handle jobs physically queued at this machine
            foreach (var job in _jobs.AllJobs)
            {
                if (job.State != JobState.Queued) continue;
                if (job.LocationMachineId != machineId) continue;

                job.TransitionTo(JobState.NeedsRouting, _simTimeRef);
                SimLogger.Medium($"[Failure] Queued job {job.JobId} re-routed from failed machine {machineId}.");
            }

            // Step 3: Cancel WaitingForPickup jobs whose target machine failed
            foreach (var agv in _agvPool.AllAGVs)
            {
                if (agv.State != AGVState.MovingToPickup) continue;

                int agvJobId = agv.CurrentJobId;
                if (agvJobId < 0) continue;

                JobData waitingJob = _jobs.Get(agvJobId);
                if (waitingJob == null) continue;
                if (waitingJob.State != JobState.WaitingForPickup) continue;
                if (waitingJob.TargetMachineId != machineId) continue;

                agv.CancelPickup();
                waitingJob.TransitionTo(JobState.NeedsRouting, _simTimeRef);
                waitingJob.TargetMachineId = -1;
                waitingJob.AssignedAgvId = -1;
                SimLogger.Medium($"[Failure] WaitingForPickup job {waitingJob.JobId}: " +
                              $"AGV {agv.AgvId} pickup cancelled, job re-routed.");
            }

            // Step 4: Redirect or abort in-transit AGVs headed to the failed machine
            foreach (var agv in _agvPool.AllAGVs)
            {
                if (agv.State != AGVState.MovingToDropoff) continue;

                int agvJobId = agv.CurrentJobId;
                if (agvJobId < 0) continue;

                JobData transitJob = _jobs.Get(agvJobId);
                if (transitJob == null) continue;
                if (transitJob.State != JobState.InTransit) continue;
                if (transitJob.TargetMachineId != machineId) continue;

                PhysicalMachine alternate = FindBestAlternateMachine(transitJob, machineId);
                if (alternate != null)
                {
                    // Job stays InTransit — only the destination changes, so StateEntryTime
                    // (and the TimeInTransit bucket it feeds) must NOT reset here.
                    transitJob.TargetMachineId = alternate.MachineId;
                    // AssignedAgvId preserved — the AGV still owns this job

                    agv.RedirectDropoff(alternate.GetDropoffPosition(), alternate, transitJob.Visual);
                    SimLogger.Medium($"[Failure] AGV {agv.AgvId} carrying job {agvJobId} " +
                                  $"redirected: machine {machineId} failed → machine {alternate.MachineId}.");
                }
                else
                {
                    // No operational alternate available — abort transit and re-enter job pool
                    transitJob.TransitionTo(JobState.NeedsRouting, _simTimeRef);
                    transitJob.TargetMachineId = -1;
                    transitJob.AssignedAgvId = -1;

                    agv.AbortTransit();
                    SimLogger.Medium($"[Failure] AGV {agv.AgvId} carrying job {agvJobId}: " +
                                  $"no alternate found, transit aborted.");
                }
            }

            // Step 5: Cancel pre-dispatches aimed at the failed machine
            foreach (var job in _jobs.AllJobs)
            {
                if (job.PreDispatchedAgvId < 0) continue;
                if (job.TargetMachineId != machineId) continue;

                SimLogger.Medium($"[Failure] Pre-dispatch for job {job.JobId} to machine {machineId} cancelled.");
                job.PreDispatchedAgvId = -1;
            }

            machine.AcknowledgeFailure();
            _refreshLabels?.Invoke(machineId);
            _onMachineFailedInvalidateDecision?.Invoke(machineId);
        }
        /// <summary>
        /// Finds the best operational alternate machine for a job, excluding a specified machine.
        /// Selects the machine with the lowest current load (sum of processing times of
        /// queued and committed in-transit jobs).
        /// </summary>
        /// <param name="job">The job requiring an alternate machine.</param>
        /// <param name="excludeMachineId">The machine ID to exclude (typically the failed machine).</param>
        /// <returns>The best alternate PhysicalMachine, or null if none is available.</returns>
        private PhysicalMachine FindBestAlternateMachine(JobData job, int excludeMachineId)
        {
            if (job.CurrentOpIndex < 0 || job.CurrentOpIndex >= job.EligibleMachinesPerOp.Length)
                return null;

            var eligible = job.EligibleMachinesPerOp[job.CurrentOpIndex];

            PhysicalMachine best = null;
            float bestLoad = float.MaxValue;

            foreach (var kvp in eligible)
            {
                int candidateId = kvp.Key;
                if (candidateId == excludeMachineId) continue;

                PhysicalMachine candidate = _layout.GetMachine(candidateId);
                if (candidate == null) continue;
                if (candidate.HealthState != MachineHealthState.Operational) continue;

                // GetMachineLoad sums processing times of queued + committed in-transit jobs —
                // a better proxy than raw job count since operation durations vary
                float load = _jobs.GetMachineLoad(candidateId);
                if (load < bestLoad)
                {
                    bestLoad = load;
                    best = candidate;
                }
            }

            return best;
        }
        private void HandleMachineRepairComplete(PhysicalMachine machine)
        {
            SimLogger.Medium($"[Orchestrator] Machine {machine.MachineId} repair complete — " +
                          $"returning to OPERATIONAL.");

            machine.AcknowledgeRepairComplete();
            _tracker.RecordRepairComplete(machine.MachineId, _simTimeRef);
            _refreshLabels?.Invoke(machine.MachineId);
        }
    }
}