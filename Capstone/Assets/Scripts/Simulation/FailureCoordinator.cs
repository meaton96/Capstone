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
    public class FailureCoordinator
    {
        private JobStore _jobs;
        private AGVPool _agvPool;
        private FactoryLayoutManager _layout;
        private EpisodeTracker _tracker;
        private Dictionary<int, double> _machineProcessingStartTime;
        private Action<int> _onMachineFailedInvalidateDecision;
        private Action<int> _refreshLabels;
        private double _simTimeRef;


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

        public void SetSimTime(double simTime)
        {
            _simTimeRef = simTime;
        }

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

        private void HandleMachineFailure(PhysicalMachine machine)
        {
            int machineId = machine.MachineId;
            SimLogger.Medium($"[Orchestrator] Machine {machineId} FAILED. RepairTime={machine.SampledRepairDuration:F1}s");

            _machineProcessingStartTime.Remove(machineId);
            _tracker.RecordMachineFailure(machineId, machine.SampledRepairDuration, _simTimeRef);

            // ── 1. Job actively being processed on this machine ──────────────────────
            if (machine.ActiveJobId >= 0)
            {
                JobData processingJob = _jobs.Get(machine.ActiveJobId);
                if (processingJob != null && processingJob.State == JobState.Processing)
                {
                    processingJob.State = JobState.NeedsRouting;
                    processingJob.LocationMachineId = machineId;
                    processingJob.StateEntryTime = _simTimeRef;
                    SimLogger.Medium($"[Failure] Job {processingJob.JobId} returned to NeedsRouting " +
                                  $"(was Processing on failed machine {machineId}).");
                }
            }

            // ── 2. Jobs physically queued at this machine ─────────────────────────────
            foreach (var job in _jobs.AllJobs)
            {
                if (job.State != JobState.Queued) continue;
                if (job.LocationMachineId != machineId) continue;

                job.State = JobState.NeedsRouting;
                job.StateEntryTime = _simTimeRef;
                SimLogger.Medium($"[Failure] Queued job {job.JobId} re-routed from failed machine {machineId}.");
            }

            // ── 3. WaitingForPickup jobs routed TO this machine ───────────────────────
            // The routing decision already assigned this machine but the AGV hasn't
            // picked the job up yet. We cancel the assigned AGV (if any) and
            // return the job to NeedsRouting so the agent re-routes it.
            foreach (var agv in _agvPool.AllAGVs)
            {
                if (agv.State != AGVState.MovingToPickup) continue;

                int agvJobId = agv.CurrentJobId;
                if (agvJobId < 0) continue;

                JobData waitingJob = _jobs.Get(agvJobId);
                if (waitingJob == null) continue;
                if (waitingJob.State != JobState.WaitingForPickup) continue;
                if (waitingJob.TargetMachineId != machineId) continue;

                agv.CancelPickup(); // AGV returns to parking, job gets re-dispatched later
                waitingJob.State = JobState.NeedsRouting;
                waitingJob.TargetMachineId = -1;
                waitingJob.AssignedAgvId = -1;
                waitingJob.StateEntryTime = _simTimeRef;
                SimLogger.Medium($"[Failure] WaitingForPickup job {waitingJob.JobId}: " +
                              $"AGV {agv.AgvId} pickup cancelled, job re-routed.");
            }

            // ── 4. In-transit jobs: redirect the carrying AGV immediately ────────────
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
                    transitJob.TargetMachineId = alternate.MachineId;
                    // AssignedAgvId intentionally preserved — AGV still owns this job
                    transitJob.StateEntryTime = _simTimeRef;

                    agv.RedirectDropoff(alternate.GetDropoffPosition(), alternate, transitJob.Visual);
                    SimLogger.Medium($"[Failure] AGV {agv.AgvId} carrying job {agvJobId} " +
                                  $"redirected: machine {machineId} failed → machine {alternate.MachineId}.");
                }
                else
                {
                    // No operational alternate right now — abort transit, job re-enters pool
                    transitJob.State = JobState.NeedsRouting;
                    transitJob.TargetMachineId = -1;
                    transitJob.AssignedAgvId = -1;
                    transitJob.StateEntryTime = _simTimeRef;

                    agv.AbortTransit();
                    SimLogger.Medium($"[Failure] AGV {agv.AgvId} carrying job {agvJobId}: " +
                                  $"no alternate found, transit aborted.");
                }
            }

            // ── 5. Cancel pre-dispatches aimed at this machine ────────────────────────
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
        private PhysicalMachine FindBestAlternateMachine(JobData job, int excludeMachineId)
        {
            // EligibleMachinesPerOp[currentOpIndex] is Dictionary<int machineId, float processingTime>
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