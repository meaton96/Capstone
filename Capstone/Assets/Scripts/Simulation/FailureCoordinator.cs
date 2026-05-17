using System;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.AGV;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Simulation.Stochastic;
using Assets.Scripts.Simulation.Logging;
using System.Collections.Generic;

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
                    HandleMachineRepairComplete(machine);
            }
        }

        private void HandleMachineFailure(PhysicalMachine machine)
        {
            int machineId = machine.MachineId;
            SimLogger.Low($"[Orchestrator] Machine {machineId} FAILED. " +
                          $"RepairTime={machine.SampledRepairDuration:F1}s");

            _machineProcessingStartTime.Remove(machineId);
            _tracker.RecordMachineFailure(machineId, machine.SampledRepairDuration, _simTimeRef);

            if (machine.ActiveJobId >= 0)
            {
                JobData processingJob = _jobs.Get(machine.ActiveJobId);
                if (processingJob != null && processingJob.State == JobState.Processing)
                {
                    processingJob.State = JobState.NeedsRouting;
                    processingJob.LocationMachineId = machineId;
                    processingJob.StateEntryTime = _simTimeRef;
                    SimLogger.Low($"[Orchestrator] Job {processingJob.JobId} returned to " +
                                  $"NeedsRouting (was Processing on failed machine {machineId}).");
                }
            }

            foreach (var job in _jobs.AllJobs)
            {
                if (job.LocationMachineId == machineId && job.State == JobState.Queued)
                {
                    job.State = JobState.NeedsRouting;
                    job.StateEntryTime = _simTimeRef;
                    SimLogger.Low($"[Orchestrator] Queued job {job.JobId} re-routed " +
                                  $"from failed machine {machineId}.");
                }
            }

            foreach (var agv in _agvPool.AllAGVs)
            {
                int agvJobId = agv.CurrentJobId;
                if (agvJobId < 0) continue;

                JobData transitJob = _jobs.Get(agvJobId);
                if (transitJob == null) continue;
                if (transitJob.State != JobState.InTransit) continue;
                if (transitJob.TargetMachineId != machineId) continue;

                transitJob.State = JobState.NeedsRouting;
                transitJob.TargetMachineId = -1;
                transitJob.AssignedAgvId = -1;
                transitJob.StateEntryTime = _simTimeRef;

                SimLogger.Low($"[Orchestrator] AGV {agv.AgvId} carrying job {agvJobId} " +
                              $"re-routed: destination machine {machineId} has failed.");
            }

            foreach (var job in _jobs.AllJobs)
            {
                if (job.PreDispatchedAgvId < 0) continue;
                if (job.TargetMachineId != machineId) continue;

                SimLogger.Low($"[Orchestrator] Pre-dispatch for job {job.JobId} to " +
                              $"machine {machineId} cancelled.");
                job.PreDispatchedAgvId = -1;
            }

            machine.AcknowledgeFailure();
            _refreshLabels?.Invoke(machineId);
            _onMachineFailedInvalidateDecision?.Invoke(machineId);
        }

        private void HandleMachineRepairComplete(PhysicalMachine machine)
        {
            SimLogger.Low($"[Orchestrator] Machine {machine.MachineId} repair complete — " +
                          $"returning to OPERATIONAL.");

            machine.AcknowledgeRepairComplete();
            _tracker.RecordRepairComplete(machine.MachineId, _simTimeRef);
            _refreshLabels?.Invoke(machine.MachineId);
        }
    }
}