using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.AGV;
using Assets.Scripts.Simulation.FactoryLayout;

namespace Assets.Scripts.Simulation
{
    public class FlagHarvester
    {
        private JobStore _jobs;
        private AGVPool _agvPool;
        private FactoryLayoutManager _layout;
        private EpisodeTracker _tracker;
        private Dictionary<int, double> _machineProcessingStartTime;
        private double _simTimeRef;

        public void Initialize(
            JobStore jobs,
            AGVPool agvPool,
            FactoryLayoutManager layout,
            EpisodeTracker tracker,
            Dictionary<int, double> processingStartTimes)
        {
            _jobs = jobs;
            _agvPool = agvPool;
            _layout = layout;
            _tracker = tracker;
            _machineProcessingStartTime = processingStartTimes;
        }

        public void SetSimTime(double simTime)
        {
            _simTimeRef = simTime;
        }

        public void HarvestMachineFlags()
        {
            foreach (var machine in _layout.Machines)
            {
                if (!machine.FinishedFlag) continue;

                int jobId = machine.ActiveJobId;
                int mid = machine.MachineId;
                machine.ClearFinished();

                if (_machineProcessingStartTime.TryGetValue(mid, out double procStart))
                {
                    _tracker.AddProcessingTime(mid, _simTimeRef - procStart);
                    _machineProcessingStartTime.Remove(mid);
                }
                _tracker.RecordOperationComplete(mid);

                JobData job = _jobs.Get(jobId);
                if (job == null) continue;

                job.CompletedOps++;
                if (job.CurrentOpIndex < job.TotalOperations)
                    job.CurrentOpIndex++;

                machine.PlaceOnOutgoing(jobId, job.Visual);
                RefreshMachineLabels(mid);

                if (job.IsLastOperation)
                {
                    job.State = JobState.WaitingForPickup;
                    job.TargetMachineId = -1;
                    job.LocationMachineId = mid;
                    job.StateEntryTime = _simTimeRef;

                    if (job.PreDispatchedAgvId >= 0)
                    {
                        AGVController preAgv = _agvPool.GetPreDispatchedAGV(job.JobId);
                        if (preAgv != null)
                        {
                            preAgv.FinalizePreDispatch(job.JobId, _layout.OutgoingBeltPosition, null, job.Visual);
                            job.AssignedAgvId = preAgv.AgvId;
                        }
                        job.PreDispatchedAgvId = -1;
                    }
                }
                else
                {
                    job.State = JobState.NeedsRouting;
                    job.LocationMachineId = mid;
                    job.StateEntryTime = _simTimeRef;
                }
            }
        }

        public void HarvestAlmostDoneFlags(int preDispatchLeadTime)
        {
            foreach (var machine in _layout.Machines)
            {
                if (!machine.AlmostDoneFlag) continue;

                int jobId = machine.AlmostDoneJobId;
                machine.ClearAlmostDone();

                JobData job = _jobs.Get(jobId);
                if (job == null || job.State != JobState.Processing || job.PreDispatchedAgvId >= 0) continue;
                if (job.CompletedOps == job.TotalOperations - 1) continue;

                AGVController agv = _agvPool.GetAvailableAGV();
                if (agv == null) continue;

                agv.PreDispatch(jobId, machine.GetPickupPosition(), machine);
                job.PreDispatchedAgvId = agv.AgvId;
            }
        }

        public void HarvestAGVFlags()
        {
            foreach (var agv in _agvPool.AllAGVs)
            {
                if (agv.PickedUpFlag)
                {
                    JobData job = _jobs.Get(agv.CurrentJobId);
                    if (job != null && job.State == JobState.WaitingForPickup)
                    {
                        job.State = JobState.InTransit;
                        job.StateEntryTime = _simTimeRef;
                    }
                }

                if (agv.DeliveredFlag)
                {
                    int jobId = agv.DeliveredJobId;
                    int machineId = agv.DeliveredMachineId;
                    JobData job = _jobs.Get(jobId);

                    if (job != null)
                    {
                        if (machineId < 0)
                        {
                            job.State = JobState.Exited;
                            job.LocationMachineId = -1;
                            job.StateEntryTime = _simTimeRef;
                            if (job.Visual != null) job.Visual.gameObject.SetActive(false);
                        }
                        else
                        {
                            job.State = JobState.Queued;
                            job.LocationMachineId = machineId;
                            job.StateEntryTime = _simTimeRef;

                            PhysicalMachine targetMachine = _layout.GetMachine(machineId);
                            targetMachine.PlaceOnIncoming(jobId, job.Visual);
                            RefreshMachineLabels(machineId);
                        }
                        job.TotalTransitTime += (_simTimeRef - job.StateEntryTime);
                        job.AssignedAgvId = -1;
                    }
                }

                if (agv.PickedUpFlag || agv.DeliveredFlag)
                    agv.ClearFlags();
            }
        }

        public void AssignAGVs()
        {
            var candidates = new List<JobData>();
            foreach (var job in _jobs.AllJobs)
            {
                if (job.State == JobState.WaitingForPickup
                    && job.AssignedAgvId == -1
                    && job.PreDispatchedAgvId < 0)
                    candidates.Add(job);
            }

            foreach (var job in candidates)
            {
                AGVController agv = _agvPool.GetAvailableAGV();
                if (agv == null) break;

                PhysicalMachine src = job.LocationMachineId >= 0
                    ? _layout.GetMachine(job.LocationMachineId) : null;
                Vector3 pickupPos = src != null
                    ? src.GetPickupPosition() : _layout.IncomingBeltPosition;

                PhysicalMachine dst = job.TargetMachineId >= 0
                    ? _layout.GetMachine(job.TargetMachineId) : null;
                Vector3 dropoffPos = dst != null
                    ? dst.GetDropoffPosition() : _layout.OutgoingBeltPosition;

                job.AssignedAgvId = agv.AgvId;
                agv.Dispatch(job.JobId, pickupPos, dropoffPos, src, dst, job.Visual);
                agv.SetCarryVisual(job.Visual);
            }
        }

        public void RefreshMachineLabels(int machineId)
        {
            PhysicalMachine machine = _layout.GetMachine(machineId);
            if (machine == null) return;
            int inCount = _jobs.GetDispatchableJobs(machineId).Count;
            int outCount = _jobs.AllJobs.Count(j =>
                j.LocationMachineId == machineId && j.State == JobState.WaitingForPickup);
            machine.RefreshQueueLabels(inCount, outCount);
        }
    }
}