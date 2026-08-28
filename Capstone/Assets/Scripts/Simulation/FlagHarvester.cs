using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.AGV;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Simulation.Logging;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// FlagHarvester coordinates the flow of jobs through the factory simulation by harvesting state-change
    /// flags from machines, AGVs, and the job scheduler. It orchestrates job state transitions, AGV dispatching,
    /// pre-dispatch logic, and timing statistics collection.
    /// </summary>
    public class FlagHarvester
    {
        /// <summary>Reference to the job store managing all job data.</summary>
        private JobStore _jobs;
        /// <summary>Reference to the AGV pool for vehicle allocation and management.</summary>
        private AGVPool _agvPool;
        /// <summary>Reference to the factory layout manager for machine and position queries.</summary>
        private FactoryLayoutManager _layout;
        /// <summary>Reference to the episode tracker for recording timing and operation statistics.</summary>
        private EpisodeTracker _tracker;
        /// <summary>Maps machine IDs to the simulation time when processing started, used for tracking processing durations.</summary>
        private Dictionary<int, double> _machineProcessingStartTime;
        /// <summary>The current simulation time reference used for timing calculations.</summary>
        private double _simTimeRef;

        /// <summary>
        /// Initializes the FlagHarvester with required dependencies. Must be called before any Harvest methods.
        /// </summary>
        /// <param name="jobs">The job store containing all job data and state.</param>
        /// <param name="agvPool">The AGV pool for vehicle allocation and management.</param>
        /// <param name="layout">The factory layout manager for machine and position queries.</param>
        /// <param name="tracker">The episode tracker for recording timing and operation statistics.</param>
        /// <param name="processingStartTimes">Dictionary mapping machine IDs to their job processing start times.</param>
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

        /// <summary>
        /// Sets the current simulation time reference used for timing calculations throughout the simulation step.
        /// </summary>
        /// <param name="simTime">The current simulation time.</param>
        public void SetSimTime(double simTime)
        {
            _simTimeRef = simTime;
        }

        /// <summary>
        /// Harvests completion flags from all machines. For each machine that has finished processing a job,
        /// this method updates job state, records processing times, places the job on the outgoing belt,
        /// and handles pre-dispatched AGV assignments if the job is complete.
        /// </summary>
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

                // CurrentOpIndex still refers to the op that just finished — stamp before advancing.
                job.OpProcEndTimes[job.CurrentOpIndex] = (float)_simTimeRef;

                job.CompletedOps++;
                if (job.CurrentOpIndex < job.TotalOperations)
                    job.CurrentOpIndex++;

                machine.PlaceOnOutgoing(jobId, job.Visual);
                RefreshMachineLabels(mid);

                if (job.IsLastOperation)
                {
                    job.TransitionTo(JobState.WaitingForPickup, _simTimeRef);
                    job.TargetMachineId = -1;
                    job.LocationMachineId = mid;

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
                    job.TransitionTo(JobState.NeedsRouting, _simTimeRef);
                    job.LocationMachineId = mid;
                }
            }
        }

        /// <summary>
        /// Harvests "almost done" flags from machines and initiates AGV pre-dispatch for jobs that are
        /// nearing completion. Pre-dispatching allows AGVs to be assigned before a job fully finishes,
        /// reducing wait times. Machines that are not in an operational health state are skipped.
        /// </summary>
        /// <param name="preDispatchLeadTime">The lead time threshold for triggering pre-dispatch.</param>
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

                // Skip pre-dispatch if the source machine is not operational
                if (machine.HealthState != MachineHealthState.Operational) continue;

                AGVController agv = _agvPool.GetNearestAvailableAGV(machine, machine.GetPickupPosition());
                if (agv == null) continue;

                agv.PreDispatch(jobId, machine.GetPickupPosition(), machine);
                job.PreDispatchedAgvId = agv.AgvId;
            }
        }

        /// <summary>
        /// Harvests state-change flags from all AGVs. Handles pickup events (transitioning jobs to InTransit)
        /// and delivery events (transitioning jobs to Queued or Exited state). Updates job timing statistics
        /// and places delivered jobs on the target machine's incoming belt or marks them as exited.
        /// </summary>
        public void HarvestAGVFlags()
        {
            foreach (var agv in _agvPool.AllAGVs)
            {
                if (agv.PickedUpFlag)
                {
                    JobData job = _jobs.Get(agv.CurrentJobId);
                    if (job != null && job.State == JobState.WaitingForPickup)
                    {
                        job.TransitionTo(JobState.InTransit, _simTimeRef);
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
                            // Job has exited the system entirely
                            job.TransitionTo(JobState.Exited, _simTimeRef);
                            job.ExitTime = (float)_simTimeRef;
                            _tracker.RecordJobExit();
                            SimLogger.Medium($"Job {job.JobId} exited the system after delivery.");
                            job.LocationMachineId = -1;
                            if (job.Visual != null) job.Visual.gameObject.SetActive(false);
                        }
                        else
                        {
                            // Job is delivered to a target machine — stamp per-op travel time
                            job.OperationTravelTimes[job.CurrentOpIndex] = agv.LastTripDuration;
                            job.OpQueueEntryTimes[job.CurrentOpIndex] = (float)_simTimeRef;
                            job.TransitionTo(JobState.Queued, _simTimeRef);
                            job.LocationMachineId = machineId;

                            PhysicalMachine targetMachine = _layout.GetMachine(machineId);
                            targetMachine.PlaceOnIncoming(jobId, job.Visual);
                            RefreshMachineLabels(machineId);
                        }
                        job.AssignedAgvId = -1;
                    }
                }

                if (agv.PickedUpFlag || agv.DeliveredFlag)
                    agv.ClearFlags();
            }
        }

        /// <summary>
        /// Handles AGVs that self-recovered from a suspected traffic-zone deadlock
        /// (AGVController.HandleZoneStall). The AGV has already released its own route and
        /// zone reservations and begun returning to parking; this owns the JobData side —
        /// clearing the job's link to that AGV and, if the job was physically committed to it
        /// (WaitingForPickup or InTransit), returning it to NeedsRouting so a fresh AGV can be
        /// assigned. A job that was only pre-dispatched (still Processing at its source
        /// machine, never picked up) is left alone — only its PreDispatchedAgvId claim is
        /// cleared, mirroring how HandleZoneStall itself distinguishes the two cases.
        /// </summary>
        public void HarvestStalledAGVs()
        {
            foreach (var agv in _agvPool.AllAGVs)
            {
                if (!agv.StalledFlag) continue;

                JobData job = _jobs.Get(agv.StalledJobId);
                if (job != null)
                {
                    if (job.PreDispatchedAgvId == agv.AgvId) job.PreDispatchedAgvId = -1;

                    if (job.State == JobState.WaitingForPickup || job.State == JobState.InTransit)
                    {
                        job.TransitionTo(JobState.NeedsRouting, _simTimeRef);
                        job.TargetMachineId = -1;
                        job.AssignedAgvId = -1;
                        SimLogger.Medium($"[FlagHarvester] Job {job.JobId} returned to NeedsRouting — " +
                                          $"AGV {agv.AgvId} stalled on a traffic-zone reservation.");
                    }
                }

                agv.ClearFlags();
            }
        }

        /// <summary>
        /// Assigns available AGVs to jobs that are waiting for pickup and have not yet been assigned one.
        /// Jobs are matched with available AGVs based on their location and target destination. If a job's
        /// target machine is non-operational, the job is returned to NeedsRouting state. AGVs are consumed
        /// in candidate order until none remain available.
        /// </summary>
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
                // Return job to NeedsRouting if the routed destination machine is non-operational
                if (job.TargetMachineId >= 0)
                {
                    PhysicalMachine dst = _layout.GetMachine(job.TargetMachineId);
                    if (dst != null && dst.HealthState != MachineHealthState.Operational)
                    {
                        job.TransitionTo(JobState.NeedsRouting, _simTimeRef);
                        job.TargetMachineId = -1;
                        SimLogger.Low($"[FlagHarvester] Job {job.JobId} returned to NeedsRouting — " +
                                      $"target machine {dst.MachineId} is {dst.HealthState}.");
                        continue;
                    }
                }

                PhysicalMachine src = job.LocationMachineId >= 0
                    ? _layout.GetMachine(job.LocationMachineId) : null;
                Vector3 pickupPos = src != null
                    ? src.GetPickupPosition() : _layout.IncomingBeltPosition;

                AGVController agv = _agvPool.GetNearestAvailableAGV(src, pickupPos);
                if (agv == null) break;

                PhysicalMachine target = job.TargetMachineId >= 0
                    ? _layout.GetMachine(job.TargetMachineId) : null;
                Vector3 dropoffPos = target != null
                    ? target.GetDropoffPosition() : _layout.OutgoingBeltPosition;

                job.AssignedAgvId = agv.AgvId;
                agv.Dispatch(job.JobId, pickupPos, dropoffPos, src, target, job.Visual);
                agv.SetCarryVisual(job.Visual);
            }
        }

        /// <summary>
        /// Refreshes the queue count labels on a specified machine based on current job states.
        /// Updates both the incoming queue count (dispatchable jobs waiting at the machine)
        /// and the outgoing queue count (jobs waiting for AGV pickup at the machine).
        /// </summary>
        /// <param name="machineId">The ID of the machine whose labels should be refreshed.</param>
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