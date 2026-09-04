using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Assets.Scripts.Simulation.Jobs
{
    /// <summary>
    /// Central repository for all active <see cref="JobData"/> instances in the simulation.
    /// </summary>
    /// <remarks>
    /// Acts as a passive data store that translates static job definitions into live
    /// runtime trackers. The orchestrator queries and modifies these instances directly
    /// to manage the factory lifecycle.
    /// </remarks>
    public class JobStore : MonoBehaviour
    {
        [Header("Visuals")]
        [SerializeField] private JobVisual jobVisualPrefab;
        [SerializeField] private Transform jobVisualContainer;

        private readonly List<JobData> allJobs = new List<JobData>();

        public IReadOnlyList<JobData> AllJobs => allJobs;
        public int JobCount => allJobs.Count;
        public readonly HashSet<int> DeferredJobIds = new HashSet<int>();

        public bool IsInitialized = false;

        /// <summary>
        /// Converts static job definitions into runtime state trackers.
        /// </summary>
        /// <param name="definitions">An enumerable of <see cref="FJSSPJobDefinition"/> blueprint data.</param>
        /// <param name="spawnVisuals">Whether to instantiate 3D visuals for each job.</param>
        /// <remarks>
        /// Clears existing data and populates the store with fresh <see cref="JobData"/>.
        /// If <c>spawnVisuals</c> is true, it links each data entry to a newly instantiated
        /// <see cref="JobVisual"/> within the designated container.
        /// </remarks>
        public void Initialize(IEnumerable<FJSSPJobDefinition> definitions, bool spawnVisuals)
        {
            allJobs.Clear();

            foreach (var def in definitions)
            {
                int opCount = def.OperationSequence.Length;
                var jobData = new JobData
                {
                    JobId = def.JobId,
                    ArrivalTime = def.ArrivalTime,
                    OperationTypes = def.OperationSequence,
                    EligibleMachinesPerOp = def.EligibleMachinesPerOp,
                    TotalOperations = opCount,
                    OperationTravelTimes = new float[opCount],
                    OpQueueEntryTimes = NewUnstampedArray(opCount),
                    OpProcStartTimes = NewUnstampedArray(opCount),
                    OpProcEndTimes = NewUnstampedArray(opCount),

                    State = JobState.NeedsRouting,
                    LocationMachineId = -1,
                    TargetMachineId = -1,
                    AssignedAgvId = -1,
                    CurrentOpIndex = 0,
                    CompletedOps = 0,
                    StateEntryTime = def.ArrivalTime,
                };

                if (spawnVisuals && jobVisualPrefab != null)
                {
                    JobVisual vis = Instantiate(jobVisualPrefab, jobVisualContainer);
                    vis.gameObject.name = $"JobVisual_{def.JobId}";
                    vis.Initialize(def.JobId, def.OperationSequence.Length);
                    jobData.Visual = vis;
                }

                allJobs.Add(jobData);
            }
            IsInitialized = true;
        }

        /// <summary>
        /// Inserts a single dynamically-arrived job into the store mid-episode.
        /// </summary>
        /// <param name="def">The job definition to instantiate as a live <see cref="JobData"/>.</param>
        /// <param name="spawnVisuals">Whether to instantiate a 3D visual for the job.</param>
        /// <returns>The new <see cref="JobData"/> instance, already in <c>NeedsRouting</c> state.</returns>
        /// <remarks>
        /// Mirrors the inner loop of <see cref="Initialize"/> for a single entry.
        /// Called by the orchestrator's Poisson arrival clock each time a new job is injected.
        /// The job is immediately visible to <see cref="GetNextNeedsRouting"/> and
        /// <see cref="AreAllExited"/>, so the existing decision and termination logic
        /// picks it up without any further changes.
        /// </remarks>
        public JobData AddDynamicJob(FJSSPJobDefinition def, bool spawnVisuals)
        {
            int opCount = def.OperationSequence.Length;
            var jobData = new JobData
            {
                JobId = def.JobId,
                ArrivalTime = def.ArrivalTime,
                OperationTypes = def.OperationSequence,
                EligibleMachinesPerOp = def.EligibleMachinesPerOp,
                TotalOperations = opCount,
                OperationTravelTimes = new float[opCount],
                OpQueueEntryTimes = NewUnstampedArray(opCount),
                OpProcStartTimes = NewUnstampedArray(opCount),
                OpProcEndTimes = NewUnstampedArray(opCount),
                State = JobState.NeedsRouting,
                LocationMachineId = -1,
                TargetMachineId = -1,
                AssignedAgvId = -1,
                CurrentOpIndex = 0,
                CompletedOps = 0,
                // Dynamic jobs enter mid-episode — anchor the clock at their actual arrival,
                // not 0, so their first TransitionTo() doesn't backfill time before they existed.
                StateEntryTime = def.ArrivalTime,
            };

            if (spawnVisuals && jobVisualPrefab != null)
            {
                JobVisual vis = Instantiate(jobVisualPrefab, jobVisualContainer);
                vis.gameObject.name = $"JobVisual_{def.JobId}";
                vis.Initialize(def.JobId, def.OperationSequence.Length);
                jobData.Visual = vis;
            }

            allJobs.Add(jobData);
            return jobData;
        }

        /// <summary>
        /// Allocates a per-operation timestamp array pre-filled with -1 (sentinel for "not yet reached").
        /// </summary>
        private static float[] NewUnstampedArray(int length)
        {
            var arr = new float[length];
            for (int i = 0; i < length; i++) arr[i] = -1f;
            return arr;
        }

        /// <summary>
        /// Destroys all associated <see cref="JobVisual"/> objects and clears the internal list.
        /// </summary>
        public void Cleanup()
        {
            foreach (var job in allJobs)
            {
                if (job.Visual != null)
                {
                    Destroy(job.Visual.gameObject);
                }
            }
            allJobs.Clear();
            IsInitialized = false;
        }

        /// <summary>
        /// Retrieves a specific <see cref="JobData"/> instance by its unique identifier.
        /// </summary>
        /// <param name="jobId">The unique ID of the job to retrieve.</param>
        /// <returns>The <see cref="JobData"/> instance if found; otherwise, <c>null</c>.</returns>
        public JobData Get(int jobId)
        {
            return allJobs.FirstOrDefault(j => j.JobId == jobId);
        }

        /// <summary>
        /// Finds the first job that requires a routing decision.
        /// </summary>
        /// <returns>The first <see cref="JobData"/> with state <see cref="JobState.NeedsRouting"/>, or <c>null</c>.</returns>
        public JobData GetNextNeedsRouting()
        {
            return allJobs.FirstOrDefault(j => j.State == JobState.NeedsRouting);
        }

        /// <summary>
        /// Finds every job that currently requires a routing decision, in list (~arrival) order.
        /// </summary>
        /// <remarks>
        /// Used by DecisionCoordinator so job-priority rules (SPT/LPT/SRT/LRT/FIFO) can select
        /// among ALL simultaneously-ready jobs, rather than always taking whichever one happens
        /// to be first in list order (see GetNextNeedsRouting) — measured at ~47-52% of routing
        /// decisions having 2+ jobs simultaneously ready (results/0902e decision_log.csv analysis).
        /// </remarks>
        public List<int> GetAllNeedingRouting()
        {
            return allJobs.Where(j => j.State == JobState.NeedsRouting).Select(j => j.JobId).ToList();
        }

        /// <summary>
        /// Finds the first job waiting for pickup that has not yet been assigned an AGV.
        /// </summary>
        /// <returns>The first unassigned <see cref="JobData"/> with state <see cref="JobState.WaitingForPickup"/>, or <c>null</c>.</returns>
        public JobData GetNextUnassignedPickup()
        {
            return allJobs.FirstOrDefault(j => j.State == JobState.WaitingForPickup && j.AssignedAgvId == -1);
        }

        /// <summary>
        /// Returns a list of job IDs currently queued and ready for processing at a specific machine.
        /// </summary>
        /// <param name="machineId">The ID of the machine to check.</param>
        /// <returns>A list of job IDs with state <see cref="JobState.Queued"/> at the specified machine.</returns>
        public List<int> GetDispatchableJobs(int machineId)
        {
            return allJobs
                .Where(j => j.State == JobState.Queued && j.LocationMachineId == machineId)
                .Select(j => j.JobId)
                .ToList();
        }

        /// <summary>
        /// Checks if any jobs are currently queued at the specified machine.
        /// </summary>
        /// <param name="machineId">The ID of the machine to check.</param>
        /// <returns><c>true</c> if any jobs have state <see cref="JobState.Queued"/> at the specified machine.</returns>
        public bool HasDispatchableJob(int machineId)
        {
            return allJobs.Any(j => j.State == JobState.Queued && j.LocationMachineId == machineId);
        }

        /// <summary>
        /// Determines if all jobs in the simulation have reached the <c>Exited</c> state.
        /// </summary>
        /// <returns><c>true</c> if all jobs have state <see cref="JobState.Exited"/>; otherwise, <c>false</c>.</returns>
        public bool AreAllExited()
        {
            if (allJobs.Count == 0) return false;
            return allJobs.All(j => j.State == JobState.Exited);
        }

        /// <summary>
        /// Returns the total count of jobs currently in the specified <see cref="JobState"/>.
        /// </summary>
        /// <param name="state">The job state to count.</param>
        /// <returns>The number of jobs in the specified state.</returns>
        public int CountInState(JobState state)
        {
            return allJobs.Count(j => j.State == state);
        }

        /// <summary>
        /// Calculates the total remaining processing load for a specific machine.
        /// </summary>
        /// <param name="machineId">The ID of the machine to evaluate.</param>
        /// <returns>
        /// The sum of processing times for all jobs currently queued at the machine,
        /// plus all jobs committed to the machine but not yet arrived.
        /// </returns>
        public float GetMachineLoad(int machineId)
        {
            float load = 0f;
            foreach (var job in allJobs)
            {
                // Count jobs physically queued here
                if (job.State == JobState.Queued && job.LocationMachineId == machineId)
                    load += job.GetProcessingTime(machineId);

                // Also count jobs committed to this machine but not yet arrived
                else if (job.TargetMachineId == machineId &&
                         (job.State == JobState.WaitingForPickup || job.State == JobState.InTransit))
                    load += job.GetProcessingTime(machineId);
            }
            return load;
        }

        /// <summary>
        /// Retrieves the estimated processing time for a specific job on a specific machine.
        /// </summary>
        /// <param name="jobId">The ID of the job.</param>
        /// <param name="machineId">The ID of the machine.</param>
        /// <returns>The processing time in simulation seconds; returns 0 if the job or machine is not found.</returns>
        public float GetProcessingTime(int jobId, int machineId)
        {
            var job = Get(jobId);
            return job?.GetProcessingTime(machineId) ?? 0f;
        }
    }
}