using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Assets.Scripts.Simulation.Jobs
{
    /// @brief Central repository for all active JobData instances in the simulation.
    ///
    /// @details Acts as a passive data store that translates static job definitions into 
    /// live runtime trackers. The orchestrator queries and modifies these instances 
    /// directly to manage the factory lifecycle.
    public class JobStore : MonoBehaviour
    {
        [Header("Visuals")]
        [SerializeField] private JobVisual jobVisualPrefab;
        [SerializeField] private Transform jobVisualContainer;

        private readonly List<JobData> allJobs = new List<JobData>();

        public IReadOnlyList<JobData> AllJobs => allJobs;
        public int JobCount => allJobs.Count;

        public bool IsInitialized = false;

        /// @brief Converts static job definitions into runtime state trackers.
        ///
        /// @param definitions An enumerable of @c FJSSPJobDefinition blueprint data.
        /// @param spawnVisuals Whether to instantiate 3D visuals for each job.
        ///
        /// @details Clears existing data and populates the store with fresh @c JobData. 
        /// If @c spawnVisuals is true, it links each data entry to a newly instantiated 
        /// @c JobVisual within the designated container.
        public void Initialize(IEnumerable<FJSSPJobDefinition> definitions, bool spawnVisuals)
        {
            allJobs.Clear();

            foreach (var def in definitions)
            {
                var jobData = new JobData
                {
                    JobId = def.JobId,
                    ArrivalTime = def.ArrivalTime,
                    OperationTypes = def.OperationSequence,
                    EligibleMachinesPerOp = def.EligibleMachinesPerOp,
                    TotalOperations = def.OperationSequence.Length,

                    State = JobState.NeedsRouting,
                    LocationMachineId = -1,
                    TargetMachineId = -1,
                    AssignedAgvId = -1,
                    CurrentOpIndex = 0,
                    CompletedOps = 0,
                    StateEntryTime = 0,
                    TotalWaitTime = 0,
                    TotalTransitTime = 0
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

        /// @brief Destroys all associated @c JobVisual objects and clears the internal list.
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

        /// @brief Retrieves a specific @c JobData instance by its unique identifier.
        ///
        /// @param jobId The unique ID of the job to retrieve.
        /// @return The @c JobData instance if found; otherwise, @c null.
        public JobData Get(int jobId)
        {
            return allJobs.FirstOrDefault(j => j.JobId == jobId);
        }

        /// @brief Finds the first job that requires a routing decision.
        public JobData GetNextNeedsRouting()
        {
            return allJobs.FirstOrDefault(j => j.State == JobState.NeedsRouting);
        }

        /// @brief Finds the first job waiting for pickup that has not yet been assigned an AGV.
        public JobData GetNextUnassignedPickup()
        {
            return allJobs.FirstOrDefault(j => j.State == JobState.WaitingForPickup && j.AssignedAgvId == -1);
        }

        /// @brief Returns a list of job IDs currently queued and ready for processing at a specific machine.
        ///
        /// @param machineId The ID of the machine to check.
        public List<int> GetDispatchableJobs(int machineId)
        {
            return allJobs
                .Where(j => j.State == JobState.Queued && j.LocationMachineId == machineId)
                .Select(j => j.JobId)
                .ToList();
        }

        /// @brief Checks if any jobs are currently queued at the specified machine.
        public bool HasDispatchableJob(int machineId)
        {
            return allJobs.Any(j => j.State == JobState.Queued && j.LocationMachineId == machineId);
        }

        /// @brief Determines if all jobs in the simulation have reached the @c Exited state.
        public bool AreAllExited()
        {
            if (allJobs.Count == 0) return false;
            return allJobs.All(j => j.State == JobState.Exited);
        }

        /// @brief Returns the total count of jobs currently in the specified @c JobState.
        public int CountInState(JobState state)
        {
            return allJobs.Count(j => j.State == state);
        }

        /// @brief Calculates the total remaining processing load for a specific machine.
        ///
        /// @param machineId The ID of the machine to evaluate.
        /// @return The sum of processing times for all jobs currently queued at that machine.
        public float GetMachineLoad(int machineId)
        {
            float load = 0f;
            foreach (var job in allJobs)
            {
                if (job.State == JobState.Queued && job.LocationMachineId == machineId)
                {
                    load += job.GetProcessingTime(machineId);
                }
            }
            return load;
        }

        /// @brief Retrieves the estimated processing time for a specific job on a specific machine.
        public float GetProcessingTime(int jobId, int machineId)
        {
            var job = Get(jobId);
            return job?.GetProcessingTime(machineId) ?? 0f;
        }
    }
}