using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Assets.Scripts.Simulation.Jobs
{
    /// <summary>
    /// A passive data store. The Orchestrator modifies these jobs directly.
    /// Maps static FJSSPJobDefinitions to runtime JobData instances.
    /// </summary>
    public class JobStore : MonoBehaviour
    {
        [Header("Visuals")]
        [SerializeField] private JobVisual jobVisualPrefab;
        [SerializeField] private Transform jobVisualContainer;

        private readonly List<JobData> allJobs = new List<JobData>();

        public IReadOnlyList<JobData> AllJobs => allJobs;
        public int JobCount => allJobs.Count;

        public bool IsInitialized = false;

        /// <summary>
        /// Converts the static job definitions into live runtime JobData state trackers.
        /// </summary>
        public void Initialize(IEnumerable<FJSSPJobDefinition> definitions, bool spawnVisuals)
        {
            allJobs.Clear();

            foreach (var def in definitions)
            {
                var jobData = new JobData
                {
                    // Blueprint data
                    JobId = def.JobId,
                    ArrivalTime = def.ArrivalTime,
                    OperationTypes = def.OperationSequence,
                    EligibleMachinesPerOp = def.EligibleMachinesPerOp,
                    TotalOperations = def.OperationSequence.Length,

                    // Runtime state initialization
                    State = JobState.NeedsRouting,
                    LocationMachineId = -1, // -1 means entry/spawn area
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
                    // Instantiate visual and link it to the data
                    JobVisual vis = Instantiate(jobVisualPrefab, Vector3.zero, Quaternion.identity, jobVisualContainer);
                    vis.gameObject.name = $"JobVisual_{def.JobId}";
                    vis.Initialize(def.JobId, def.OperationSequence.Length);
                    jobData.Visual = vis;
                }

                allJobs.Add(jobData);
                IsInitialized = true;
            }
        }

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
        }

        /// <summary>
        /// Returns the reference to the job. Modifying this instance 
        /// modifies the job in the store.
        /// </summary>
        public JobData Get(int jobId)
        {
            return allJobs.FirstOrDefault(j => j.JobId == jobId);
        }

        // ═══════════════════════════════════════════════════════════════
        //  ORCHESTRATOR QUERIES (Strictly based on the JobState enum)
        // ═══════════════════════════════════════════════════════════════

        public JobData GetNextNeedsRouting()
        {
            return allJobs.FirstOrDefault(j => j.State == JobState.NeedsRouting);
        }

        public JobData GetNextUnassignedPickup()
        {
            return allJobs.FirstOrDefault(j => j.State == JobState.WaitingForPickup && j.AssignedAgvId == -1);
        }

        public List<int> GetDispatchableJobs(int machineId)
        {
            return allJobs
                .Where(j => j.State == JobState.Queued && j.LocationMachineId == machineId)
                .Select(j => j.JobId)
                .ToList();
        }

        public bool HasDispatchableJob(int machineId)
        {
            return allJobs.Any(j => j.State == JobState.Queued && j.LocationMachineId == machineId);
        }

        public bool AreAllExited()
        {
            if (allJobs.Count == 0) return false;
            return allJobs.All(j => j.State == JobState.Exited);
        }

        public int CountInState(JobState state)
        {
            return allJobs.Count(j => j.State == state);
        }

        // ═══════════════════════════════════════════════════════════════
        //  METRIC HELPERS
        // ═══════════════════════════════════════════════════════════════

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

        public float GetProcessingTime(int jobId, int machineId)
        {
            var job = Get(jobId);
            return job?.GetProcessingTime(machineId) ?? 0f;
        }
    }
}