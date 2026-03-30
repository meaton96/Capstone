using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;

namespace Assets.Scripts.Simulation
{
    /// @brief Acts as the physical anchor for a machine in the scene.
    /// Manages the actual passage of time and listens for Unity Physics collisions.
    [RequireComponent(typeof(Collider))]
    [RequireComponent(typeof(Rigidbody))]
    [RequireComponent(typeof(MachineVisual))]
    public class PhysicalMachine : MonoBehaviour
    {
        public int MachineId { get; private set; }
        public bool IsIdle { get; private set; } = true;

        /// @brief The list of job IDs currently sitting in the physical trigger zone.
        public List<int> PhysicalQueue = new List<int>();

        private MachineVisual visualLayer;

        /// @brief Wires up the physical machine to the logical instance and visual layer.
        public void Initialize(int id, Machine coreMachineData)
        {
            MachineId = id;
            IsIdle = true;
            PhysicalQueue.Clear();

            // Link up the visual layer so we can still flash colors and update progress bars
            visualLayer = GetComponent<MachineVisual>();
            if (visualLayer != null)
            {
                visualLayer.Initialise(id, coreMachineData);
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Physics Event: A Job Arrives
        // ─────────────────────────────────────────────────────────

        /// @brief Fired when a Job's collider enters the Machine's trigger zone.
        private void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Job"))
            {
                JobVisual job = other.GetComponent<JobVisual>();
                if (job != null)
                {
                    JobTracker tracker = SimulationBridge.Instance.JobManager.GetJobTracker(job.JobId);
                    if (tracker != null && tracker.NextMachineId != MachineId)
                    {
                        return; // It's just flying past us. Ignore it!
                    }

                    if (!PhysicalQueue.Contains(job.JobId))
                    {
                        PhysicalQueue.Add(job.JobId);

                        if (visualLayer != null) visualLayer.UpdateQueueLabel(PhysicalQueue.Count);

                        Debug.Log($"[PhysicalMachine {MachineId}] Job {job.JobId} physically arrived in queue.");

                        if (SimulationBridge.Instance != null)
                        {
                            SimulationBridge.Instance.OnJobArrivedInQueue(MachineId, job.JobId);
                        }
                    }
                }
            }
        }

        private void OnTriggerExit(Collider other)
        {
            if (other.CompareTag("Job"))
            {
                JobVisual job = other.GetComponent<JobVisual>();
                if (job != null)
                {
                    // Ensure we only remove it if it was actually in our queue
                    if (PhysicalQueue.Contains(job.JobId))
                    {
                        PhysicalQueue.Remove(job.JobId);

                        if (visualLayer != null) visualLayer.UpdateQueueLabel(PhysicalQueue.Count);
                    }
                }
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Time Management: Processing a Job
        // ─────────────────────────────────────────────────────────

        /// @brief Starts the actual Unity Coroutine timer for processing.
        public void StartProcessing(int jobId, float realTimeDuration)
        {
            IsIdle = false;
            PhysicalQueue.Remove(jobId);

            if (visualLayer != null)
            {
                visualLayer.BeginOperation(jobId, Time.time, realTimeDuration);
                visualLayer.UpdateQueueLabel(PhysicalQueue.Count);
            }

            StartCoroutine(ProcessJobRoutine(jobId, realTimeDuration));
        }

        private IEnumerator ProcessJobRoutine(int jobId, float duration)
        {
            float elapsed = 0f;

            // Loop yielding per frame allows us to update the visual progress bar smoothly
            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;

                if (visualLayer != null)
                {
                    visualLayer.UpdateProgress(elapsed / duration); // Assuming UpdateProgress takes a 0-1 float now
                }

                yield return null;
            }

            IsIdle = true;

            if (visualLayer != null)
            {
                visualLayer.CompleteOperation(jobId);
            }

            Debug.Log($"[PhysicalMachine {MachineId}] Finished processing Job {jobId}.");

            // Tell the bridge we are done so it can dispatch an AGV and schedule the next job
            if (SimulationBridge.Instance != null)
            {
                SimulationBridge.Instance.OnMachineFinished(MachineId, jobId);
            }
        }
    }
}