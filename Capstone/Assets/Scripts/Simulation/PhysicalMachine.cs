using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation
{
    /// @brief Physical anchor for a machine in the Unity scene.
    ///
    /// @details Manages real-time coroutine processing and Unity Physics trigger
    /// events for job arrival and departure. Delegates visual updates to the
    /// attached @c MachineVisual component.
    [RequireComponent(typeof(Collider))]
    [RequireComponent(typeof(Rigidbody))]
    [RequireComponent(typeof(MachineVisual))]
    public class PhysicalMachine : MonoBehaviour
    {
        /// @brief Unique identifier matching the logical machine index.
        public int MachineId { get; private set; }

        /// @brief True when the machine is not currently processing a job.
        public bool IsIdle { get; private set; } = true;

        /// @brief Job IDs currently inside this machine's trigger zone.
        public List<int> PhysicalQueue = new List<int>();

        private MachineVisual visualLayer;

        // ─────────────────────────────────────────────────────────
        //  Initialization
        // ─────────────────────────────────────────────────────────

        /// @brief Wires this physical machine to its logical data and visual layer.
        /// @param id         The machine index used throughout the simulation.
        /// @param coreMachineData  The logical @c Machine object carrying scheduling metadata.
        public void Initialize(int id, Machine coreMachineData)
        {
            MachineId = id;
            IsIdle = true;
            PhysicalQueue.Clear();

            visualLayer = GetComponent<MachineVisual>();
            if (visualLayer != null)
            {
                visualLayer.Initialise(id, coreMachineData);
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Physics Events
        // ─────────────────────────────────────────────────────────

        /// @brief Called when a job's collider enters this machine's trigger zone.
        /// @details Validates that the arriving job is actually routed to this
        ///          machine before adding it to the physical queue and notifying
        ///          the @c SimulationBridge.
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
                        return;
                    }

                    if (!PhysicalQueue.Contains(job.JobId))
                    {
                        PhysicalQueue.Add(job.JobId);

                        if (visualLayer != null) visualLayer.UpdateQueueLabel(PhysicalQueue.Count);

                        SimLogger.High($"[PhysicalMachine {MachineId}] Job {job.JobId} physically arrived in queue.");

                        if (SimulationBridge.Instance != null)
                        {
                            SimulationBridge.Instance.OnJobArrivedInQueue(MachineId, job.JobId);
                        }
                    }
                }
            }
        }

        /// @brief Called when a job's collider exits this machine's trigger zone.
        /// @details Removes the job from the physical queue if it was registered.
        private void OnTriggerExit(Collider other)
        {
            if (other.CompareTag("Job"))
            {
                JobVisual job = other.GetComponent<JobVisual>();
                if (job != null)
                {
                    if (PhysicalQueue.Contains(job.JobId))
                    {
                        PhysicalQueue.Remove(job.JobId);

                        if (visualLayer != null) visualLayer.UpdateQueueLabel(PhysicalQueue.Count);
                    }
                }
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Processing
        // ─────────────────────────────────────────────────────────

        /// @brief Begins the coroutine timer that simulates job processing.
        /// @param jobId            The job to process.
        /// @param realTimeDuration Wall-clock seconds the operation should take.
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

        /// @brief Coroutine that advances a progress bar each frame and fires
        ///        @c SimulationBridge.OnMachineFinished when complete.
        /// @param jobId    The job being processed.
        /// @param duration Total wall-clock seconds for the operation.
        private IEnumerator ProcessJobRoutine(int jobId, float duration)
        {
            float elapsed = 0f;

            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;

                if (visualLayer != null)
                {
                    visualLayer.UpdateProgress(elapsed / duration);
                }

                yield return null;
            }

            IsIdle = true;

            if (visualLayer != null)
            {
                visualLayer.CompleteOperation(jobId);
            }

            SimLogger.High($"[PhysicalMachine {MachineId}] Finished processing Job {jobId}.");

            if (SimulationBridge.Instance != null)
            {
                SimulationBridge.Instance.OnMachineFinished(MachineId, jobId);
            }
        }
    }
}