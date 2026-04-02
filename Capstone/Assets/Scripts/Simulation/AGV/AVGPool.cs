using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation.AGV
{
    /// <summary>
    /// Manages a pool of AGV controllers, handling their initialization, dispatching, and request queuing.
    /// </summary>
    public class AGVPool : MonoBehaviour
    {
        [Header("Fleet Settings")]
        [SerializeField] private AGVController agvPrefab;
        [SerializeField] private int fleetSize = 5;
        [SerializeField] private Transform parkingArea;

        private List<AGVController> fleet = new List<AGVController>();

        /// <summary>
        /// Represents a queued transport request when all AGVs are currently occupied.
        /// </summary>
        private struct DispatchRequest
        {
            public int JobId;
            public Vector3 PickupPosition;
            public Vector3 DropoffSlotPosition;
            public PhysicalMachine Source;
            public PhysicalMachine Dropoff;
        }

        private Queue<DispatchRequest> pendingRequests = new Queue<DispatchRequest>();

        /// <summary>
        /// Instantiates and initializes the fleet of AGVs based on the configured fleet size, clearing any existing state.
        /// </summary>
        public void InitializeFleet()
        {
            foreach (var agv in fleet) Destroy(agv.gameObject);
            fleet.Clear();
            pendingRequests.Clear();

            for (int i = 0; i < fleetSize; i++)
            {
                Vector3 spawnPos = parkingArea != null ? parkingArea.position : Vector3.zero;
                spawnPos += new Vector3(i * 2f, 0, 0);

                AGVController newAgv = Instantiate(agvPrefab, spawnPos, Quaternion.identity, this.transform);
                newAgv.gameObject.name = $"AGV_{i}";
                newAgv.Initialize(i);
                newAgv.SetIdleCallback(OnAnyAGVBecameIdle);
                fleet.Add(newAgv);
            }

            Debug.Log($"[AGVPool] Spawned fleet of {fleetSize} AGVs.");
        }

        /// <summary>
        /// Dispatches an AGV with a staggered delay to prevent simultaneous pathfinding computations at startup.
        /// </summary>
        /// <param name="jobId">The unique identifier of the job to dispatch.</param>
        /// <param name="pickupPos">The world position for pickup.</param>
        /// <param name="dropoffSlotPos">The pre-reserved world position for dropoff.</param>
        /// <param name="source">The source machine of the job.</param>
        /// <param name="dropoff">The destination machine receiving the job.</param>
        /// <param name="dispatchIndex">The index used to calculate the stagger delay multiplier.</param>
        public void TryDispatchStaggered(int jobId, Vector3 pickupPos, Vector3 dropoffSlotPos,
                                         PhysicalMachine source, PhysicalMachine dropoff, int dispatchIndex)
        {
            StartCoroutine(StaggeredDispatch(jobId, pickupPos, dropoffSlotPos, source, dropoff, dispatchIndex));
        }

        private IEnumerator StaggeredDispatch(int jobId, Vector3 pickupPos, Vector3 dropoffSlotPos,
                                              PhysicalMachine source, PhysicalMachine dropoff, int index)
        {
            yield return new WaitForSeconds(index * 0.3f);
            TryDispatch(jobId, pickupPos, dropoffSlotPos, source, dropoff);
        }

        /// <summary>
        /// Attempts to dispatch an available AGV immediately. Queues the request if the entire fleet is busy.
        /// </summary>
        /// <param name="jobId">The unique identifier of the job to dispatch.</param>
        /// <param name="pickupPos">The world position for pickup.</param>
        /// <param name="dropoffSlotPos">The pre-reserved world position for dropoff.</param>
        /// <param name="source">The source machine of the job.</param>
        /// <param name="dropoff">The destination machine receiving the job.</param>
        public void TryDispatch(int jobId, Vector3 pickupPos, Vector3 dropoffSlotPos,
                                PhysicalMachine source, PhysicalMachine dropoff)
        {
            AGVController agv = GetAvailableAGV();
            if (agv != null)
            {
                agv.Dispatch(jobId, pickupPos, dropoffSlotPos, source, dropoff);
            }
            else
            {
                SimLogger.Low($"[AGVPool] No AGV free for Job {jobId} — queuing request.");
                pendingRequests.Enqueue(new DispatchRequest
                {
                    JobId = jobId,
                    PickupPosition = pickupPos,
                    DropoffSlotPosition = dropoffSlotPos,
                    Source = source,
                    Dropoff = dropoff
                });
            }
        }

        /// <summary>
        /// Callback invoked when an AGV becomes idle, automatically assigning it the next pending request from the queue.
        /// </summary>
        private void OnAnyAGVBecameIdle()
        {
            if (pendingRequests.Count == 0) return;
            AGVController agv = GetAvailableAGV();
            if (agv == null) return;

            DispatchRequest req = pendingRequests.Dequeue();
            SimLogger.Low($"[AGVPool] Draining queue — assigning Job {req.JobId} to AGV {agv.AgvId}.");
            agv.Dispatch(req.JobId, req.PickupPosition, req.DropoffSlotPosition, req.Source, req.Dropoff);
        }

        /// <summary>
        /// Scans the fleet to find an AGV that is currently not assigned a task.
        /// </summary>
        /// <returns>An idle <see cref="AGVController"/> if one is available; otherwise, null.</returns>
        public AGVController GetAvailableAGV()
        {
            foreach (var agv in fleet)
                if (agv.State == AGVState.Idle) return agv;
            return null;
        }
    }
}