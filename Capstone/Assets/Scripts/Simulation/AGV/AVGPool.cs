using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Logging;
using Assets.Scripts.Simulation.FactoryLayout;

namespace Assets.Scripts.Simulation.AGV
{
    /// @brief Manages a fleet of AGVs, handling initialization, job dispatching, and request queuing.
    public class AGVPool : MonoBehaviour
    {
        [SerializeField] private AGVController agvPrefab;
        [SerializeField] private int fleetSize = 5;
        [SerializeField] private FactoryLayoutManager layoutManager;

        private List<AGVController> fleet = new List<AGVController>();
        public List<AGVController> Fleet => fleet;

        private struct DispatchRequest
        {
            public int JobId;
            public Vector3 PickupPosition;
            public Vector3 DropoffSlotPosition;
            public PhysicalMachine Source;
            public PhysicalMachine Dropoff;
        }

        private Queue<DispatchRequest> pendingRequests = new Queue<DispatchRequest>();

        /// @brief Spawns and prepares the AGV fleet at the designated parking area.
        ///
        /// @details Clears any existing fleet instances and the pending request queue. 
        /// Instances are staggered along the X-axis starting from the layout manager's 
        /// parking position. Each AGV is initialized with a unique ID and a callback 
        /// for idle state transitions.
        ///
        /// @post The fleet list is populated with initialized AGVControllers and 
        /// pendingRequests is empty.
        public void InitializeFleet()
        {
            foreach (var agv in fleet) Destroy(agv.gameObject);
            fleet.Clear();
            pendingRequests.Clear();

            Vector3 baseParkingPos = layoutManager != null ? layoutManager.AGVParkingPosition : Vector3.zero;

            for (int i = 0; i < fleetSize; i++)
            {
                Vector3 spawnPos = baseParkingPos + new Vector3(i * 2f, 0, 0);

                AGVController newAgv = Instantiate(agvPrefab, spawnPos, Quaternion.identity, this.transform);
                newAgv.gameObject.name = $"AGV_{i}";
                newAgv.Initialize(i);
                newAgv.SetIdleCallback(OnAnyAGVBecameIdle);
                fleet.Add(newAgv);
            }

            Debug.Log($"[AGVPool] Spawned fleet of {fleetSize} AGVs.");
        }

        /// @brief Initiates a job dispatch with a calculated delay.
        ///
        /// @details Offloads the dispatch to a coroutine that waits for a duration based 
        /// on the @p dispatchIndex. This prevents CPU spikes caused by multiple AGVs 
        /// calculating complex BFS paths on the same frame during simulation startup.
        ///
        /// @param jobId The unique ID of the job.
        /// @param pickupPos The world position for pickup.
        /// @param dropoffSlotPos The pre-reserved world position for dropoff.
        /// @param source The source machine.
        /// @param dropoff The destination machine.
        /// @param dispatchIndex The index used to calculate the stagger delay (index * 0.3s).
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

        /// @brief Attempts to assign a transport task to an available AGV or queues it if the fleet is full.
        ///
        /// @details Queries the fleet for an idle AGV. If one is found, the job is dispatched 
        /// immediately. If all AGVs are busy, the task is stored in @ref pendingRequests 
        /// to be processed once an AGV becomes idle.
        ///
        /// @param jobId The unique ID of the job.
        /// @param pickupPos The world position for pickup.
        /// @param dropoffSlotPos The pre-reserved world position for dropoff.
        /// @param source The source machine.
        /// @param dropoff The destination machine.
        ///
        /// @post Job is either assigned to an AGV or added to the tail of the queue.
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

        /// @brief Processes the next pending request when an AGV signals it has returned to idle.
        ///
        /// @details Triggered via callback from an AGVController. If requests are queued, 
        /// it retrieves the first available AGV and dequeues the next @ref DispatchRequest.
        ///
        /// @pre pendingRequests.Count must be greater than 0.
        /// @post The next job in the queue is assigned to an AGV and dequeued.
        private void OnAnyAGVBecameIdle()
        {
            if (pendingRequests.Count == 0) return;
            AGVController agv = GetAvailableAGV();
            if (agv == null) return;

            DispatchRequest req = pendingRequests.Dequeue();
            SimLogger.Low($"[AGVPool] Draining queue — assigning Job {req.JobId} to AGV {agv.AgvId}.");
            agv.Dispatch(req.JobId, req.PickupPosition, req.DropoffSlotPosition, req.Source, req.Dropoff);
        }

        /// @brief Scans the fleet for an AGV in the Idle state.
        ///
        /// @return The first available @ref AGVController found, or null if all are busy.
        public AGVController GetAvailableAGV()
        {
            foreach (var agv in fleet)
                if (agv.State == AGVState.Idle) return agv;
            return null;
        }
    }
}