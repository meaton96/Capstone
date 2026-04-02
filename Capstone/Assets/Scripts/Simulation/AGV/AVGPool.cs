using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation.AGV
{
    public class AGVPool : MonoBehaviour
    {
        [Header("Fleet Settings")]
        [SerializeField] private AGVController agvPrefab;
        [SerializeField] private int fleetSize = 5;
        [SerializeField] private Transform parkingArea;

        private List<AGVController> fleet = new List<AGVController>();

        private struct DispatchRequest
        {
            public int JobId;
            public Vector3 PickupPosition;
            public Vector3 DropoffSlotPosition;
            public PhysicalMachine Source;
            public PhysicalMachine Dropoff;
        }
        private Queue<DispatchRequest> pendingRequests = new Queue<DispatchRequest>();

        // ─────────────────────────────────────────────────────────
        //  Fleet Lifecycle
        // ─────────────────────────────────────────────────────────

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

        // ─────────────────────────────────────────────────────────
        //  Dispatch API
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Dispatches with a per-index stagger delay so AGVs don't all pathfind
        /// simultaneously at episode start. dropoffSlotPos must be pre-reserved.
        /// </summary>
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
        /// Dispatches an available AGV immediately, or queues the request if
        /// the whole fleet is busy. Drained automatically as AGVs become idle.
        /// </summary>
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

        // ─────────────────────────────────────────────────────────
        //  Idle Callback
        // ─────────────────────────────────────────────────────────

        private void OnAnyAGVBecameIdle()
        {
            if (pendingRequests.Count == 0) return;
            AGVController agv = GetAvailableAGV();
            if (agv == null) return;
            DispatchRequest req = pendingRequests.Dequeue();
            SimLogger.Low($"[AGVPool] Draining queue — assigning Job {req.JobId} to AGV {agv.AgvId}.");
            agv.Dispatch(req.JobId, req.PickupPosition, req.DropoffSlotPosition, req.Source, req.Dropoff);
        }

        // ─────────────────────────────────────────────────────────
        //  Query
        // ─────────────────────────────────────────────────────────

        public AGVController GetAvailableAGV()
        {
            foreach (var agv in fleet)
                if (agv.State == AGVState.Idle) return agv;
            return null;
        }
    }
}