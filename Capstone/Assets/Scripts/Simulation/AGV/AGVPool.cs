using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Logging;
using Assets.Scripts.Simulation.FactoryLayout;

namespace Assets.Scripts.Simulation.AGV
{
    /// @brief Manages a fleet of AGVs, handling initialization, job dispatching, and request queuing.
    /// @details Dispatch requests store ONLY the job ID. Pickup/dropoff positions are resolved
    ///          at execution time from JobManager's authoritative state — never captured and cached.
    public class AGVPool : MonoBehaviour
    {
        public static AGVPool Instance;
        private Vector3[] parkingPositions;
        [SerializeField] private AGVController agvPrefab;
        [SerializeField] private int fleetSize = 5;
        [SerializeField] private FactoryLayoutManager layoutManager;

        private List<AGVController> fleet = new List<AGVController>();
        public List<AGVController> Fleet => fleet;

        /// @brief A pending transport request. Only stores the job ID — positions
        ///        are resolved fresh when the AGV actually starts the task.
        private struct DispatchRequest
        {
            public int JobId;
        }

        private Queue<DispatchRequest> pendingRequests = new Queue<DispatchRequest>();

        void Awake()
        {
            Instance = this;
        }

        public void InitializeFleet()
        {
            foreach (var agv in fleet) Destroy(agv.gameObject);
            fleet.Clear();
            pendingRequests.Clear();

            Vector3 baseParkingPos = layoutManager != null ? layoutManager.AGVParkingPosition : Vector3.zero;
            parkingPositions = new Vector3[fleetSize];
            for (int i = 0; i < fleetSize; i++)
            {
                Vector3 spawnPos = baseParkingPos + new Vector3(i * 2f, 0, 0);
                parkingPositions[i] = spawnPos;
                AGVController newAgv = Instantiate(agvPrefab, spawnPos, Quaternion.identity, this.transform);
                newAgv.gameObject.name = $"AGV_{i}";
                newAgv.Initialize(i);
                newAgv.SetIdleCallback(OnAnyAGVBecameIdle);
                fleet.Add(newAgv);
            }

            SimLogger.Medium($"[AGVPool] Spawned fleet of {fleetSize} AGVs.");
        }

        public Vector3 GetParkingPosition(int agvId)
        {
            if (parkingPositions != null && agvId < parkingPositions.Length)
                return parkingPositions[agvId];
            return Vector3.zero;
        }

        /// @brief Dispatches a job to an AGV or queues it. Only the job ID is stored.
        public void TryDispatch(int jobId)
        {
            AGVController agv = GetAvailableAGV();
            SimLogger.High($"[AGVPool] TryDispatch job={jobId} - " +
               (agv != null ? $"assigned to AGV {agv.AgvId}" : "no AGV free, queuing"));

            if (agv != null)
            {
                agv.Dispatch(jobId);
            }
            else
            {
                SimLogger.High($"[AGVPool] No AGV free for Job {jobId} - queuing request.");
                pendingRequests.Enqueue(new DispatchRequest { JobId = jobId });
            }
        }

        /// @brief Called when any AGV becomes idle. Drains the next queued request.
        private void OnAnyAGVBecameIdle()
        {
            if (pendingRequests.Count == 0) return;
            AGVController agv = GetAvailableAGV();
            if (agv == null) return;

            DispatchRequest req = pendingRequests.Dequeue();
            SimLogger.High($"[AGVPool] Draining queue - assigning Job {req.JobId} to AGV {agv.AgvId}.");
            agv.Dispatch(req.JobId);
        }

        public AGVController GetAvailableAGV()
        {
            foreach (var agv in fleet)
                if (agv.State == AGVState.Idle) return agv;
            return null;
        }
    }
}