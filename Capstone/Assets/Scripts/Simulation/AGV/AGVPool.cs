using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Logging;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Simulation.Jobs;

namespace Assets.Scripts.Simulation.AGV
{
    /// @brief Manages a fleet of AGVs using a PULL model.
    /// @details There is NO dispatch queue. When an AGV becomes idle, it asks
    ///          JobManager for the next job that needs transport (Location == AwaitingPickup).
    ///          This eliminates stale-request bugs entirely — there's nothing to go stale.
    public class AGVPool : MonoBehaviour
    {
        public static AGVPool Instance;
        private Vector3[] parkingPositions;
        [SerializeField] private AGVController agvPrefab;
        [SerializeField] private int fleetSize = 5;
        [SerializeField] private FactoryLayoutManager layoutManager;

        private List<AGVController> fleet = new List<AGVController>();
        public List<AGVController> Fleet => fleet;

        void Awake()
        {
            Instance = this;
        }

        public void InitializeFleet()
        {
            foreach (var agv in fleet) Destroy(agv.gameObject);
            fleet.Clear();

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

        /// @brief Called when any AGV becomes idle. Pulls the next job from JobManager.
        private void OnAnyAGVBecameIdle()
        {
            TryAssignWork();
        }

        /// @brief Pairs an idle AGV with the next job needing transport.
        ///        Reads directly from JobManager — no queue, nothing to go stale.
        ///        Call this when a new job becomes AwaitingPickup, or when an AGV parks.
        public void TryAssignWork()
        {
            AGVController agv = GetAvailableAGV();
            if (agv == null) return;

            JobManager jm = SimulationBridge.Instance?.JobManager;
            if (jm == null) return;

            JobTracker job = jm.GetNextTransportJob();
            if (job == null) return;

            // Claim the job so no other AGV grabs it
            job.AssignedAGVId = agv.AgvId;

            SimLogger.High($"[AGVPool] Assigning Job {job.JobId} to AGV {agv.AgvId} (pull model).");
            agv.Dispatch(job.JobId);
        }

        public AGVController GetAvailableAGV()
        {
            foreach (var agv in fleet)
                if (agv.State == AGVState.Idle) return agv;
            return null;
        }
    }
}