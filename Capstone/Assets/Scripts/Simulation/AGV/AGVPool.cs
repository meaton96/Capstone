using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation.AGV
{
    /// <summary>
    /// Manages a fleet of AGVs.
    /// In the orchestrator architecture, this is purely a container and factory.
    /// The SimulationBridge (Orchestrator) pulls idle AGVs from here and dispatches them.
    /// </summary>
    public class AGVPool : MonoBehaviour
    {
        public static AGVPool Instance;
        private Vector3[] parkingPositions;
        [SerializeField] private AGVController agvPrefab;
        [SerializeField] private int fleetSize = 5;
        [SerializeField] private FactoryLayoutManager layoutManager;

        private List<AGVController> fleet = new List<AGVController>();

        // Expose fleet for the orchestrator to harvest flags
        public IReadOnlyList<AGVController> AllAGVs => fleet;

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

                // Note: Idle callbacks are no longer assigned here. 
                // The orchestrator actively polls GetIdleAGV() during Phase 3.
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

        /// <summary>
        /// Queried by the Orchestrator to find an available AGV for dispatch.
        /// </summary>
        public AGVController GetIdleAGV()
        {
            foreach (var agv in fleet)
            {
                if (agv.IsIdle) return agv;
            }
            return null;
        }

        /// <summary>
        /// Returns the best available AGV for a new dispatch.
        /// Prefers truly idle AGVs (already parked, no travel cost).
        /// Falls back to AGVs currently returning to parking — they can be
        /// intercepted mid-route and redirected, saving the full home trip.
        /// </summary>
        public AGVController GetAvailableAGV()
        {
            // Pass 1: truly idle (at or near parking)
            foreach (var agv in fleet)
                if (agv.IsIdle) return agv;

            // Pass 2: returning home — preemptable
            foreach (var agv in fleet)
                if (agv.State == AGVState.ReturningToParking) return agv;

            return null;
        }
    }
}