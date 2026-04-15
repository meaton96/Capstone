using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.FactoryLayout;
using Assets.Scripts.Logging;

namespace Assets.Scripts.Simulation.AGV
{
    /// @brief Manages the lifecycle and retrieval of the AGV fleet.
    ///
    /// @details Serves as a centralized container and factory. The @c SimulationBridge
    /// orchestrator queries this pool to identify and dispatch available units based
    /// on their current operational state.
    public class AGVPool : MonoBehaviour
    {
        public static AGVPool Instance;
        private Vector3[] parkingPositions;
        [SerializeField] private AGVController agvPrefab;
        [SerializeField] private FactoryLayoutManager layoutManager;

        private List<AGVController> fleet = new List<AGVController>();

        public IReadOnlyList<AGVController> AllAGVs => fleet;

        private void Awake()
        {
            Instance = this;
        }

        /// @brief Destroys the existing fleet and instantiates new AGV units.
        ///
        /// @param fleetSize The number of AGV units to spawn, driven by @c FJSSPConfig.AGVCount.
        ///
        /// @details Calculates parking positions based on the @c layoutManager coordinates
        /// and spawns units in a linear arrangement. Each unit is initialized with
        /// a unique ID corresponding to its index in the fleet.
        public void InitializeFleet(int fleetSize)
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

                fleet.Add(newAgv);
            }

            SimLogger.Medium($"[AGVPool] Spawned fleet of {fleetSize} AGVs.");
        }

        /// @brief Destroys all AGVs in the fleet and clears the pool.
        public void ClearFleet()
        {
            foreach (var agv in fleet)
            {
                if (agv != null)
                    Destroy(agv.gameObject);
            }
            fleet.Clear();
        }

        /// @brief Retrieves the designated world-space parking coordinate for a specific AGV.
        ///
        /// @param agvId The unique identifier of the AGV.
        /// @return The @c Vector3 position for the unit's parking station, or @c Vector3.zero if invalid.
        public Vector3 GetParkingPosition(int agvId)
        {
            if (parkingPositions != null && agvId < parkingPositions.Length)
                return parkingPositions[agvId];
            return Vector3.zero;
        }

        /// @brief Locates the first unit that is currently in a strictly @c Idle state.
        ///
        /// @return An @c AGVController instance if an idle unit exists; otherwise, @c null.
        public AGVController GetIdleAGV()
        {
            foreach (var agv in fleet)
                if (agv.IsIdle) return agv;
            return null;
        }

        /// @brief Identifies the best candidate for a new task dispatch.
        ///
        /// @details Performs a two-pass search: first for units already at their
        /// parking stations (@c Idle), and second for units currently @c ReturningToParking
        /// that can be redirected mid-route to optimize travel time.
        public AGVController GetAvailableAGV()
        {
            foreach (var agv in fleet)
                if (agv.IsIdle) return agv;

            foreach (var agv in fleet)
                if (agv.State == AGVState.ReturningToParking) return agv;

            return null;
        }

        /// @brief Returns the AGV unit assigned to a specific job ID via pre-dispatch.
        ///
        /// @param jobId The identifier of the job to check.
        /// @return The assigned @c AGVController if found; otherwise, @c null.
        public AGVController GetPreDispatchedAGV(int jobId)
        {
            foreach (var agv in fleet)
                if (agv.IsPreDispatched && agv.PreDispatchedJobId == jobId)
                    return agv;
            return null;
        }
    }
}