using System.Collections.Generic;
using UnityEngine;

namespace Assets.Scripts.AGV
{
    public class AGVPool : MonoBehaviour
    {
        [Header("Fleet Settings")]
        [SerializeField] private AGVController agvPrefab;
        [SerializeField] private int fleetSize = 5;
        [SerializeField] private Transform parkingArea; // Where they spawn

        private List<AGVController> fleet = new List<AGVController>();

        public void InitializeFleet()
        {
            // Clear old fleet if resetting episode
            foreach (var agv in fleet) Destroy(agv.gameObject);
            fleet.Clear();

            // Spawn new robots
            for (int i = 0; i < fleetSize; i++)
            {
                Vector3 spawnPos = parkingArea != null ? parkingArea.position : Vector3.zero;
                // Add a small offset so they don't spawn exactly inside each other
                spawnPos += new Vector3(i * 2f, 0, 0);

                AGVController newAgv = Instantiate(agvPrefab, spawnPos, Quaternion.identity, this.transform);
                newAgv.gameObject.name = $"AGV_{i}";
                newAgv.Initialize(i);
                fleet.Add(newAgv);
            }
            Debug.Log($"[AGVPool] Spawned fleet of {fleetSize} AGVs.");
        }

        /// <summary>
        /// Finds the first Idle AGV in the fleet. Returns null if all are busy.
        /// </summary>
        public AGVController GetAvailableAGV()
        {
            foreach (var agv in fleet)
            {
                if (agv.State == AGVState.Idle)
                {
                    return agv;
                }
            }
            return null; // Uh oh, factory traffic jam!
        }
    }
}