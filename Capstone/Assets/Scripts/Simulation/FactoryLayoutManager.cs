using System;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;

namespace Assets.Scripts.Simulation
{
    public class FactoryLayoutManager : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Inspector
        // ─────────────────────────────────────────────────────────

        [Header("Prefabs")]
        [Tooltip("Machine prefab with a PhysicalMachine component attached.")]
        [SerializeField] private PhysicalMachine machinePrefab; // <-- CHANGED

        [Header("Floor")]
        [Tooltip("Reference to the ground plane Transform so we can centre the layout.")]
        [SerializeField] private Transform floorTransform;

        [Tooltip("World-space size of the factory floor (units along X and Z).")]
        [SerializeField] private Vector2 floorSize = new Vector2(20f, 20f);

        [Header("Layout")]
        [Tooltip("Vertical offset so machine bases sit on top of the floor plane.")]
        [SerializeField] private float machineYOffset = 0.5f;

        [Tooltip("If true, log the distance matrix to the console on build.")]
        [SerializeField] private bool logDistanceMatrix = true;

        // ─────────────────────────────────────────────────────────
        //  Runtime state
        // ─────────────────────────────────────────────────────────

        /// @brief All instantiated physical machines, indexed by machine ID.
        private PhysicalMachine[] machines; // <-- CHANGED

        private float[,] distanceMatrix;
        private float[] distanceMatrixFlat;
        private Vector3[] customPositions;

        // ─────────────────────────────────────────────────────────
        //  Public accessors
        // ─────────────────────────────────────────────────────────

        public IReadOnlyList<PhysicalMachine> Machines => machines; // <-- CHANGED
        public int MachineCount => machines?.Length ?? 0;
        public float[,] DistanceMatrix => distanceMatrix;
        public float[] DistanceMatrixFlat => distanceMatrixFlat;

        // ─────────────────────────────────────────────────────────
        //  Public API
        // ─────────────────────────────────────────────────────────

        public void BuildFloor(DESSimulator simulator)
        {
            if (simulator == null)
                throw new ArgumentNullException(nameof(simulator));

            ClearFloor();

            int count = simulator.Machines.Length;
            Vector3[] positions = ResolvePositions(count);
            machines = new PhysicalMachine[count];

            Vector3 floorCentre = floorTransform != null
                ? floorTransform.position
                : Vector3.zero;

            for (int i = 0; i < count; i++)
            {
                Vector3 worldPos = floorCentre + positions[i];
                worldPos.y = machineYOffset;

                // <-- CHANGED to spawn PhysicalMachine
                PhysicalMachine physicalMachine = Instantiate(machinePrefab, worldPos, Quaternion.identity, transform);
                physicalMachine.gameObject.name = $"Machine_{i}";

                // Initialize the physical logic and the visual component
                physicalMachine.Initialize(i, simulator.Machines[i]);

                machines[i] = physicalMachine;
            }

            ComputeDistanceMatrix();

            if (logDistanceMatrix)
                LogDistanceMatrix();

            Debug.Log($"[FactoryLayout] Built floor: {count} machines, " +
                      $"floor {floorSize.x}×{floorSize.y}");
        }

        public void ClearFloor()
        {
            if (machines == null) return;

            foreach (PhysicalMachine pm in machines)
            {
                if (pm != null)
                    Destroy(pm.gameObject);
            }

            machines = null;
            distanceMatrix = null;
            distanceMatrixFlat = null;
        }

        public void SetCustomLayout(Vector3[] positions)
        {
            customPositions = positions;
        }

        // <-- CHANGED NAME AND RETURN TYPE
        public PhysicalMachine GetMachine(int machineId)
        {
            if (machines == null || machineId < 0 || machineId >= machines.Length)
                return null;
            return machines[machineId];
        }

        // ─────────────────────────────────────────────────────────
        //  Position resolution & Layout Tables (Unchanged)
        // ─────────────────────────────────────────────────────────

        private Vector3[] ResolvePositions(int count)
        {
            if (customPositions != null)
            {
                if (customPositions.Length != count)
                    throw new InvalidOperationException($"Custom layout has {customPositions.Length} positions but simulator has {count} machines.");
                return customPositions;
            }

            return count switch
            {
                8 => GetLayout8(),
                15 => GetLayout15(),
                20 => GetLayout20(),
                _ => GenerateGridPositions(count)
            };
        }

        private Vector3[] GenerateGridPositions(int count)
        {
            int cols = Mathf.CeilToInt(Mathf.Sqrt(count));
            int rows = Mathf.CeilToInt((float)count / cols);

            float margin = 2f;
            float spacingX = (floorSize.x - margin * 2) / Mathf.Max(cols - 1, 1);
            float spacingZ = (floorSize.y - margin * 2) / Mathf.Max(rows - 1, 1);

            float originX = -(floorSize.x - margin * 2) / 2f;
            float originZ = -(floorSize.y - margin * 2) / 2f;

            Vector3[] positions = new Vector3[count];
            for (int i = 0; i < count; i++)
            {
                int col = i % cols;
                int row = i / cols;
                positions[i] = new Vector3(originX + col * spacingX, 0f, originZ + row * spacingZ);
            }
            return positions;
        }

        private static Vector3[] GetLayout8()
        {
            return new[] {
                new Vector3(-6f, 0f, -3f), new Vector3(-2f, 0f, -3f), new Vector3( 2f, 0f, -3f), new Vector3( 6f, 0f, -3f),
                new Vector3(-6f, 0f,  3f), new Vector3(-2f, 0f,  3f), new Vector3( 2f, 0f,  3f), new Vector3( 6f, 0f,  3f),
            };
        }

        private static Vector3[] GetLayout15()
        {
            Vector3[] positions = new Vector3[15];
            float[] xSlots = { -7f, -3.5f, 0f, 3.5f, 7f };
            float[] zRows = { -4f, 0f, 4f };
            int index = 0;
            for (int row = 0; row < 3; row++)
                for (int col = 0; col < 5; col++)
                    positions[index++] = new Vector3(xSlots[col], 0f, zRows[row]);
            return positions;
        }

        private static Vector3[] GetLayout20()
        {
            Vector3[] positions = new Vector3[20];
            float[] xSlots = { -7f, -3.5f, 0f, 3.5f, 7f };
            float[] zRows = { -4.5f, -1.5f, 1.5f, 4.5f };
            int index = 0;
            for (int row = 0; row < 4; row++)
                for (int col = 0; col < 5; col++)
                    positions[index++] = new Vector3(xSlots[col], 0f, zRows[row]);
            return positions;
        }

        // ─────────────────────────────────────────────────────────
        //  Distance matrix (Updated to use machines array)
        // ─────────────────────────────────────────────────────────

        private void ComputeDistanceMatrix()
        {
            int n = machines.Length;
            distanceMatrix = new float[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    Vector3 a = machines[i].transform.position;
                    Vector3 b = machines[j].transform.position;
                    float dx = a.x - b.x;
                    float dz = a.z - b.z;
                    float dist = Mathf.Sqrt(dx * dx + dz * dz);

                    distanceMatrix[i, j] = dist;
                    distanceMatrix[j, i] = dist;
                }
            }

            const int obsSize = 8;
            distanceMatrixFlat = new float[obsSize * obsSize];
            int limit = Mathf.Min(n, obsSize);
            for (int i = 0; i < limit; i++)
                for (int j = 0; j < limit; j++)
                    distanceMatrixFlat[i * obsSize + j] = distanceMatrix[i, j];

            float maxDist = 0f;
            for (int k = 0; k < distanceMatrixFlat.Length; k++)
                if (distanceMatrixFlat[k] > maxDist)
                    maxDist = distanceMatrixFlat[k];

            if (maxDist > 0f)
                for (int k = 0; k < distanceMatrixFlat.Length; k++)
                    distanceMatrixFlat[k] /= maxDist;
        }

        private void LogDistanceMatrix()
        {
            if (distanceMatrix == null) return;
            int n = machines.Length;
            string header = "     ";
            for (int j = 0; j < n; j++) header += $"  M{j,-4}";
            Debug.Log($"[FactoryLayout] Distance matrix ({n}×{n}) — world units:\n{header}");

            for (int i = 0; i < n; i++)
            {
                string row = $"M{i,-3} ";
                for (int j = 0; j < n; j++) row += $"{distanceMatrix[i, j],6:F1} ";
                Debug.Log(row);
            }
            Debug.Log($"[FactoryLayout] Flat 64-D (normalised): [{string.Join(", ", FormatFlat())}]");
        }

        private string[] FormatFlat()
        {
            string[] result = new string[distanceMatrixFlat.Length];
            for (int i = 0; i < distanceMatrixFlat.Length; i++)
                result[i] = distanceMatrixFlat[i].ToString("F2");
            return result;
        }

        private void OnDrawGizmosSelected()
        {
            if (floorTransform == null) return;
            Vector3 centre = floorTransform.position;
            Gizmos.color = new Color(1f, 1f, 1f, 0.15f);
            Gizmos.DrawWireCube(centre + Vector3.up * 0.01f, new Vector3(floorSize.x, 0f, floorSize.y));

            Gizmos.color = Color.cyan;
            Vector3[] preview = GetLayout8();
            foreach (Vector3 pos in preview)
            {
                Vector3 world = centre + pos + Vector3.up * machineYOffset;
                Gizmos.DrawWireCube(world, Vector3.one);
            }
        }
    }
}