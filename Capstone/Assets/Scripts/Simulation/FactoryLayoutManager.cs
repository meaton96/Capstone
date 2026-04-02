using System;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;
using Assets.Scripts.Logging;
using Unity.AI.Navigation;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation
{
    /// @brief Instantiates and positions @c PhysicalMachine prefabs on the factory floor.
    ///
    /// @details Selects a hard-coded layout for known machine counts (8, 15, 20) and
    /// falls back to a generated grid for all other counts. After building, it
    /// computes and exposes a normalised pairwise distance matrix for use as
    /// agent observations.
    public class FactoryLayoutManager : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Inspector
        // ─────────────────────────────────────────────────────────

        [SerializeField] private NavMeshSurface navMeshSurface;

        [Header("Prefabs")]
        [Tooltip("Machine prefab with a PhysicalMachine component attached.")]
        [SerializeField] private PhysicalMachine machinePrefab;

        [Header("Floor")]
        [Tooltip("Reference to the ground plane Transform so we can centre the layout.")]
        [SerializeField] private Transform floorTransform;

        [Tooltip("World-space size of the factory floor (units along X and Z).")]
        [SerializeField] private Vector2 floorSize = new Vector2(30f, 30f);

        [Header("Layout")]
        [Tooltip("Vertical offset so machine bases sit on top of the floor plane.")]
        [SerializeField] private float machineYOffset = 0.5f;

        [Tooltip("Minimum distance between machine centres in the auto-grid. " +
                 "Should be at least machine width + 2× conveyor belt length + AGV lane. " +
                 "With default belt settings (capacity 3, spacing 0.5 = 1 unit per belt) " +
                 "a value of 5–6 gives comfortable clearance.")]
        [SerializeField] private float minCellSize = 5.5f;

        [Tooltip("If true, log the distance matrix to the console on build.")]
        [SerializeField] private bool logDistanceMatrix = true;

        // ─────────────────────────────────────────────────────────
        //  Runtime State
        // ─────────────────────────────────────────────────────────

        private PhysicalMachine[] machines;
        private float[,] distanceMatrix;
        private float[] distanceMatrixFlat;
        private Vector3[] customPositions;

        // ─────────────────────────────────────────────────────────
        //  Public Accessors
        // ─────────────────────────────────────────────────────────

        /// @brief Read-only view of all instantiated physical machines.
        public IReadOnlyList<PhysicalMachine> Machines => machines;

        /// @brief Number of machines currently on the floor.
        public int MachineCount => machines?.Length ?? 0;

        /// @brief Raw pairwise world-unit distance matrix (n × n).
        public float[,] DistanceMatrix => distanceMatrix;

        /// @brief Normalised flat distance matrix padded to 8×8 = 64 values.
        public float[] DistanceMatrixFlat => distanceMatrixFlat;

        // ─────────────────────────────────────────────────────────
        //  Public API
        // ─────────────────────────────────────────────────────────

        /// @brief Clears any existing floor, then instantiates and initialises
        ///        machines according to the simulator's machine count.
        /// @param simulator  Loaded @c DESSimulator carrying machine metadata.
        /// @exception ArgumentNullException Thrown when @p simulator is null.
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

                PhysicalMachine physicalMachine = Instantiate(machinePrefab, worldPos, Quaternion.identity, transform);
                physicalMachine.gameObject.name = $"Machine_{i}";
                physicalMachine.Initialize(i, simulator.Machines[i]);
                machines[i] = physicalMachine;
            }

            ComputeDistanceMatrix();

            if (logDistanceMatrix)
                LogDistanceMatrix();

            navMeshSurface.BuildNavMesh();

            SimLogger.Medium($"[FactoryLayout] Built floor: {count} machines, " +
                      $"floor {floorSize.x}×{floorSize.y}");
        }

        /// @brief Destroys all instantiated machines and resets runtime state.
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
            navMeshSurface.RemoveData();
        }

        /// @brief Overrides the automatic layout with a caller-supplied position array.
        /// @param positions  World-space offsets relative to @c floorTransform, one per machine.
        public void SetCustomLayout(Vector3[] positions)
        {
            customPositions = positions;
        }

        /// @brief Returns the @c PhysicalMachine at the given index, or @c null if out of range.
        /// @param machineId  Zero-based machine index.
        /// @return           The corresponding @c PhysicalMachine, or @c null.
        public PhysicalMachine GetMachine(int machineId)
        {
            if (machines == null || machineId < 0 || machineId >= machines.Length)
                return null;
            return machines[machineId];
        }

        // ─────────────────────────────────────────────────────────
        //  Position Resolution
        // ─────────────────────────────────────────────────────────

        /// @brief Returns the appropriate position array for @p count machines,
        ///        preferring a custom layout, then hard-coded layouts, then a grid.
        /// @param count  Number of machines to place.
        /// @return       Array of local offsets relative to @c floorTransform.
        /// @exception InvalidOperationException Thrown when a custom layout has the wrong count.
        private Vector3[] ResolvePositions(int count)
        {
            if (customPositions != null)
            {
                if (customPositions.Length != count)
                    throw new InvalidOperationException(
                        $"Custom layout has {customPositions.Length} positions " +
                        $"but simulator has {count} machines.");
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

        // ─────────────────────────────────────────────────────────
        //  Grid Generator
        // ─────────────────────────────────────────────────────────

        /// @brief Generates a uniform grid layout for an arbitrary machine count.
        ///        Enforces @c minCellSize so machines never crowd each other even
        ///        on small floors or with large machine counts.
        /// @param count  Number of machines to position.
        /// @return       Array of local offsets centred on the floor.
        private Vector3[] GenerateGridPositions(int count)
        {
            int cols = Mathf.CeilToInt(Mathf.Sqrt(count));
            int rows = Mathf.CeilToInt((float)count / cols);

            float margin = 2f;

            // Compute spacing from available floor area, but never go below
            // minCellSize so conveyors and AGV lanes have room.
            float spacingX = Mathf.Max(
                (floorSize.x - margin * 2) / Mathf.Max(cols - 1, 1),
                minCellSize);
            float spacingZ = Mathf.Max(
                (floorSize.y - margin * 2) / Mathf.Max(rows - 1, 1),
                minCellSize);

            // Centre the grid on the floor
            float originX = -((cols - 1) * spacingX) / 2f;
            float originZ = -((rows - 1) * spacingZ) / 2f;

            Vector3[] positions = new Vector3[count];
            for (int i = 0; i < count; i++)
            {
                int col = i % cols;
                int row = i / cols;
                positions[i] = new Vector3(
                    originX + col * spacingX,
                    0f,
                    originZ + row * spacingZ);
            }

            // If the grid exceeds the floor, warn but don't clamp — the user
            // can increase floorSize or lower minCellSize in the inspector.
            float neededX = (cols - 1) * spacingX + margin * 2;
            float neededZ = (rows - 1) * spacingZ + margin * 2;
            if (neededX > floorSize.x || neededZ > floorSize.y)
            {
                SimLogger.Error(
                    $"[FactoryLayout] Grid needs {neededX:F0}×{neededZ:F0} but floor is " +
                    $"{floorSize.x}×{floorSize.y}. Increase Floor Size or reduce Min Cell Size.");
            }

            return positions;
        }

        // ─────────────────────────────────────────────────────────
        //  Hard-Coded Layouts  (widened for conveyor clearance)
        // ─────────────────────────────────────────────────────────

        /// @brief Symmetric 2-row layout for 8 machines.
        ///        X spacing: 6 units  |  Z spacing: 6 units
        private static Vector3[] GetLayout8()
        {
            return new[]
            {
                new Vector3(-9f,  0f, -3f),
                new Vector3(-3f,  0f, -3f),
                new Vector3( 3f,  0f, -3f),
                new Vector3( 9f,  0f, -3f),

                new Vector3(-9f,  0f,  3f),
                new Vector3(-3f,  0f,  3f),
                new Vector3( 3f,  0f,  3f),
                new Vector3( 9f,  0f,  3f),
            };
        }

        /// @brief 5×3 grid layout for 15 machines.
        ///        X spacing: 5.5 units  |  Z spacing: 6 units
        private static Vector3[] GetLayout15()
        {
            Vector3[] positions = new Vector3[15];
            float[] xSlots = { -11f, -5.5f, 0f, 5.5f, 11f };
            float[] zRows = { -6f, 0f, 6f };
            int index = 0;
            for (int row = 0; row < 3; row++)
                for (int col = 0; col < 5; col++)
                    positions[index++] = new Vector3(xSlots[col], 0f, zRows[row]);
            return positions;
        }

        /// @brief 5×4 grid layout for 20 machines.
        ///        X spacing: 5.5 units  |  Z spacing: 5 units
        private static Vector3[] GetLayout20()
        {
            Vector3[] positions = new Vector3[20];
            float[] xSlots = { -11f, -5.5f, 0f, 5.5f, 11f };
            float[] zRows = { -7.5f, -2.5f, 2.5f, 7.5f };
            int index = 0;
            for (int row = 0; row < 4; row++)
                for (int col = 0; col < 5; col++)
                    positions[index++] = new Vector3(xSlots[col], 0f, zRows[row]);
            return positions;
        }

        // ─────────────────────────────────────────────────────────
        //  Distance Matrix
        // ─────────────────────────────────────────────────────────

        /// @brief Builds a symmetric n×n Euclidean distance matrix and a
        ///        normalised 64-element flat version capped at the first 8 machines.
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

        /// @brief Prints the full distance matrix and the normalised flat vector to the console.
        private void LogDistanceMatrix()
        {
            if (distanceMatrix == null) return;
            int n = machines.Length;
            string header = "     ";
            for (int j = 0; j < n; j++) header += $"  M{j,-4}";
            SimLogger.Medium($"[FactoryLayout] Distance matrix ({n}×{n}) — world units:\n{header}");

            for (int i = 0; i < n; i++)
            {
                string row = $"M{i,-3} ";
                for (int j = 0; j < n; j++) row += $"{distanceMatrix[i, j],6:F1} ";
                SimLogger.High(row);
            }
            SimLogger.Medium($"[FactoryLayout] Flat 64-D (normalised): [{string.Join(", ", FormatFlat())}]");
        }

        /// @brief Formats the flat distance array as fixed-precision strings for logging.
        private string[] FormatFlat()
        {
            string[] result = new string[distanceMatrixFlat.Length];
            for (int i = 0; i < distanceMatrixFlat.Length; i++)
                result[i] = distanceMatrixFlat[i].ToString("F2");
            return result;
        }

        // ─────────────────────────────────────────────────────────
        //  Editor Gizmos
        // ─────────────────────────────────────────────────────────

        /// @brief Draws floor bounds and an 8-machine layout preview in the Scene view.
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