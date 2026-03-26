using System;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;

namespace Assets.Scripts.Simulation
{
    /// @file FactoryLayoutManager.cs
    /// @brief Spawns machine prefabs on a spatial grid and computes the pairwise distance matrix.
    ///
    /// @details This is the entry point for building the visual factory floor.
    /// It reads the machine count from a @ref DESSimulator instance, places
    /// @ref MachinePrefab instances at predetermined grid positions, and produces
    /// the flattened 64-D distance matrix that feeds the observation space.
    ///
    /// Layout tables are provided for 8, 15, and 20 machines on a 20×20 unit floor.
    /// Custom layouts can be supplied at runtime via @ref SetCustomLayout.
    ///
    /// @par Phase 1 coverage
    /// - Step 1 — Scene skeleton and grid layout
    /// - Partial Step 5 — Exposes machine visuals array for SimulationBridge wiring

    public class FactoryLayoutManager : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Inspector
        // ─────────────────────────────────────────────────────────

        [Header("Prefabs")]
        [Tooltip("Machine prefab with a MachineVisual component attached.")]
        [SerializeField] private MachineVisual machinePrefab;

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

        /// @brief All instantiated machine visuals, indexed by machine ID.
        private MachineVisual[] machineVisuals;

        /// @brief NxN pairwise distance matrix (world-space Euclidean).
        private float[,] distanceMatrix;

        /// @brief Flattened distance matrix for the observation space.
        /// @details Always 64 elements (8×8). Padded with zeros when fewer
        /// than 8 machines are present, truncated if more than 8 — matching
        /// the fixed observation shape defined in the architecture.
        private float[] distanceMatrixFlat;

        /// @brief Optional user-supplied grid positions. If non-null, overrides
        /// the built-in layout tables.
        private Vector3[] customPositions;

        // ─────────────────────────────────────────────────────────
        //  Public accessors
        // ─────────────────────────────────────────────────────────

        /// @brief Read-only access to all spawned machine visuals.
        public IReadOnlyList<MachineVisual> MachineVisuals => machineVisuals;

        /// @brief Number of machines currently on the floor.
        public int MachineCount => machineVisuals?.Length ?? 0;

        /// @brief The full NxN distance matrix (world-space units).
        public float[,] DistanceMatrix => distanceMatrix;

        /// @brief The flattened 64-D distance vector for the observation space.
        public float[] DistanceMatrixFlat => distanceMatrixFlat;

        // ─────────────────────────────────────────────────────────
        //  Public API
        // ─────────────────────────────────────────────────────────

        /// @brief Builds the factory floor for the given simulator.
        ///
        /// @details Instantiates one @ref MachinePrefab per machine in the
        /// simulator, places them on the grid, wires each @ref MachineVisual
        /// to its core @ref Machine, and computes the distance matrix.
        ///
        /// Call this once per episode from @ref SimulationBridge after
        /// the DES has been initialised with a Taillard instance.
        ///
        /// @param simulator The initialised DES simulator whose @c Machines
        ///        array determines how many prefabs to spawn.
        public void BuildFloor(DESSimulator simulator)
        {
            if (simulator == null)
                throw new ArgumentNullException(nameof(simulator));

            ClearFloor();

            int count = simulator.Machines.Length;
            Vector3[] positions = ResolvePositions(count);
            machineVisuals = new MachineVisual[count];

            Vector3 floorCentre = floorTransform != null
                ? floorTransform.position
                : Vector3.zero;

            for (int i = 0; i < count; i++)
            {
                Vector3 worldPos = floorCentre + positions[i];
                worldPos.y = machineYOffset;

                MachineVisual visual = Instantiate(machinePrefab, worldPos, Quaternion.identity, transform);
                visual.gameObject.name = $"Machine_{i}";
                visual.Initialise(i, simulator.Machines[i]);
                machineVisuals[i] = visual;
            }

            ComputeDistanceMatrix();

            if (logDistanceMatrix)
                LogDistanceMatrix();

            Debug.Log($"[FactoryLayout] Built floor: {count} machines, " +
                      $"floor {floorSize.x}×{floorSize.y}");
        }

        /// @brief Destroys all spawned machine visuals and clears cached data.
        ///
        /// @details Called automatically at the start of @ref BuildFloor so
        /// consecutive episodes don't accumulate GameObjects. Safe to call
        /// externally for a manual teardown.
        public void ClearFloor()
        {
            if (machineVisuals == null) return;

            foreach (MachineVisual mv in machineVisuals)
            {
                if (mv != null)
                    Destroy(mv.gameObject);
            }

            machineVisuals = null;
            distanceMatrix = null;
            distanceMatrixFlat = null;
        }

        /// @brief Supplies a custom set of grid positions, bypassing the
        /// built-in layout tables.
        ///
        /// @details Call before @ref BuildFloor. The array length must match
        /// the simulator's machine count or an exception is thrown at build time.
        ///
        /// @param positions World-space offsets from the floor centre (Y is ignored).
        public void SetCustomLayout(Vector3[] positions)
        {
            customPositions = positions;
        }

        /// @brief Returns the @ref MachineVisual for a given machine ID.
        ///
        /// @param machineId Zero-based machine index.
        /// @returns The corresponding visual, or @c null if the floor has not been built.
        public MachineVisual GetMachineVisual(int machineId)
        {
            if (machineVisuals == null || machineId < 0 || machineId >= machineVisuals.Length)
                return null;
            return machineVisuals[machineId];
        }

        // ─────────────────────────────────────────────────────────
        //  Position resolution
        // ─────────────────────────────────────────────────────────

        /// @brief Selects grid positions for @p count machines.
        ///
        /// @details Checks for a custom layout first, then falls back to
        /// hardcoded tables for 8, 15, and 20 machines. If no table matches,
        /// generates positions automatically with @ref GenerateGridPositions.
        ///
        /// @param count Number of machines to place.
        /// @returns Array of local-space offset positions (Y = 0).
        private Vector3[] ResolvePositions(int count)
        {
            if (customPositions != null)
            {
                if (customPositions.Length != count)
                    throw new InvalidOperationException(
                        $"Custom layout has {customPositions.Length} positions but simulator has {count} machines.");
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

        /// @brief Automatic grid generation for arbitrary machine counts.
        ///
        /// @details Arranges machines in a roughly square grid centred on the
        /// floor. Spacing is computed from @ref floorSize with a margin so
        /// machines don't sit on the very edge.
        ///
        /// @param count Number of machines.
        /// @returns Array of centred grid positions.
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
                positions[i] = new Vector3(
                    originX + col * spacingX,
                    0f,
                    originZ + row * spacingZ
                );
            }

            return positions;
        }

        // ─────────────────────────────────────────────────────────
        //  Hardcoded layout tables
        // ─────────────────────────────────────────────────────────

        /// @brief 8-machine layout — 2 rows × 4 columns with aisle gaps.
        ///
        /// @details Designed for the Taillard ta01–ta10 (15×15) instances
        /// when using the 8-machine distance-matrix observation.
        /// Positioned on a 20×20 floor with ~4-unit column spacing
        /// and ~6-unit row spacing, leaving wide aisles for AGV pathing.
        ///
        /// @verbatim
        ///   Row 1:  M0   M1   M2   M3       z = +3
        ///                                    (aisle)
        ///   Row 0:  M4   M5   M6   M7       z = -3
        ///
        ///           x: -6  -2  +2  +6
        /// @endverbatim
        private static Vector3[] GetLayout8()
        {
            return new[]
            {
                // Row 0 (front)
                new Vector3(-6f, 0f, -3f),  // M0
                new Vector3(-2f, 0f, -3f),  // M1
                new Vector3( 2f, 0f, -3f),  // M2
                new Vector3( 6f, 0f, -3f),  // M3
                // Row 1 (back)
                new Vector3(-6f, 0f,  3f),  // M4
                new Vector3(-2f, 0f,  3f),  // M5
                new Vector3( 2f, 0f,  3f),  // M6
                new Vector3( 6f, 0f,  3f),  // M7
            };
        }

        /// @brief 15-machine layout — 3 rows × 5 columns.
        ///
        /// @details Covers Taillard ta01–ta10 (15 jobs × 15 machines).
        /// Column spacing ~3.5 units, row spacing ~4 units.
        ///
        /// @verbatim
        ///   Row 2:  M10  M11  M12  M13  M14     z = +4
        ///   Row 1:  M5   M6   M7   M8   M9      z =  0
        ///   Row 0:  M0   M1   M2   M3   M4      z = -4
        ///
        ///           x: -7  -3.5  0  +3.5  +7
        /// @endverbatim
        private static Vector3[] GetLayout15()
        {
            Vector3[] positions = new Vector3[15];
            float[] xSlots = { -7f, -3.5f, 0f, 3.5f, 7f };
            float[] zRows = { -4f, 0f, 4f };

            int index = 0;
            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 5; col++)
                {
                    positions[index++] = new Vector3(xSlots[col], 0f, zRows[row]);
                }
            }

            return positions;
        }

        /// @brief 20-machine layout — 4 rows × 5 columns.
        ///
        /// @details Covers Taillard ta11–ta20 (20 jobs × 20 machines).
        /// Tighter spacing (~3.5 × 3 units) to fit within the 20×20 floor.
        ///
        /// @verbatim
        ///   Row 3:  M15  M16  M17  M18  M19     z = +4.5
        ///   Row 2:  M10  M11  M12  M13  M14     z = +1.5
        ///   Row 1:  M5   M6   M7   M8   M9      z = -1.5
        ///   Row 0:  M0   M1   M2   M3   M4      z = -4.5
        ///
        ///           x: -7  -3.5  0  +3.5  +7
        /// @endverbatim
        private static Vector3[] GetLayout20()
        {
            Vector3[] positions = new Vector3[20];
            float[] xSlots = { -7f, -3.5f, 0f, 3.5f, 7f };
            float[] zRows = { -4.5f, -1.5f, 1.5f, 4.5f };

            int index = 0;
            for (int row = 0; row < 4; row++)
            {
                for (int col = 0; col < 5; col++)
                {
                    positions[index++] = new Vector3(xSlots[col], 0f, zRows[row]);
                }
            }

            return positions;
        }

        // ─────────────────────────────────────────────────────────
        //  Distance matrix
        // ─────────────────────────────────────────────────────────

        /// @brief Computes the NxN pairwise Euclidean distance matrix from
        /// the actual world-space transforms of the spawned machines.
        ///
        /// @details Also produces @ref distanceMatrixFlat, an 8×8 = 64-element
        /// array that maps directly to the @c distance_matrix observation
        /// channel. When fewer than 8 machines exist the trailing elements
        /// are zero-padded; when more than 8 exist only the first 8 are used.
        ///
        /// Distances are computed on the XZ plane (Y is ignored) since
        /// all machines share the same vertical position.
        private void ComputeDistanceMatrix()
        {
            int n = machineVisuals.Length;

            // ── Full NxN matrix ─────────────────────────────────
            distanceMatrix = new float[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    Vector3 a = machineVisuals[i].transform.position;
                    Vector3 b = machineVisuals[j].transform.position;

                    // XZ-plane distance (ignore vertical offset)
                    float dx = a.x - b.x;
                    float dz = a.z - b.z;
                    float dist = Mathf.Sqrt(dx * dx + dz * dz);

                    distanceMatrix[i, j] = dist;
                    distanceMatrix[j, i] = dist;
                }
            }

            // ── Flattened 8×8 for observation space ─────────────
            const int obsSize = 8;
            distanceMatrixFlat = new float[obsSize * obsSize];

            int limit = Mathf.Min(n, obsSize);
            for (int i = 0; i < limit; i++)
            {
                for (int j = 0; j < limit; j++)
                {
                    distanceMatrixFlat[i * obsSize + j] = distanceMatrix[i, j];
                }
            }

            // Normalise to [0, 1] by dividing by the maximum distance.
            float maxDist = 0f;
            for (int k = 0; k < distanceMatrixFlat.Length; k++)
            {
                if (distanceMatrixFlat[k] > maxDist)
                    maxDist = distanceMatrixFlat[k];
            }

            if (maxDist > 0f)
            {
                for (int k = 0; k < distanceMatrixFlat.Length; k++)
                {
                    distanceMatrixFlat[k] /= maxDist;
                }
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Debug
        // ─────────────────────────────────────────────────────────

        /// @brief Logs the full NxN distance matrix and the normalised
        /// 64-D flat vector to the Unity console.
        private void LogDistanceMatrix()
        {
            if (distanceMatrix == null) return;

            int n = machineVisuals.Length;

            // Header row
            string header = "     ";
            for (int j = 0; j < n; j++)
                header += $"  M{j,-4}";
            Debug.Log($"[FactoryLayout] Distance matrix ({n}×{n}) — world units:\n{header}");

            for (int i = 0; i < n; i++)
            {
                string row = $"M{i,-3} ";
                for (int j = 0; j < n; j++)
                    row += $"{distanceMatrix[i, j],6:F1} ";
                Debug.Log(row);
            }

            Debug.Log($"[FactoryLayout] Flat 64-D (normalised): [{string.Join(", ", FormatFlat())}]");
        }

        /// @brief Formats the flat distance vector as an array of
        /// two-decimal strings for logging.
        private string[] FormatFlat()
        {
            string[] result = new string[distanceMatrixFlat.Length];
            for (int i = 0; i < distanceMatrixFlat.Length; i++)
                result[i] = distanceMatrixFlat[i].ToString("F2");
            return result;
        }

        // ─────────────────────────────────────────────────────────
        //  Gizmos — editor-only layout preview
        // ─────────────────────────────────────────────────────────

        /// @brief Draws grid position previews in the Scene view when
        /// the manager is selected, even before Play mode.
        private void OnDrawGizmosSelected()
        {
            if (floorTransform == null) return;

            Vector3 centre = floorTransform.position;

            // Draw floor bounds
            Gizmos.color = new Color(1f, 1f, 1f, 0.15f);
            Gizmos.DrawWireCube(
                centre + Vector3.up * 0.01f,
                new Vector3(floorSize.x, 0f, floorSize.y)
            );

            // Preview the 8-machine layout as a sensible default
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