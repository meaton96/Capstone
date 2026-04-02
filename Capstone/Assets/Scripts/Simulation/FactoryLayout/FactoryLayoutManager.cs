using System;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;
using Assets.Scripts.Logging;
using Unity.AI.Navigation;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation.FactoryLayout
{
    /// <summary>
    /// Builds an aisle-based factory floor with one-way traffic lanes,
    /// physical wall colliders, and zone markers.
    ///
    /// </summary>
    public class FactoryLayoutManager : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Inspector — References
        // ─────────────────────────────────────────────────────────
        public static FactoryLayoutManager Instance;
        [SerializeField] private NavMeshSurface navMeshSurface;

        [Header("Prefabs")]
        [SerializeField] private PhysicalMachine machinePrefab;
        [SerializeField] private PhysicalMachine doubleSidedMachinePrefab;

        [Header("I/O & AGV Infrastructure")]
        [SerializeField] private GameObject conveyorPrefab;
        [SerializeField] private Vector3 ioConveyorScale = new Vector3(.05f, .2f, .30f);
        [SerializeField] private Material incomingBeltMaterial;

        public Vector3 IncomingBeltPosition { get; private set; }
        public Vector3 OutgoingBeltPosition { get; private set; }
        public Vector3 AGVParkingPosition { get; private set; }

        public ConveyorBelt IncomingBelt { get; private set; }
        public ConveyorBelt OutgoingBelt { get; private set; }
        // public Vector3 AGVParkingPosition { get; private set; }

        [Tooltip("Optional prefab for aisle wall segments. If null, a " +
                 "primitive cube is generated at runtime.")]
        [SerializeField] private GameObject wallPrefab;

        [Header("Floor")]
        [SerializeField] private Transform floorTransform;

        // ─────────────────────────────────────────────────────────
        //  Inspector — Dimensions (world units)
        // ─────────────────────────────────────────────────────────

        [Header("Machine Grid")]
        [Tooltip("Horizontal centre-to-centre distance between adjacent " +
                 "machines in the same row.")]
        [SerializeField] private float machineSpacingX = 6f;

        [Tooltip("Depth of the machine body along the row-perpendicular axis.")]
        [SerializeField] private float machineDepth = 1.5f;

        [Tooltip("How far conveyors extend from the machine face into the aisle.")]
        [SerializeField] private float conveyorReach = 1.5f;

        [Header("Aisles")]
        [Tooltip("Clear width of narrow row aisles between machine rows. " +
                 "Should fit exactly one AGV plus safety margin (~3 m).")]
        [SerializeField] private float rowAisleWidth = 3f;

        [Tooltip("Clear width of the top/bottom peripheral spine aisles.")]
        [SerializeField] private float spineAisleWidth = 4f;

        [Tooltip("Clear width of the left/right vertical connector aisles.")]
        [SerializeField] private float verticalAisleWidth = 3.5f;

        [Header("Walls")]
        [Tooltip("Height of aisle boundary walls.")]
        [SerializeField] private float wallHeight = 0.6f;

        [Tooltip("Thickness of the wall colliders.")]
        [SerializeField] private float wallThickness = 0.15f;

        [Header("Visual")]
        [Tooltip("Material for floor direction arrows (optional).")]
        [SerializeField] private Material arrowMaterial;

        [Tooltip("Vertical offset so machines sit on the floor plane.")]
        [SerializeField] private float machineYOffset = 0.5f;

        [Header("Debug")]
        [SerializeField] private bool logDistanceMatrix = true;

        // ─────────────────────────────────────────────────────────
        //  Runtime State
        // ─────────────────────────────────────────────────────────

        private PhysicalMachine[] machines;
        private float[,] distanceMatrix;
        private float[] distanceMatrixFlat;
        private Vector3[] customPositions;
        private readonly List<GameObject> spawnedObjects = new List<GameObject>();

        // Computed layout metadata exposed to TrafficZoneManager
        private int layoutRows;
        private int layoutCols;
        private float totalFloorWidth;
        private float totalFloorDepth;

        // ─────────────────────────────────────────────────────────
        //  Public Accessors
        // ─────────────────────────────────────────────────────────

        public IReadOnlyList<PhysicalMachine> Machines => machines;
        public int MachineCount => machines?.Length ?? 0;
        public float[,] DistanceMatrix => distanceMatrix;
        public float[] DistanceMatrixFlat => distanceMatrixFlat;

        /// <summary>Number of machine rows in the current layout.</summary>
        public int LayoutRows => layoutRows;

        /// <summary>Number of machine columns in the current layout.</summary>
        public int LayoutCols => layoutCols;

        /// <summary>Row-perpendicular distance between adjacent machine row centres.</summary>
        public float RowPitch => machineDepth + conveyorReach * 2f + rowAisleWidth;

        public float MachineSpacingX => machineSpacingX;
        public float SpineAisleWidth => spineAisleWidth;
        public float VerticalAisleWidth => verticalAisleWidth;
        public float RowAisleWidth => rowAisleWidth;
        public float ConveyorReach => conveyorReach;
        public float MachineDepth => machineDepth;

        /// <summary>
        /// World-space origin of the layout grid (top-left corner of
        /// the machine area, excluding vertical/spine aisles).
        /// </summary>
        public Vector3 GridOrigin { get; private set; }

        /// <summary>Total floor bounds in world units.</summary>
        public Vector2 FloorSize => new Vector2(totalFloorWidth, totalFloorDepth);

        void Awake()
        {
            Instance = this;
        }

        // ─────────────────────────────────────────────────────────
        //  Public API
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Clears any existing floor, then instantiates machines in an
        /// aisle-based layout with physical walls and floor markings.
        /// </summary>
        public void BuildFloor(DESSimulator simulator)
        {
            if (simulator == null)
                throw new ArgumentNullException(nameof(simulator));

            ClearFloor();

            int count = simulator.Machines.Length;

            // Decide grid shape
            layoutCols = Mathf.CeilToInt(Mathf.Sqrt(count));
            layoutRows = Mathf.CeilToInt((float)count / layoutCols);

            // Compute total floor dimensions
            float machineAreaWidth = (layoutCols - 1) * machineSpacingX + machineDepth;
            float machineAreaDepth = (layoutRows - 1) * RowPitch + machineDepth;

            totalFloorWidth = verticalAisleWidth + machineAreaWidth + verticalAisleWidth;
            totalFloorDepth = spineAisleWidth + machineAreaDepth + spineAisleWidth;

            // Resize and position floor plane
            if (floorTransform != null)
            {
                floorTransform.localScale = new Vector3(
                    totalFloorWidth / 10f, 1f, totalFloorDepth / 10f);
            }

            Vector3 floorCentre = floorTransform != null
                ? floorTransform.position
                : Vector3.zero;

            // Grid origin: top-left of machine area
            GridOrigin = floorCentre + new Vector3(
                -machineAreaWidth / 2f,
                0f,
                machineAreaDepth / 2f);

            // Place machines
            machines = new PhysicalMachine[count];
            for (int i = 0; i < count; i++)
            {
                int col = i % layoutCols;
                int row = i / layoutCols;

                Vector3 localPos = GetMachineLocalPosition(row, col);
                Vector3 worldPos = floorCentre + localPos;
                worldPos.y = machineYOffset;

                PhysicalMachine prefabToSpawn;
                Quaternion rotation;

                if (row == 0)
                {
                    // Top Row: Single-sided prefab, flipped 180 to face South
                    prefabToSpawn = machinePrefab;
                    rotation = Quaternion.Euler(0f, 180f, 0f);
                }
                else if (row == layoutRows - 1)
                {
                    // Bottom Row: Single-sided prefab, default rotation to face North
                    prefabToSpawn = machinePrefab;
                    rotation = Quaternion.identity;
                }
                else
                {
                    // Center Rows: Double-sided prefab
                    // (Includes a fallback just in case you forget to assign it in the inspector)
                    prefabToSpawn = doubleSidedMachinePrefab != null ? doubleSidedMachinePrefab : machinePrefab;
                    rotation = Quaternion.identity;
                }

                PhysicalMachine pm = Instantiate(prefabToSpawn, worldPos, rotation, transform);
                pm.gameObject.name = $"Machine_{i}";
                pm.Initialize(i, simulator.Machines[i]);
                machines[i] = pm;
            }

            // Build physical environment
            BuildAisleWalls(floorCentre);
            BuildFloorArrows(floorCentre);

            ComputeDistanceMatrix();
            if (logDistanceMatrix) LogDistanceMatrix();


            BuildInfrastructure(floorCentre);

            // Rebuild NavMesh with walls in place
            navMeshSurface.BuildNavMesh();

            SimLogger.Medium($"[FactoryLayout] Built aisle-based floor: " +
                $"{count} machines ({layoutRows}×{layoutCols}), " +
                $"floor {totalFloorWidth:F1}×{totalFloorDepth:F1}, " +
                $"row pitch {RowPitch:F1}");
        }

        private void BuildInfrastructure(Vector3 floorCentre)
        {
            float machineAreaHalfW = ((layoutCols - 1) * machineSpacingX) / 2f;

            // A) INCOMING BELT (Top-Left, facing East onto the top spine)
            float topZ = floorCentre.z + GetTopSpineZ();
            IncomingBeltPosition = new Vector3(floorCentre.x - machineAreaHalfW - verticalAisleWidth, 0.01f, topZ);

            if (conveyorPrefab != null)
            {
                GameObject inBelt = Instantiate(conveyorPrefab, IncomingBeltPosition, Quaternion.Euler(0, 90, 0), transform);
                inBelt.name = "Incoming_Belt";
                inBelt.transform.localScale = ioConveyorScale;
                IncomingBelt = inBelt.GetComponent<ConveyorBelt>();
                if (incomingBeltMaterial != null)
                {
                    foreach (var rend in inBelt.GetComponentsInChildren<Renderer>())
                    {
                        rend.material = incomingBeltMaterial;
                    }
                }
                spawnedObjects.Add(inBelt);
            }

            // B) OUTGOING BELT (Bottom-Right, facing South/Out of the factory)
            float botZ = floorCentre.z + GetBottomSpineZ();
            OutgoingBeltPosition = new Vector3(floorCentre.x + machineAreaHalfW + verticalAisleWidth, 0.01f, botZ);

            if (conveyorPrefab != null)
            {
                GameObject outBelt = Instantiate(conveyorPrefab, OutgoingBeltPosition, Quaternion.Euler(0, 180, 0), transform);
                outBelt.name = "Outgoing_Belt";
                outBelt.transform.localScale = ioConveyorScale;
                spawnedObjects.Add(outBelt);
                OutgoingBelt = outBelt.GetComponent<ConveyorBelt>();
            }

            // C) AGV PARKING (Bottom-Left)
            AGVParkingPosition = new Vector3(floorCentre.x - machineAreaHalfW, 0.01f, botZ);
        }

        public void ClearFloor()
        {
            if (machines != null)
            {
                foreach (PhysicalMachine pm in machines)
                    if (pm != null) Destroy(pm.gameObject);
            }

            foreach (GameObject obj in spawnedObjects)
                if (obj != null) Destroy(obj);

            spawnedObjects.Clear();
            machines = null;
            distanceMatrix = null;
            distanceMatrixFlat = null;

            if (navMeshSurface != null)
                navMeshSurface.RemoveData();
        }

        public void SetCustomLayout(Vector3[] positions) =>
            customPositions = positions;

        public PhysicalMachine GetMachine(int machineId)
        {
            if (machines == null || machineId < 0 || machineId >= machines.Length)
                return null;
            return machines[machineId];
        }

        // ─────────────────────────────────────────────────────────
        //  Position Calculation
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Returns the local position (relative to floor centre) for
        /// the machine at the given grid row and column.
        /// </summary>
        public Vector3 GetMachineLocalPosition(int row, int col)
        {
            float machineAreaWidth = (layoutCols - 1) * machineSpacingX;
            float machineAreaDepth = (layoutRows - 1) * RowPitch;

            float x = -machineAreaWidth / 2f + col * machineSpacingX;
            float z = machineAreaDepth / 2f - row * RowPitch;

            return new Vector3(x, 0f, z);
        }

        /// <summary>
        /// Returns the world-space centre of the row aisle between
        /// machine row <paramref name="rowAbove"/> and the row below it.
        /// Row aisle index 0 is between machine row 0 and row 1.
        /// </summary>
        public Vector3 GetRowAisleCentre(int aisleIndex)
        {
            Vector3 floorCentre = floorTransform != null
                ? floorTransform.position : Vector3.zero;

            float machineAreaDepth = (layoutRows - 1) * RowPitch;

            // Aisle sits between row[aisleIndex] and row[aisleIndex+1]
            float zTop = machineAreaDepth / 2f - aisleIndex * RowPitch;
            float zBot = zTop - RowPitch;
            float aisleZ = (zTop + zBot) / 2f;

            return floorCentre + new Vector3(0f, 0.01f, aisleZ);
        }

        /// <summary>
        /// Returns the flow direction for a given row aisle.
        /// Even-indexed aisles flow east (+X), odd aisles flow west (-X).
        /// </summary>
        public Vector3 GetRowAisleDirection(int aisleIndex)
        {
            return (aisleIndex % 2 == 0) ? Vector3.right : Vector3.left;
        }

        /// <summary>
        /// Returns the Z coordinate of the top spine aisle centre.
        /// </summary>
        public float GetTopSpineZ()
        {
            float machineAreaDepth = (layoutRows - 1) * RowPitch + machineDepth;
            return machineAreaDepth / 2f + spineAisleWidth / 2f;
        }

        /// <summary>
        /// Returns the Z coordinate of the bottom spine aisle centre.
        /// </summary>
        public float GetBottomSpineZ()
        {
            float machineAreaDepth = (layoutRows - 1) * RowPitch + machineDepth;
            return -(machineAreaDepth / 2f + spineAisleWidth / 2f);
        }

        // ─────────────────────────────────────────────────────────
        //  Aisle Wall Construction
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Spawns thin box colliders along the north and south edges of
        /// each machine row, forming physical aisle boundaries that the
        /// NavMesh will carve around.
        /// </summary>
        private void BuildAisleWalls(Vector3 floorCentre)
        {
            float machineAreaWidth = (layoutCols - 1) * machineSpacingX + machineDepth;
            float halfMachineDepth = machineDepth / 2f;

            for (int row = 0; row < layoutRows; row++)
            {
                Vector3 rowCentre = floorCentre + GetMachineLocalPosition(row, 0);
                rowCentre.x = floorCentre.x;
                rowCentre.y = wallHeight / 2f;

                // North wall of this machine row
                Vector3 northWallPos = rowCentre + Vector3.forward * (halfMachineDepth + 0.05f);
                SpawnWallSegment(northWallPos, machineAreaWidth, row, "North");

                // South wall of this machine row
                Vector3 southWallPos = rowCentre - Vector3.forward * (halfMachineDepth + 0.05f);
                SpawnWallSegment(southWallPos, machineAreaWidth, row, "South");
            }

            // Vertical aisle outer walls (left and right boundaries)
            float machineAreaDepth = (layoutRows - 1) * RowPitch + machineDepth;
            float fullHeight = machineAreaDepth + spineAisleWidth * 2f;

            // Left outer wall
            Vector3 leftWall = floorCentre + new Vector3(
                -(machineAreaWidth / 2f + verticalAisleWidth), wallHeight / 2f, 0f);
            SpawnWallSegmentVertical(leftWall, fullHeight, "LeftOuter");

            // Right outer wall
            Vector3 rightWall = floorCentre + new Vector3(
                machineAreaWidth / 2f + verticalAisleWidth, wallHeight / 2f, 0f);
            SpawnWallSegmentVertical(rightWall, fullHeight, "RightOuter");
        }

        private void SpawnWallSegment(Vector3 position, float length, int row, string side)
        {
            // Skip wall segments where conveyors need to poke through.
            // Instead of one long wall, spawn segments between machine columns
            // with gaps for conveyor access.

            float gapWidth = 1.2f; // clearance for conveyor + AGV docking
            float segStart = -length / 2f;

            for (int col = 0; col <= layoutCols; col++)
            {
                float colX = 0;
                float segEnd;

                if (col < layoutCols)
                {
                    colX = -((layoutCols - 1) * machineSpacingX) / 2f
                           + col * machineSpacingX;
                    segEnd = colX - gapWidth / 2f;
                }
                else
                {
                    segEnd = length / 2f;
                }

                float segLength = segEnd - segStart;
                if (segLength > 0.3f)
                {
                    Vector3 segPos = position;
                    segPos.x += (segStart + segEnd) / 2f;

                    GameObject wall = CreateWallPrimitive(
                        new Vector3(segLength, wallHeight, wallThickness));
                    wall.transform.position = segPos;
                    wall.name = $"Wall_Row{row}_{side}_Seg{col}";
                    wall.transform.parent = transform;
                    spawnedObjects.Add(wall);
                }

                if (col < layoutCols)
                    segStart = colX + gapWidth / 2f;
            }
        }

        private void SpawnWallSegmentVertical(Vector3 position, float length, string name)
        {
            GameObject wall = CreateWallPrimitive(
                new Vector3(wallThickness, wallHeight, length));
            wall.transform.position = position;
            wall.name = $"Wall_{name}";
            wall.transform.parent = transform;
            spawnedObjects.Add(wall);
        }

        private GameObject CreateWallPrimitive(Vector3 size)
        {
            if (wallPrefab != null)
            {
                GameObject wall = Instantiate(wallPrefab);
                wall.transform.localScale = size;
                return wall;
            }

            GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.transform.localScale = size;

            // Semi-transparent so they're visible but not distracting
            Renderer rend = cube.GetComponent<Renderer>();
            if (rend != null)
            {
                Material mat = new Material(Shader.Find("Standard"));
                mat.color = new Color(0.4f, 0.4f, 0.45f, 0.35f);
                SetMaterialTransparent(mat);
                rend.material = mat;
            }

            // Make it a static NavMesh obstacle
            cube.isStatic = true;
            return cube;
        }

        private static void SetMaterialTransparent(Material mat)
        {
            mat.SetFloat("_Mode", 3);
            mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetInt("_ZWrite", 0);
            mat.DisableKeyword("_ALPHATEST_ON");
            mat.EnableKeyword("_ALPHABLEND_ON");
            mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            mat.renderQueue = 3000;
        }

        // ─────────────────────────────────────────────────────────
        //  Floor Direction Arrows
        // ─────────────────────────────────────────────────────────

        /// <summary>
        /// Spawns flat quads on the floor surface showing the one-way
        /// flow direction in each aisle.
        /// </summary>
        private void BuildFloorArrows(Vector3 floorCentre)
        {
            float y = 0.02f;
            float arrowSize = 1.2f;

            // Row aisle arrows
            int numRowAisles = layoutRows - 1;
            for (int a = 0; a < numRowAisles; a++)
            {
                Vector3 aisleCentre = GetRowAisleCentre(a);
                Vector3 dir = GetRowAisleDirection(a);
                float yaw = (dir.x > 0) ? 90f : -90f;

                // Place arrows at regular intervals along the aisle
                float halfWidth = ((layoutCols - 1) * machineSpacingX) / 2f;
                for (float x = -halfWidth; x <= halfWidth; x += machineSpacingX)
                {
                    Vector3 pos = new Vector3(
                        floorCentre.x + x, y, aisleCentre.z);
                    SpawnFloorArrow(pos, yaw, arrowSize,
                        new Color(0.9f, 0.7f, 0.2f, 0.3f),
                        $"Arrow_RowAisle{a}");
                }
            }

            // Top spine arrows (eastbound)
            float topZ = floorCentre.z + GetTopSpineZ();
            float machineAreaHalfW = ((layoutCols - 1) * machineSpacingX) / 2f;
            for (float x = -machineAreaHalfW; x <= machineAreaHalfW; x += machineSpacingX)
            {
                SpawnFloorArrow(
                    new Vector3(floorCentre.x + x, y, topZ),
                    90f, arrowSize * 1.2f,
                    new Color(0.1f, 0.7f, 0.5f, 0.3f),
                    "Arrow_TopSpine");
            }

            // Bottom spine arrows (westbound)
            float botZ = floorCentre.z + GetBottomSpineZ();
            for (float x = machineAreaHalfW; x >= -machineAreaHalfW; x -= machineSpacingX)
            {
                SpawnFloorArrow(
                    new Vector3(floorCentre.x + x, y, botZ),
                    -90f, arrowSize * 1.2f,
                    new Color(0.1f, 0.7f, 0.5f, 0.3f),
                    "Arrow_BotSpine");
            }

            // Left vertical arrows (northbound)
            float leftX = floorCentre.x
                - machineAreaHalfW - machineDepth / 2f - verticalAisleWidth / 2f;
            for (int a = 0; a < numRowAisles; a++)
            {
                Vector3 ac = GetRowAisleCentre(a);
                SpawnFloorArrow(
                    new Vector3(leftX, y, ac.z),
                    0f, arrowSize, // Changed to 0f (North)
                    new Color(0.2f, 0.4f, 0.9f, 0.3f),
                    "Arrow_LeftVert");
            }

            // Right vertical arrows (southbound)
            float rightX = floorCentre.x
                + machineAreaHalfW + machineDepth / 2f + verticalAisleWidth / 2f;
            for (int a = 0; a < numRowAisles; a++)
            {
                Vector3 ac = GetRowAisleCentre(a);
                SpawnFloorArrow(
                    new Vector3(rightX, y, ac.z),
                    180f, arrowSize, // Changed to 180f (South)
                    new Color(0.2f, 0.4f, 0.9f, 0.3f),
                    "Arrow_RightVert");
            }
        }

        /// <summary>
        /// Creates a flat arrow-shaped quad on the floor plane.
        /// Yaw 0 = north (+Z), 90 = east (+X), 180 = south, -90 = west.
        /// </summary>
        private void SpawnFloorArrow(Vector3 position, float yawDegrees,
                                     float size, Color color, string name)
        {
            GameObject arrow = new GameObject(name);
            arrow.transform.position = position;
            arrow.transform.rotation = Quaternion.Euler(0f, yawDegrees, 0f);
            arrow.transform.parent = transform;

            MeshFilter mf = arrow.AddComponent<MeshFilter>();
            MeshRenderer mr = arrow.AddComponent<MeshRenderer>();

            // Simple arrow mesh (flat triangle + tail)
            Mesh mesh = new Mesh();
            float s = size / 2f;
            mesh.vertices = new[]
            {
                new Vector3(0f, 0f, s),         // tip
                new Vector3(-s * 0.5f, 0f, 0f), // left wing
                new Vector3(s * 0.5f, 0f, 0f),  // right wing
                new Vector3(-s * 0.2f, 0f, 0f), // tail left
                new Vector3(s * 0.2f, 0f, 0f),  // tail right
                new Vector3(-s * 0.2f, 0f, -s), // tail bottom left
                new Vector3(s * 0.2f, 0f, -s),  // tail bottom right
            };
            mesh.triangles = new[]
            {
                0, 2, 1,    // arrowhead
                3, 4, 5,    // tail top
                4, 6, 5,    // tail bottom
            };
            mesh.RecalculateNormals();
            mf.mesh = mesh;

            Material mat = arrowMaterial != null
                ? new Material(arrowMaterial)
                : new Material(Shader.Find("Standard"));
            mat.color = color;
            SetMaterialTransparent(mat);
            mr.material = mat;

            spawnedObjects.Add(arrow);
        }

        // ─────────────────────────────────────────────────────────
        //  Distance Matrix
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
            SimLogger.Medium($"[FactoryLayout] Distance matrix ({n}×{n}):\n{header}");

            for (int i = 0; i < n; i++)
            {
                string row = $"M{i,-3} ";
                for (int j = 0; j < n; j++)
                    row += $"{distanceMatrix[i, j],6:F1} ";
                SimLogger.High(row);
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Editor Gizmos
        // ─────────────────────────────────────────────────────────

        private void OnDrawGizmosSelected()
        {
            if (floorTransform == null) return;
            Vector3 c = floorTransform.position;

            // Approximate floor bounds
            int previewCols = 5;
            int previewRows = 4;
            float areaW = (previewCols - 1) * machineSpacingX + machineDepth;
            float areaD = (previewRows - 1) * RowPitch + machineDepth;
            float totalW = verticalAisleWidth * 2 + areaW;
            float totalD = spineAisleWidth * 2 + areaD;

            // Floor outline
            Gizmos.color = new Color(1f, 1f, 1f, 0.15f);
            Gizmos.DrawWireCube(c, new Vector3(totalW, 0f, totalD));

            // Machine preview
            Gizmos.color = new Color(0f, 0.8f, 0.5f, 0.6f);
            for (int r = 0; r < previewRows; r++)
            {
                for (int co = 0; co < previewCols; co++)
                {
                    float x = -((previewCols - 1) * machineSpacingX) / 2f
                              + co * machineSpacingX;
                    float z = ((previewRows - 1) * RowPitch) / 2f
                              - r * RowPitch;
                    Gizmos.DrawWireCube(
                        c + new Vector3(x, machineYOffset, z),
                        new Vector3(machineDepth, 1f, machineDepth));
                }
            }

            // Row aisle previews
            Gizmos.color = new Color(0.9f, 0.7f, 0.2f, 0.2f);
            for (int a = 0; a < previewRows - 1; a++)
            {
                float z = ((previewRows - 1) * RowPitch) / 2f
                          - a * RowPitch - RowPitch / 2f;
                Gizmos.DrawCube(
                    c + new Vector3(0f, 0.01f, z),
                    new Vector3(areaW, 0f, rowAisleWidth));
            }

            // Spine aisle previews
            Gizmos.color = new Color(0.1f, 0.7f, 0.5f, 0.15f);
            float topZ = (areaD / 2f + spineAisleWidth / 2f);
            float botZ = -(areaD / 2f + spineAisleWidth / 2f);
            Gizmos.DrawCube(c + new Vector3(0f, 0.01f, topZ),
                new Vector3(totalW, 0f, spineAisleWidth));
            Gizmos.DrawCube(c + new Vector3(0f, 0.01f, botZ),
                new Vector3(totalW, 0f, spineAisleWidth));
        }
    }
}