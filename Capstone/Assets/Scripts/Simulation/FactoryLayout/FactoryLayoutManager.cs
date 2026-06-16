using System;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;
using Assets.Scripts.Simulation.Logging;
using Unity.AI.Navigation;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation.FactoryLayout
{
    public enum ParkingMethod { Single, Multiple }

    public struct ParkingArea
    {
        public Vector3 Position;
        public int RowAisleIndex;   // -1 for the single south alcove
        public bool IsLeftSide;     // meaningful only in Multiple mode
    }
    /// @brief Builds an aisle-based factory floor with one-way traffic lanes, physical wall colliders, and zone markers.
    public class FactoryLayoutManager : MonoBehaviour
    {
        public static FactoryLayoutManager Instance;
        [SerializeField] private NavMeshSurface navMeshSurface;

        [Header("Prefabs")]
        [SerializeField] private PhysicalMachine machinePrefab;
        [SerializeField] private PhysicalMachine doubleSidedMachinePrefab;

        [Header("I/O & AGV Infrastructure")]
        [SerializeField] private GameObject conveyorPrefab;
        [SerializeField] private Vector3 ioConveyorScale = new Vector3(.05f, .2f, .30f);
        [SerializeField] private Material incomingBeltMaterial;
        [SerializeField] private Vector3 incomingBeltOffset = new Vector3(-2f, 0.01f, 1.5f);
        [SerializeField] private Vector3 outgoingBeltOffset = new Vector3(-2f, 0.01f, 1.5f);

        /// Probability [0,1] that a machine gains each non-primary type as a
        /// secondary capability. 0 = single-type, backward compatible.
        /// 1 = full flexibility (every machine handles every operation type).
        [Range(0f, 1f)]
        public float MachineFlexibilityProbability = 0f;

        public Vector3 IncomingBeltPosition { get; private set; }
        public Vector3 OutgoingBeltPosition { get; private set; }
        public Vector3 AGVParkingPosition { get; private set; }

        public ConveyorBelt IncomingBelt { get; private set; }
        public ConveyorBelt OutgoingBelt { get; private set; }
        public ParkingMethod ActiveParkingMethod { get; private set; }
        private readonly List<ParkingArea> parkingAreas = new List<ParkingArea>();
        public IReadOnlyList<ParkingArea> ParkingAreas => parkingAreas;

        [SerializeField] private GameObject wallPrefab;

        [Header("Floor")]
        [SerializeField] private Transform floorTransform;

        [Header("Machine Grid")]
        [SerializeField] private float machineSpacingX = 6f;
        [SerializeField] private float machineDepth = 1.5f;
        [SerializeField] private float conveyorReach = 1.5f;

        [Header("Aisles")]
        [SerializeField] private float rowAisleWidth = 3f;
        [SerializeField] private float spineAisleWidth = 4f;
        [SerializeField] private float verticalAisleWidth = 3.5f;
        [SerializeField] private float parkingAlcoveDepth = 5f;

        [Header("Walls")]
        [SerializeField] private float wallHeight = 0.6f;
        [SerializeField] private float wallThickness = 0.15f;

        [Header("Visual")]
        [SerializeField] private Material arrowMaterial;
        [SerializeField] private float machineYOffset = 0.5f;

        [Header("Debug")]
        [SerializeField] private bool logDistanceMatrix = true;

        private PhysicalMachine[] machines;
        private float[,] distanceMatrix;
        private float[] distanceMatrixFlat;
        private Vector3[] customPositions;
        private readonly List<GameObject> spawnedObjects = new List<GameObject>();

        private int layoutRows;
        private int layoutCols;
        private float totalFloorWidth;
        private float totalFloorDepth;

        public IReadOnlyList<PhysicalMachine> Machines => machines;
        public int MachineCount => machines?.Length ?? 0;
        public float[,] DistanceMatrix => distanceMatrix;
        public float[] DistanceMatrixFlat => distanceMatrixFlat;
        public int LayoutRows => layoutRows;
        public int LayoutCols => layoutCols;
        public float RowPitch => machineDepth + conveyorReach * 2f + rowAisleWidth;
        public float MachineSpacingX => machineSpacingX;
        public float SpineAisleWidth => spineAisleWidth;
        public float VerticalAisleWidth => verticalAisleWidth;
        public float RowAisleWidth => rowAisleWidth;
        public float ConveyorReach => conveyorReach;
        public float MachineDepth => machineDepth;
        public Vector3 GridOrigin { get; private set; }
        public Vector2 FloorSize => new Vector2(totalFloorWidth, totalFloorDepth);

        public float ParkingAlcoveDepth => parkingAlcoveDepth;

        void Awake()
        {
            Instance = this;
        }

        private ParkingMethod ParseParkingMethod(string method)
        {
            return method.ToLower() switch
            {
                "single" => ParkingMethod.Single,
                "multiple" => ParkingMethod.Multiple,
                _ => throw new ArgumentException($"Invalid parking method: {method}")
            };
        }

        /// @brief Generates the factory floor, places machines, and builds navigation boundaries.
        /// 
        /// @details Calculates the grid dimensions based on machine count, instantiates the 
        /// appropriate machine prefabs (single or double-sided), builds aisle walls for 
        /// NavMesh carving, and computes the spatial distance matrix.
        /// 
        /// @param simulator The active simulation instance containing machine data.
        /// 
        /// @pre @p simulator must not be null.
        /// @post The floor is populated with machines, walls, and arrows; the NavMesh is rebuilt.
        public Dictionary<MachineType, List<int>> BuildFloor(FJSSPConfig config)
        {
            // if (simulator == null)
            //     throw new ArgumentNullException(nameof(simulator));

            ClearFloor();

            int machineCount = config.MachineTypeLayout.Length;
            var machinesByType = new Dictionary<MachineType, List<int>>();

            layoutCols = Mathf.CeilToInt(Mathf.Sqrt(machineCount));
            layoutRows = Mathf.CeilToInt((float)machineCount / layoutCols);

            float machineAreaWidth = (layoutCols - 1) * machineSpacingX + machineDepth;
            float machineAreaDepth = (layoutRows - 1) * RowPitch + machineDepth;

            ActiveParkingMethod = ParseParkingMethod(config.parkingMethod);

            totalFloorWidth = verticalAisleWidth + machineAreaWidth + verticalAisleWidth;
            totalFloorDepth = spineAisleWidth + machineAreaDepth + spineAisleWidth;

            if (ActiveParkingMethod == ParkingMethod.Single)
            {
                totalFloorDepth += parkingAlcoveDepth;                 // south alcove band

                int agvCount = config.AGVCount;
                float minParkingWidth = (agvCount - 1) * 2f + 4f;      // fit AGVs across the south pool
                if (minParkingWidth > totalFloorWidth)
                    totalFloorWidth = minParkingWidth;
            }
            else // Multiple
            {
                totalFloorWidth += parkingAlcoveDepth * 2f;            // side alcoves, both edges
            }

            if (floorTransform != null)
            {
                floorTransform.localScale = new Vector3(
                    totalFloorWidth / 10f, 1f, totalFloorDepth / 10f);
            }

            Vector3 floorCentre = floorTransform != null ? floorTransform.position : Vector3.zero;

            GridOrigin = floorCentre + new Vector3(
                -machineAreaWidth / 2f,
                0f,
                machineAreaDepth / 2f);

            // Distribute machine types so no two machines of the same type share the same
            // physical row aisle.  Same-type clustering would force all AGVs heading to that
            // type to compete for the same aisle; spreading them across rows lets the
            // _SRWT composite PDR route jobs to a less-congested aisle instead.
            MachineType[] distributedLayout = BuildDistributedTypeLayout(
                config.MachineTypeLayout, layoutRows, layoutCols);
            MachineType[] allTypes = (MachineType[])Enum.GetValues(typeof(MachineType));
            LogDistributedLayout(distributedLayout, layoutRows, layoutCols);

            machines = new PhysicalMachine[machineCount];
            for (int i = 0; i < machineCount; i++)
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
                    prefabToSpawn = machinePrefab;
                    rotation = Quaternion.Euler(0f, 180f, 0f);
                }
                else if (row == layoutRows - 1)
                {
                    prefabToSpawn = machinePrefab;
                    rotation = Quaternion.identity;
                }
                else
                {
                    prefabToSpawn = doubleSidedMachinePrefab != null ? doubleSidedMachinePrefab : machinePrefab;
                    rotation = Quaternion.identity;
                }
                MachineType primary = distributedLayout[i];
                HashSet<MachineType> caps = SampleCapabilities(primary, config, allTypes);

                PhysicalMachine pm = Instantiate(prefabToSpawn, worldPos, rotation, transform);
                pm.gameObject.name = $"Machine_{i}_{primary}";
                pm.Initialize(i, primary, caps);
                machines[i] = pm;

                // Register under every capability — FJSSPJobGenerator queries machinesByType[opType]
                // so a Mill+Lathe machine must appear in both buckets.
                foreach (MachineType cap in caps)
                {
                    if (!machinesByType.ContainsKey(cap))
                        machinesByType[cap] = new List<int>();
                    machinesByType[cap].Add(i);
                }
            }

            BuildAisleWalls(floorCentre);
            BuildFloorArrows(floorCentre);
            ComputeDistanceMatrix();
            if (logDistanceMatrix) LogDistanceMatrix();

            BuildInfrastructure(floorCentre);

            navMeshSurface.BuildNavMesh();

            SimLogger.Medium($"[FactoryLayout] Built aisle-based floor: {machineCount} machines.");
            return machinesByType;
        }
        /// @brief Samples a set of MachineType capabilities for one machine.
        ///
        /// @details The primary type is always included. Each remaining type is
        /// added independently with probability config.MachineFlexibilityProbability.
        /// Uses UnityEngine.Random, which is seeded in SpawnFactory before BuildFloor
        /// is called, so layouts are fully reproducible from config.Seed.
        private static HashSet<MachineType> SampleCapabilities(
            MachineType primary, FJSSPConfig config, MachineType[] allTypes)
        {
            var caps = new HashSet<MachineType> { primary };
            float p = config.MachineFlexibilityProbability;
            if (p <= 0f) return caps;   // fast path — backward compatible default

            foreach (MachineType t in allTypes)
            {
                if (t == primary) continue;
                if (UnityEngine.Random.value < p)
                    caps.Add(t);
            }
            return caps;
        }

        /// @brief Instantiates I/O conveyors and the AGV parking zone.
        /// 
        /// @details Places the incoming belt at the top-left spine and the outgoing belt 
        /// at the bottom-right. Sets the AGV parking anchor at the bottom-left.
        /// 
        /// @param floorCentre The central world position of the factory floor.
        private void BuildInfrastructure(Vector3 floorCentre)
        {
            float machineAreaHalfW = ((layoutCols - 1) * machineSpacingX) / 2f;

            float topZ = floorCentre.z + GetTopSpineZ();
            IncomingBeltPosition = new Vector3(
                floorCentre.x - machineAreaHalfW + incomingBeltOffset.x,
                incomingBeltOffset.y,
                topZ + incomingBeltOffset.z);

            if (conveyorPrefab != null)
            {
                GameObject inBelt = Instantiate(conveyorPrefab, IncomingBeltPosition, Quaternion.Euler(0, 0, 0), transform);
                inBelt.name = "Incoming_Belt";
                inBelt.transform.localScale = ioConveyorScale;
                IncomingBelt = inBelt.GetComponent<ConveyorBelt>();
                IncomingBelt.Capacity = 8;
                if (incomingBeltMaterial != null)
                {
                    foreach (var rend in inBelt.GetComponentsInChildren<Renderer>())
                        rend.material = incomingBeltMaterial;
                }
                spawnedObjects.Add(inBelt);
            }

            float botZ = floorCentre.z + GetBottomSpineZ();
            OutgoingBeltPosition = new Vector3(
                floorCentre.x + machineAreaHalfW + verticalAisleWidth + outgoingBeltOffset.x,
                outgoingBeltOffset.y,
                botZ + outgoingBeltOffset.z);

            if (conveyorPrefab != null)
            {
                GameObject outBelt = Instantiate(conveyorPrefab, OutgoingBeltPosition, Quaternion.Euler(0, 180, 0), transform);
                outBelt.name = "Outgoing_Belt";
                outBelt.transform.localScale = ioConveyorScale;
                spawnedObjects.Add(outBelt);
                OutgoingBelt = outBelt.GetComponent<ConveyorBelt>();
            }

            parkingAreas.Clear();

            int numRowAisles = layoutRows - 1;
            if (ActiveParkingMethod == ParkingMethod.Multiple && numRowAisles > 0)
            {
                BuildMultipleParkingAreas(floorCentre);
                AGVParkingPosition = parkingAreas[0].Position;   // back-compat default
            }
            else
            {
                // Single (or degenerate single-row layout): one south alcove.
                float alcoveZ = botZ - (spineAisleWidth / 2f) - (parkingAlcoveDepth / 2f);
                AGVParkingPosition = new Vector3(floorCentre.x, 0.01f, alcoveZ);
                parkingAreas.Add(new ParkingArea
                {
                    Position = AGVParkingPosition,
                    RowAisleIndex = -1,
                    IsLeftSide = false
                });
            }
        }
        /// @brief Places one parking alcove per row aisle, on that aisle's exit side
        ///        (right for eastbound, left for westbound), just beyond the outer wall.
        private void BuildMultipleParkingAreas(Vector3 floorCentre)
        {
            int numRowAisles = layoutRows - 1;
            float machineAreaHalfW = ((layoutCols - 1) * machineSpacingX + machineDepth) / 2f;
            float outerEdgeX = machineAreaHalfW + verticalAisleWidth;          // current outer wall x
            float alcoveOffsetX = outerEdgeX + parkingAlcoveDepth / 2f;        // alcove centre, beyond wall

            for (int a = 0; a < numRowAisles; a++)
            {
                bool eastbound = (a % 2 == 0);
                bool leftSide = !eastbound;                                    // eastbound exits right
                float x = floorCentre.x + (leftSide ? -alcoveOffsetX : alcoveOffsetX);
                float z = GetRowAisleCentre(a).z;                             // world z (already offset)

                parkingAreas.Add(new ParkingArea
                {
                    Position = new Vector3(x, 0.01f, z),
                    RowAisleIndex = a,
                    IsLeftSide = leftSide
                });
            }
        }

        /// @brief Destroys all spawned factory components and clears memory.
        /// 
        /// @post The floor is empty, machine arrays are null, and NavMesh data is removed.
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

        public void SetCustomLayout(Vector3[] positions) => customPositions = positions;

        /// @brief Retrieves a machine instance by its ID.
        /// @param machineId The index of the machine.
        /// @return The PhysicalMachine component or null if out of bounds.
        public PhysicalMachine GetMachine(int machineId)
        {
            if (machines == null || machineId < 0 || machineId >= machines.Length)
                return null;
            return machines[machineId];
        }

        /// @brief Calculates the local XZ coordinates for a machine at a specific grid coordinate.
        /// @param row The machine row.
        /// @param col The machine column.
        /// @return A Vector3 local position relative to the floor center.
        public Vector3 GetMachineLocalPosition(int row, int col)
        {
            float machineAreaWidth = (layoutCols - 1) * machineSpacingX;
            float machineAreaDepth = (layoutRows - 1) * RowPitch;

            float x = -machineAreaWidth / 2f + col * machineSpacingX;
            float z = machineAreaDepth / 2f - row * RowPitch;

            return new Vector3(x, 0f, z);
        }

        /// @brief Returns the world position of the center of a row aisle.
        /// @param aisleIndex The index of the aisle (0 is between machine row 0 and 1).
        public Vector3 GetRowAisleCentre(int aisleIndex)
        {
            Vector3 floorCentre = floorTransform != null ? floorTransform.position : Vector3.zero;
            float machineAreaDepth = (layoutRows - 1) * RowPitch;

            float zTop = machineAreaDepth / 2f - aisleIndex * RowPitch;
            float zBot = zTop - RowPitch;
            float aisleZ = (zTop + zBot) / 2f;

            return floorCentre + new Vector3(0f, 0.01f, aisleZ);
        }

        /// @brief Returns the restricted flow direction for a specific row aisle.
        public Vector3 GetRowAisleDirection(int aisleIndex)
        {
            return (aisleIndex % 2 == 0) ? Vector3.right : Vector3.left;
        }

        /// @brief Returns the Z offset for the top peripheral spine.
        public float GetTopSpineZ()
        {
            float machineAreaDepth = (layoutRows - 1) * RowPitch + machineDepth;
            return machineAreaDepth / 2f + spineAisleWidth / 2f;
        }

        /// @brief Returns the Z offset for the bottom peripheral spine.
        public float GetBottomSpineZ()
        {
            float machineAreaDepth = (layoutRows - 1) * RowPitch + machineDepth;
            return -(machineAreaDepth / 2f + spineAisleWidth / 2f);
        }

        /// @brief Generates physical aisle wall segments.
        /// 
        /// @details Places collider segments along machine row edges and factory boundaries. 
        /// These walls force the NavMesh to only allow navigation through designated aisles.
        /// 
        /// @param floorCentre The central world position of the factory floor.
        private void BuildAisleWalls(Vector3 floorCentre)
        {
            float machineAreaWidth = (layoutCols - 1) * machineSpacingX + machineDepth;
            float halfMachineDepth = machineDepth / 2f;

            for (int row = 0; row < layoutRows; row++)
            {
                Vector3 rowCentre = floorCentre + GetMachineLocalPosition(row, 0);
                rowCentre.x = floorCentre.x;
                rowCentre.y = wallHeight / 2f;

                SpawnWallSegment(rowCentre + Vector3.forward * (halfMachineDepth + 0.05f), machineAreaWidth, row, "North");
                SpawnWallSegment(rowCentre - Vector3.forward * (halfMachineDepth + 0.05f), machineAreaWidth, row, "South");
            }

            float machineAreaDepth = (layoutRows - 1) * RowPitch + machineDepth;
            float fullHeight = machineAreaDepth + spineAisleWidth * 2f;

            SpawnWallSegmentVertical(floorCentre + new Vector3(-(machineAreaWidth / 2f + verticalAisleWidth), wallHeight / 2f, 0f), fullHeight, "LeftOuter");
            SpawnWallSegmentVertical(floorCentre + new Vector3(machineAreaWidth / 2f + verticalAisleWidth, wallHeight / 2f, 0f), fullHeight, "RightOuter");
        }

        /// @brief Spawns wall segments for machine rows with gaps for conveyor access.
        /// @param position The center position of the wall line.
        /// @param length The total length of the machine row.
        /// @param row The row index.
        /// @param side Label for naming (North/South).
        private void SpawnWallSegment(Vector3 position, float length, int row, string side)
        {
            float gapWidth = 1.2f;
            float segStart = -length / 2f;

            for (int col = 0; col <= layoutCols; col++)
            {
                float colX = 0;
                float segEnd;

                if (col < layoutCols)
                {
                    colX = -((layoutCols - 1) * machineSpacingX) / 2f + col * machineSpacingX;
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

                    GameObject wall = CreateWallPrimitive(new Vector3(segLength, wallHeight, wallThickness));
                    wall.transform.position = segPos;
                    wall.name = $"Wall_Row{row}_{side}_Seg{col}";
                    wall.transform.parent = transform;
                    spawnedObjects.Add(wall);
                }

                if (col < layoutCols) segStart = colX + gapWidth / 2f;
            }
        }

        /// @brief Spawns a single vertical wall segment.
        private void SpawnWallSegmentVertical(Vector3 position, float length, string name)
        {
            GameObject wall = CreateWallPrimitive(new Vector3(wallThickness, wallHeight, length));
            wall.transform.position = position;
            wall.name = $"Wall_{name}";
            wall.transform.parent = transform;
            spawnedObjects.Add(wall);
        }

        /// @brief Creates the primitive mesh and material for a wall.
        /// @param size Dimensions of the wall cube.
        /// @return The instantiated GameObject.
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
            Renderer rend = cube.GetComponent<Renderer>();
            if (rend != null)
            {
                Material mat = new Material(Shader.Find("Standard"));
                mat.color = new Color(0.4f, 0.4f, 0.45f, 0.35f);
                SetMaterialTransparent(mat);
                rend.material = mat;
            }
            cube.isStatic = true;
            return cube;
        }

        /// @brief Configures a material to use transparent alpha blending.
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

        /// @brief Places directional arrow markers on the floor to visualize traffic flow.
        /// @param floorCentre The central world position of the factory floor.
        private void BuildFloorArrows(Vector3 floorCentre)
        {
            float y = 0.02f;
            float arrowSize = 1.2f;

            int numRowAisles = layoutRows - 1;
            for (int a = 0; a < numRowAisles; a++)
            {
                Vector3 aisleCentre = GetRowAisleCentre(a);
                Vector3 dir = GetRowAisleDirection(a);
                float yaw = (dir.x > 0) ? 90f : -90f;

                float halfWidth = ((layoutCols - 1) * machineSpacingX) / 2f;
                for (float x = -halfWidth; x <= halfWidth; x += machineSpacingX)
                {
                    SpawnFloorArrow(new Vector3(floorCentre.x + x, y, aisleCentre.z), yaw, arrowSize, new Color(0.9f, 0.7f, 0.2f, 0.3f), $"Arrow_RowAisle{a}");
                }
            }

            float topZ = floorCentre.z + GetTopSpineZ();
            float machineAreaHalfW = ((layoutCols - 1) * machineSpacingX) / 2f;
            for (float x = -machineAreaHalfW; x <= machineAreaHalfW; x += machineSpacingX)
                SpawnFloorArrow(new Vector3(floorCentre.x + x, y, topZ), 90f, arrowSize * 1.2f, new Color(0.1f, 0.7f, 0.5f, 0.3f), "Arrow_TopSpine");

            float botZ = floorCentre.z + GetBottomSpineZ();
            for (float x = machineAreaHalfW; x >= -machineAreaHalfW; x -= machineSpacingX)
                SpawnFloorArrow(new Vector3(floorCentre.x + x, y, botZ), -90f, arrowSize * 1.2f, new Color(0.1f, 0.7f, 0.5f, 0.3f), "Arrow_BotSpine");

            float leftX = floorCentre.x - machineAreaHalfW - machineDepth / 2f - verticalAisleWidth / 2f;
            for (int a = 0; a < numRowAisles; a++)
                SpawnFloorArrow(new Vector3(leftX, y, GetRowAisleCentre(a).z), 0f, arrowSize, new Color(0.2f, 0.4f, 0.9f, 0.3f), "Arrow_LeftVert");

            float rightX = floorCentre.x + machineAreaHalfW + machineDepth / 2f + verticalAisleWidth / 2f;
            for (int a = 0; a < numRowAisles; a++)
                SpawnFloorArrow(new Vector3(rightX, y, GetRowAisleCentre(a).z), 180f, arrowSize, new Color(0.2f, 0.4f, 0.9f, 0.3f), "Arrow_RightVert");
        }

        /// @brief Generates a custom mesh for a flat arrow on the floor plane.
        /// @param position World position.
        /// @param yawDegrees Rotation on the Y axis.
        /// @param size Scale of the arrow.
        /// @param color Tint of the arrow material.
        /// @param name Object name.
        private void SpawnFloorArrow(Vector3 position, float yawDegrees, float size, Color color, string name)
        {
            GameObject arrow = new GameObject(name);
            arrow.transform.position = position;
            arrow.transform.rotation = Quaternion.Euler(0f, yawDegrees, 0f);
            arrow.transform.parent = transform;

            MeshFilter mf = arrow.AddComponent<MeshFilter>();
            MeshRenderer mr = arrow.AddComponent<MeshRenderer>();

            Mesh mesh = new Mesh();
            float s = size / 2f;
            mesh.vertices = new[]
            {
                new Vector3(0f, 0f, s),
                new Vector3(-s * 0.5f, 0f, 0f),
                new Vector3(s * 0.5f, 0f, 0f),
                new Vector3(-s * 0.2f, 0f, 0f),
                new Vector3(s * 0.2f, 0f, 0f),
                new Vector3(-s * 0.2f, 0f, -s),
                new Vector3(s * 0.2f, 0f, -s),
            };
            mesh.triangles = new[] { 0, 2, 1, 3, 4, 5, 4, 6, 5 };
            mesh.RecalculateNormals();
            mf.mesh = mesh;

            Material mat = arrowMaterial != null ? new Material(arrowMaterial) : new Material(Shader.Find("Standard"));
            mat.color = color;
            SetMaterialTransparent(mat);
            mr.material = mat;
            spawnedObjects.Add(arrow);
        }

        /// @brief Calculates the Euclidean distance between all machines in the factory.
        /// 
        /// @details Populates a 2D matrix for internal logic and a flattened, 
        /// normalized array for the simulation's observation space.
        private void ComputeDistanceMatrix()
        {
            int n = machines.Length;
            distanceMatrix = new float[n, n];

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    float dist = Vector3.Distance(machines[i].transform.position, machines[j].transform.position);
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
            foreach (float d in distanceMatrixFlat) if (d > maxDist) maxDist = d;
            if (maxDist > 0f) for (int k = 0; k < distanceMatrixFlat.Length; k++) distanceMatrixFlat[k] /= maxDist;
        }

        private void LogDistanceMatrix()
        {
            if (distanceMatrix == null) return;
            int n = machines.Length;
            string header = "     ";
            for (int j = 0; j < n; j++) header += $"  M{j,-4}";
            SimLogger.Medium($"[FactoryLayout] Distance matrix ({n}x{n}):\n{header}");

            for (int i = 0; i < n; i++)
            {
                string row = $"M{i,-3} ";
                for (int j = 0; j < n; j++) row += $"{distanceMatrix[i, j],6:F1} ";
                SimLogger.High(row);
            }
        }

        /// @brief Reorders a flat MachineType array so that, within each grid row,
        ///        every machine type appears at most once — spreading same-type machines
        ///        across different physical row aisles.
        ///
        /// @details Uses a greedy "largest-remaining-count first" heuristic per row.
        ///          Within a single row it picks the type with the highest outstanding
        ///          quota that has not already been placed in that row.  If all distinct
        ///          types are exhausted before the row is full (i.e. one type has more
        ///          machines than there are rows), the constraint is relaxed and the
        ///          type with the highest remaining count fills the overflow slots.
        ///
        /// @param original The flat type array from FJSSPConfig — not mutated.
        /// @param rows     Number of grid rows already calculated by BuildFloor.
        /// @param cols     Number of grid columns already calculated by BuildFloor.
        /// @return A new array of the same length with redistributed type assignments.
        private static MachineType[] BuildDistributedTypeLayout(MachineType[] original, int rows, int cols)
        {
            // Tally how many of each type we need to place in total.
            var remaining = new Dictionary<MachineType, int>();
            foreach (MachineType t in original)
            {
                if (!remaining.ContainsKey(t)) remaining[t] = 0;
                remaining[t]++;
            }

            var result = new MachineType[original.Length];
            int total = original.Length;

            for (int row = 0; row < rows; row++)
            {
                var usedThisRow = new HashSet<MachineType>();

                for (int col = 0; col < cols; col++)
                {
                    int idx = row * cols + col;
                    if (idx >= total) break;

                    // Pass 1 – pick the type with the highest quota not yet used in this row.
                    MachineType best = default;
                    int bestCount = -1;

                    foreach (var kvp in remaining)
                    {
                        if (kvp.Value <= 0 || usedThisRow.Contains(kvp.Key)) continue;
                        if (kvp.Value > bestCount) { best = kvp.Key; bestCount = kvp.Value; }
                    }

                    // Pass 2 (fallback) – every distinct type already appears in this row;
                    // relax the uniqueness constraint and just fill with the highest-quota type.
                    if (bestCount < 0)
                    {
                        foreach (var kvp in remaining)
                        {
                            if (kvp.Value <= 0) continue;
                            if (kvp.Value > bestCount) { best = kvp.Key; bestCount = kvp.Value; }
                        }
                    }

                    result[idx] = best;
                    remaining[best]--;
                    usedThisRow.Add(best);
                }
            }

            return result;
        }

        /// @brief Logs the distributed machine-type grid so you can visually verify
        ///        that no row contains duplicate types (unless overflow forces it).
        private void LogDistributedLayout(MachineType[] layout, int rows, int cols)
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("[FactoryLayout] Distributed type layout (row x col):");

            for (int row = 0; row < rows; row++)
            {
                sb.Append($"  Row{row}: ");
                var typesInRow = new List<string>();
                for (int col = 0; col < cols; col++)
                {
                    int idx = row * cols + col;
                    if (idx < layout.Length)
                        typesInRow.Add(layout[idx].ToString());
                }
                sb.AppendLine(string.Join(", ", typesInRow));
            }

            // Verify the invariant: warn if any row has two same-type machines.
            bool violation = false;
            for (int row = 0; row < rows; row++)
            {
                var seen = new HashSet<MachineType>();
                for (int col = 0; col < cols; col++)
                {
                    int idx = row * cols + col;
                    if (idx >= layout.Length) break;
                    if (!seen.Add(layout[idx]))
                    {
                        sb.AppendLine($"  [WARN] Row {row} contains duplicate type '{layout[idx]}'" +
                                      " — this type has more instances than there are rows.");
                        violation = true;
                    }
                }
            }

            if (!violation)
                sb.AppendLine("  [OK] No two same-type machines share a row aisle.");

            SimLogger.Medium(sb.ToString());
        }

        private void OnDrawGizmosSelected()
        {
            if (floorTransform == null) return;
            Vector3 c = floorTransform.position;
            int previewCols = 5; int previewRows = 4;
            float areaW = (previewCols - 1) * machineSpacingX + machineDepth;
            float areaD = (previewRows - 1) * RowPitch + machineDepth;
            float totalW = verticalAisleWidth * 2 + areaW;
            float totalD = spineAisleWidth * 2 + areaD;

            Gizmos.color = new Color(1f, 1f, 1f, 0.15f);
            Gizmos.DrawWireCube(c, new Vector3(totalW, 0f, totalD));

            Gizmos.color = new Color(0f, 0.8f, 0.5f, 0.6f);
            for (int r = 0; r < previewRows; r++)
                for (int co = 0; co < previewCols; co++)
                    Gizmos.DrawWireCube(c + GetMachineLocalPosition(r, co) + Vector3.up * machineYOffset, new Vector3(machineDepth, 1f, machineDepth));
        }
    }
}