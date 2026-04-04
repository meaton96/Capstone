using System;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Scheduling.Core;
using Assets.Scripts.Logging;
using Unity.AI.Navigation;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation.FactoryLayout
{
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

        public Vector3 IncomingBeltPosition { get; private set; }
        public Vector3 OutgoingBeltPosition { get; private set; }
        public Vector3 AGVParkingPosition { get; private set; }

        public ConveyorBelt IncomingBelt { get; private set; }
        public ConveyorBelt OutgoingBelt { get; private set; }

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

        void Awake()
        {
            Instance = this;
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

            totalFloorWidth = verticalAisleWidth + machineAreaWidth + verticalAisleWidth;
            totalFloorDepth = spineAisleWidth + machineAreaDepth + spineAisleWidth;

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
                MachineType type = config.MachineTypeLayout[i];

                PhysicalMachine pm = Instantiate(prefabToSpawn, worldPos, rotation, transform);
                pm.gameObject.name = $"Machine_{i}_{type}";
                pm.Initialize(i, type);
                machines[i] = pm;
                if (!machinesByType.ContainsKey(type))
                    machinesByType[type] = new List<int>();
                machinesByType[type].Add(i);
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
            OutgoingBeltPosition = new Vector3(floorCentre.x + machineAreaHalfW + verticalAisleWidth, 0.01f, botZ);

            if (conveyorPrefab != null)
            {
                GameObject outBelt = Instantiate(conveyorPrefab, OutgoingBeltPosition, Quaternion.Euler(0, 180, 0), transform);
                outBelt.name = "Outgoing_Belt";
                outBelt.transform.localScale = ioConveyorScale;
                spawnedObjects.Add(outBelt);
                OutgoingBelt = outBelt.GetComponent<ConveyorBelt>();
            }

            AGVParkingPosition = new Vector3(floorCentre.x - machineAreaHalfW, 0.01f, botZ);
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
            SimLogger.Medium($"[FactoryLayout] Distance matrix ({n}×{n}):\n{header}");

            for (int i = 0; i < n; i++)
            {
                string row = $"M{i,-3} ";
                for (int j = 0; j < n; j++) row += $"{distanceMatrix[i, j],6:F1} ";
                SimLogger.High(row);
            }
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