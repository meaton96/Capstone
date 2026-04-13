using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.AGV;
using Assets.Scripts.Simulation.FactoryLayout;

namespace Assets.Scripts.Simulation
{
    public class ObservationBuilder
    {
        // ── Dimensions (must match Gymnasium Dict/Box space definitions) ──
        public const int SpatialGridSize = 64;
        public const int SpatialChannels = 3;
        public const int SpatialLength = SpatialGridSize * SpatialGridSize * SpatialChannels;

        public const int MaxJobs = 20;       // n  — pad/truncate to fixed width
        public const int MaxMachines = 8;     // m  — half-columns in scheduling matrix
        public const int SchedChannels = 3;
        public const int SchedulingLength = MaxJobs * (2 * MaxMachines) * SchedChannels;

        public const int GlobalScalarLength = 10;
        public const int DistanceLength = MaxMachines * MaxMachines; // 64-D flattened
        public const int EventFlagLength = 6;

        /// Total observation width sent to ML-Agents.
        public static int TotalObservationSize =>
            SpatialLength + SchedulingLength + GlobalScalarLength + DistanceLength + EventFlagLength;

        // ── Domain randomization parameters ──
        private const float NoiseStdDev = 0.02f;
        private const float DropoutRate = 0.05f;

        private readonly SimulationBridge _bridge;

        public ObservationBuilder(SimulationBridge bridge)
        {
            _bridge = bridge;
        }

        // ═══════════════════════════════════════════════════════════════════
        //  Public entry point — called by SchedulingAgent.CollectObservations
        // ═══════════════════════════════════════════════════════════════════

        public float[] BuildCompleteSnapshot(DecisionRequest currentDecision)
        {
            float[] spatialGrid = BuildSpatialOccupancyGrid();
            float[] schedulingMatrix = BuildSchedulingMatrix();
            float[] globalScalars = BuildGlobalScalars();
            float[] distanceMatrix = BuildDistanceMatrix();
            float[] eventFlags = BuildEventFlags(currentDecision);

            // ApplyDomainRandomization(spatialGrid, schedulingMatrix, globalScalars);

            return FlattenStreams(spatialGrid, schedulingMatrix, globalScalars, distanceMatrix, eventFlags);
        }

        // ═══════════════════════════════════════════════════════════════════
        //  Stream 1 — Spatial Occupancy Grid  (64 × 64 × 3)
        // ═══════════════════════════════════════════════════════════════════
        //
        //  Channel 0: Machine locations and status
        //      idle = 0.25, processing = 0.75, finished = 1.0
        //  Channel 1: Job physical locations
        //      value = normalized remaining-ops fraction (0→1)
        //  Channel 2: AGV locations and heading
        //      idle/parked = 0.25, moving-to-pickup = 0.5, carrying = 0.75, returning = 0.3

        private float[] BuildSpatialOccupancyGrid()
        {
            float[] grid = new float[SpatialLength];

            FactoryLayoutManager layout = FactoryLayoutManager.Instance;
            if (layout == null || layout.Machines == null) return grid;

            // Determine world-space bounds for normalizing positions → grid cells.
            Vector2 floorSize = layout.FloorSize;                 // (width, depth)
            Vector3 floorCentre = layout.GridOrigin;              // top-left of machine area
            // Use a bounding box centred on the factory.
            float halfW = floorSize.x / 2f;
            float halfD = floorSize.y / 2f;
            // Floor centre from the transform (GridOrigin is the top-left; we need the actual centre).
            // Approximate: midpoint is GridOrigin shifted by half machine area.
            float centreX = floorCentre.x + ((layout.LayoutCols - 1) * layout.MachineSpacingX) / 2f;
            float centreZ = floorCentre.z - ((layout.LayoutRows - 1) * layout.RowPitch) / 2f;

            // ── Channel 0: Machines ──
            foreach (PhysicalMachine m in layout.Machines)
            {
                int gx, gy;
                WorldToGrid(m.transform.position, centreX, centreZ, halfW, halfD, out gx, out gy);
                if (!InBounds(gx, gy)) continue;

                float val;
                if (m.FinishedFlag) val = 1.0f;
                else if (!m.IsIdle) val = 0.75f;
                else val = 0.25f;

                SetGrid(grid, 0, gx, gy, val);
            }

            // ── Channel 1: Jobs ──
            foreach (JobData job in _bridge.Jobs.AllJobs)
            {
                if (job.State == JobState.Exited) continue;

                Vector3 pos = GetJobWorldPosition(job, layout);
                int gx, gy;
                WorldToGrid(pos, centreX, centreZ, halfW, halfD, out gx, out gy);
                if (!InBounds(gx, gy)) continue;

                // Encode progress: 1.0 = just started, trending to 0.0 as ops complete.
                float progress = job.TotalOperations > 0
                    ? 1f - ((float)job.CompletedOps / job.TotalOperations)
                    : 0.5f;

                // Accumulate (multiple jobs can share a cell — max keeps strongest signal).
                float existing = GetGrid(grid, 1, gx, gy);
                SetGrid(grid, 1, gx, gy, Mathf.Max(existing, progress));
            }

            // ── Channel 2: AGVs ──
            if (AGVPool.Instance != null)
            {
                foreach (AGVController agv in AGVPool.Instance.AllAGVs)
                {
                    int gx, gy;
                    WorldToGrid(agv.transform.position, centreX, centreZ, halfW, halfD, out gx, out gy);
                    if (!InBounds(gx, gy)) continue;

                    float val;
                    if (agv.IsIdle) val = 0.25f;
                    else if (agv.State == AGVState.ReturningToParking) val = 0.30f;
                    else if (agv.State == AGVState.MovingToPickup ||
                             agv.State == AGVState.MovingToPrePickup) val = 0.50f;
                    else val = 0.75f; // MovingToDropoff

                    float existing = GetGrid(grid, 2, gx, gy);
                    SetGrid(grid, 2, gx, gy, Mathf.Max(existing, val));
                }
            }

            return grid;
        }

        // ═══════════════════════════════════════════════════════════════════
        //  Stream 2 — Scheduling Matrix  (n × 2m × 3)
        // ═══════════════════════════════════════════════════════════════════
        //
        //  Rows    = jobs  (padded/truncated to MaxJobs)
        //  Columns = 2 × MaxMachines (first m = current op, second m = next op)
        //  Depth channels:
        //      ch 0: normalized processing time  (0 if ineligible)
        //      ch 1: completion flag              (1 if op already done)
        //      ch 2: machine busy flag            (1 if machine currently processing)

        private float[] BuildSchedulingMatrix()
        {
            float[] matrix = new float[SchedulingLength];

            IReadOnlyList<JobData> jobs = _bridge.Jobs.AllJobs;
            FactoryLayoutManager layout = FactoryLayoutManager.Instance;
            if (layout == null) return matrix;

            // Determine normalization ceiling from config.
            float maxProcTime = _bridge.CurrentConfig != null ? _bridge.CurrentConfig.MaxProcTime : 90f;

            int colWidth = 2 * MaxMachines;

            for (int j = 0; j < Mathf.Min(jobs.Count, MaxJobs); j++)
            {
                JobData job = jobs[j];

                // ── Current operation slice (columns 0 .. MaxMachines-1) ──
                if (job.CurrentOpIndex < job.TotalOperations)
                {
                    var eligible = job.EligibleMachinesPerOp[job.CurrentOpIndex];
                    foreach (var kvp in eligible)
                    {
                        int machId = kvp.Key;
                        if (machId >= MaxMachines) continue;

                        float normTime = kvp.Value / Mathf.Max(maxProcTime, 0.001f);

                        int baseIdx = (j * colWidth + machId) * SchedChannels;
                        matrix[baseIdx + 0] = Mathf.Clamp01(normTime);
                        matrix[baseIdx + 1] = 0f; // not yet completed
                        matrix[baseIdx + 2] = layout.GetMachine(machId) != null && !layout.GetMachine(machId).IsIdle ? 1f : 0f;
                    }
                }

                // Mark completed ops.
                if (job.CompletedOps > 0 && job.CurrentOpIndex > 0)
                {
                    int prevOp = job.CurrentOpIndex - 1;
                    if (prevOp < job.EligibleMachinesPerOp.Length)
                    {
                        foreach (var kvp in job.EligibleMachinesPerOp[prevOp])
                        {
                            int machId = kvp.Key;
                            if (machId >= MaxMachines) continue;
                            int baseIdx = (j * colWidth + machId) * SchedChannels;
                            matrix[baseIdx + 1] = 1f; // completed
                        }
                    }
                }

                // ── Next operation slice (columns MaxMachines .. 2*MaxMachines-1) ──
                int nextOp = job.CurrentOpIndex + 1;
                if (nextOp < job.TotalOperations)
                {
                    var eligible = job.EligibleMachinesPerOp[nextOp];
                    foreach (var kvp in eligible)
                    {
                        int machId = kvp.Key;
                        if (machId >= MaxMachines) continue;

                        float normTime = kvp.Value / Mathf.Max(maxProcTime, 0.001f);

                        int col = MaxMachines + machId;
                        int baseIdx = (j * colWidth + col) * SchedChannels;
                        matrix[baseIdx + 0] = Mathf.Clamp01(normTime);
                        matrix[baseIdx + 1] = 0f;
                        matrix[baseIdx + 2] = layout.GetMachine(machId) != null && !layout.GetMachine(machId).IsIdle ? 1f : 0f;
                    }
                }
            }

            return matrix;
        }

        // ═══════════════════════════════════════════════════════════════════
        //  Stream 3 — Global Scalars  (10-D)
        // ═══════════════════════════════════════════════════════════════════

        private float[] BuildGlobalScalars()
        {
            float[] s = new float[GlobalScalarLength];
            JobStore store = _bridge.Jobs;
            FactoryLayoutManager layout = FactoryLayoutManager.Instance;

            int totalJobs = Mathf.Max(store.JobCount, 1);
            int totalMachines = layout != null ? layout.MachineCount : 1;

            // [0] Normalized simulation time (relative to a rough horizon estimate).
            float horizon = _bridge.CurrentConfig != null
                ? _bridge.CurrentConfig.MaxProcTime * (_bridge.CurrentConfig.MaxOpsPerJob)
                : 500f;
            s[0] = Mathf.Clamp01((float)_bridge.SimTime / horizon);

            // [1] Overall completion ratio.
            s[1] = (float)store.CountInState(JobState.Exited) / totalJobs;

            // [2] Fraction of jobs currently in processing.
            s[2] = (float)store.CountInState(JobState.Processing) / totalJobs;

            // [3] Fraction of jobs waiting for pickup / in transit.
            s[3] = (float)(store.CountInState(JobState.WaitingForPickup) +
                           store.CountInState(JobState.InTransit)) / totalJobs;

            // [4] Fraction of jobs queued at machines.
            s[4] = (float)store.CountInState(JobState.Queued) / totalJobs;

            // [5] Fraction of jobs needing routing decisions.
            s[5] = (float)store.CountInState(JobState.NeedsRouting) / totalJobs;

            // [6] Machine utilization — fraction of machines currently busy.
            if (layout != null && layout.Machines != null)
            {
                int busy = 0;
                foreach (PhysicalMachine m in layout.Machines)
                    if (!m.IsIdle) busy++;
                s[6] = (float)busy / Mathf.Max(totalMachines, 1);
            }

            // [7] AGV utilization — fraction of AGVs not idle.
            if (AGVPool.Instance != null && AGVPool.Instance.AllAGVs.Count > 0)
            {
                int activeAgvs = 0;
                foreach (AGVController agv in AGVPool.Instance.AllAGVs)
                    if (!agv.IsIdle) activeAgvs++;
                s[7] = (float)activeAgvs / AGVPool.Instance.AllAGVs.Count;
            }

            // [8] Queue pressure — max queue load across all machines, normalized.
            if (layout != null && layout.Machines != null)
            {
                float maxLoad = 0f;
                foreach (PhysicalMachine m in layout.Machines)
                {
                    float load = store.GetMachineLoad(m.MachineId);
                    if (load > maxLoad) maxLoad = load;
                }
                s[8] = Mathf.Clamp01(maxLoad / Mathf.Max(horizon, 1f));
            }

            // [9] Normalized decision index (how far into the episode are we?).
            // Use total ops as a rough ceiling for how many decisions we expect.
            int totalOps = 0;
            foreach (JobData job in store.AllJobs)
                totalOps += job.TotalOperations;
            // Each op generates ~2 decisions (routing + dispatch), so total ≈ 2 × totalOps.
            s[9] = Mathf.Clamp01((float)_bridge.DecisionCount / Mathf.Max(totalOps * 2, 1));

            return s;
        }

        // ═══════════════════════════════════════════════════════════════════
        //  Stream 4 — Distance Matrix  (MaxMachines × MaxMachines, flattened)
        // ═══════════════════════════════════════════════════════════════════

        private float[] BuildDistanceMatrix()
        {
            float[] dist = new float[DistanceLength];

            FactoryLayoutManager layout = FactoryLayoutManager.Instance;
            if (layout == null || layout.DistanceMatrixFlat == null) return dist;

            // LayoutManager already computes a normalized 8×8 flat array — copy directly.
            int copyLen = Mathf.Min(layout.DistanceMatrixFlat.Length, dist.Length);
            System.Array.Copy(layout.DistanceMatrixFlat, dist, copyLen);

            return dist;
        }

        // ═══════════════════════════════════════════════════════════════════
        //  Stream 5 — Event Flags  (6-D one-hot / binary triggers)
        // ═══════════════════════════════════════════════════════════════════

        private float[] BuildEventFlags(DecisionRequest req)
        {
            float[] flags = new float[EventFlagLength];

            // [0] Is this a Dispatch decision?
            flags[0] = req.Type == DecisionType.Dispatch ? 1f : 0f;

            // [1] Is this a Routing decision?
            flags[1] = req.Type == DecisionType.Routing ? 1f : 0f;

            // [2] Are any AGVs idle right now? (resource-availability signal)
            if (AGVPool.Instance != null)
                flags[2] = AGVPool.Instance.GetIdleAGV() != null ? 1f : 0f;

            // [3] Is any machine idle AND has a dispatchable job queued?
            //     (urgency signal — work is waiting)
            if (FactoryLayoutManager.Instance != null)
            {
                foreach (PhysicalMachine m in FactoryLayoutManager.Instance.Machines)
                {
                    if (m.IsIdle && _bridge.Jobs.HasDispatchableJob(m.MachineId))
                    {
                        flags[3] = 1f;
                        break;
                    }
                }
            }

            // [4] Are we past the halfway point of all operations?
            flags[4] = _bridge.Jobs.CountInState(JobState.Exited) > _bridge.Jobs.JobCount / 2 ? 1f : 0f;

            // [5] Is any machine's AlmostDoneFlag active? (imminent completion)
            if (FactoryLayoutManager.Instance != null)
            {
                foreach (PhysicalMachine m in FactoryLayoutManager.Instance.Machines)
                {
                    if (m.AlmostDoneFlag)
                    {
                        flags[5] = 1f;
                        break;
                    }
                }
            }

            return flags;
        }

        // ═══════════════════════════════════════════════════════════════════
        //  Domain Randomization — noise injection & sensor dropout
        // ═══════════════════════════════════════════════════════════════════

        private void ApplyDomainRandomization(float[] spatial, float[] scheduling, float[] scalars)
        {
            // Only apply during training (check Time.timeScale > 1 as a proxy,
            // or always apply and let the policy become robust).
            AddGaussianNoise(spatial, NoiseStdDev);
            AddGaussianNoise(scheduling, NoiseStdDev * 0.5f);   // lighter noise on scheduling data
            AddGaussianNoise(scalars, NoiseStdDev * 0.25f);      // very light on global scalars

            ApplyDropout(spatial, DropoutRate);
            ApplyDropout(scheduling, DropoutRate * 0.5f);
            // No dropout on scalars — they are low-dimensional and each element matters.
        }

        private static void AddGaussianNoise(float[] data, float stdDev)
        {
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] == 0f) continue; // don't inject noise into empty cells
                data[i] += SampleGaussian() * stdDev;
            }
        }

        private static void ApplyDropout(float[] data, float rate)
        {
            for (int i = 0; i < data.Length; i++)
            {
                if (Random.value < rate)
                    data[i] = 0f;
            }
        }

        /// Box-Muller transform for Gaussian sampling without System.Random.
        private static float SampleGaussian()
        {
            float u1 = 1f - Random.value; // (0,1] to avoid log(0)
            float u2 = Random.value;
            return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Sin(2f * Mathf.PI * u2);
        }

        // ═══════════════════════════════════════════════════════════════════
        //  Helpers
        // ═══════════════════════════════════════════════════════════════════

        private float[] FlattenStreams(params float[][] streams)
        {
            int totalLength = 0;
            foreach (var stream in streams) totalLength += stream.Length;

            float[] flattened = new float[totalLength];
            int offset = 0;
            foreach (var stream in streams)
            {
                System.Array.Copy(stream, 0, flattened, offset, stream.Length);
                offset += stream.Length;
            }
            return flattened;
        }

        /// Converts a world XZ position into grid cell coordinates.
        private void WorldToGrid(Vector3 worldPos, float centreX, float centreZ,
                                  float halfW, float halfD, out int gx, out int gy)
        {
            // Normalize to [0, 1] then scale to grid indices.
            float nx = (worldPos.x - centreX + halfW) / (2f * halfW);
            float nz = (worldPos.z - centreZ + halfD) / (2f * halfD);

            gx = Mathf.Clamp(Mathf.FloorToInt(nx * SpatialGridSize), 0, SpatialGridSize - 1);
            gy = Mathf.Clamp(Mathf.FloorToInt(nz * SpatialGridSize), 0, SpatialGridSize - 1);
        }

        private static bool InBounds(int gx, int gy)
        {
            return gx >= 0 && gx < SpatialGridSize && gy >= 0 && gy < SpatialGridSize;
        }

        /// Channel-major indexing:  index = channel * (W*H) + gy * W + gx
        private static void SetGrid(float[] grid, int channel, int gx, int gy, float value)
        {
            grid[channel * (SpatialGridSize * SpatialGridSize) + gy * SpatialGridSize + gx] = value;
        }

        private static float GetGrid(float[] grid, int channel, int gx, int gy)
        {
            return grid[channel * (SpatialGridSize * SpatialGridSize) + gy * SpatialGridSize + gx];
        }

        /// Resolves a job's current world position based on its lifecycle state.
        private Vector3 GetJobWorldPosition(JobData job, FactoryLayoutManager layout)
        {
            // If the job has a visual, use its actual transform (most accurate).
            if (job.Visual != null && job.Visual.gameObject.activeInHierarchy)
                return job.Visual.transform.position;

            // Fallback: infer from state.
            switch (job.State)
            {
                case JobState.Processing:
                case JobState.Queued:
                case JobState.WaitingForPickup:
                    if (job.LocationMachineId >= 0)
                    {
                        PhysicalMachine m = layout.GetMachine(job.LocationMachineId);
                        if (m != null) return m.transform.position;
                    }
                    break;

                case JobState.InTransit:
                    // Job is on an AGV — try to find it.
                    if (job.AssignedAgvId >= 0 && AGVPool.Instance != null)
                    {
                        foreach (AGVController agv in AGVPool.Instance.AllAGVs)
                        {
                            if (agv.AgvId == job.AssignedAgvId)
                                return agv.transform.position;
                        }
                    }
                    break;

                case JobState.NeedsRouting:
                    if (job.LocationMachineId >= 0)
                    {
                        PhysicalMachine m = layout.GetMachine(job.LocationMachineId);
                        if (m != null) return m.transform.position;
                    }
                    // First operation — job is at factory entrance.
                    return layout.IncomingBeltPosition;
            }

            return layout.IncomingBeltPosition;
        }
    }
}