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
        // Dimensions must match Gymnasium Dict/Box space definitions.
        public const int SpatialGridSize = 64;
        public const int SpatialChannels = 3;
        public const int SpatialLength = SpatialGridSize * SpatialGridSize * SpatialChannels;

        public const int MaxJobs = 20;
        public const int MaxMachines = 8;
        public const int SchedChannels = 3;
        public const int SchedulingLength = MaxJobs * (2 * MaxMachines) * SchedChannels;

        public const int GlobalScalarLength = 10;
        public const int DistanceLength = MaxMachines * MaxMachines;
        public const int EventFlagLength = 6;

        public static int TotalObservationSize =>
            SpatialLength + SchedulingLength + GlobalScalarLength + DistanceLength + EventFlagLength;

        private const float NoiseStdDev = 0.02f;
        private const float DropoutRate = 0.05f;

        //  private readonly FactoryOrchestrator _orchestrator;

        public ObservationBuilder(FactoryOrchestrator bridge)
        {
            //_orchestrator = bridge;
        }

        /**
         * @brief Builds the complete observation snapshot sent to ML-Agents.
         * @param currentDecision The decision request triggering this observation.
         * @return Flattened float array containing all observation streams concatenated.
         */
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

        /**
         * @brief Builds the 64x64x3 spatial occupancy grid.
         * @details Channel 0 encodes machine status (idle=0.25, processing=0.75, finished=1.0).
         *          Channel 1 encodes job locations, with intensity set to the normalized
         *          remaining-ops fraction. Channel 2 encodes AGV locations and movement
         *          state (idle=0.25, returning=0.30, moving-to-pickup=0.50, carrying=0.75).
         * @return Flattened grid in channel-major order.
         */
        private float[] BuildSpatialOccupancyGrid()
        {
            float[] grid = new float[SpatialLength];

            FactoryLayoutManager layout = FactoryLayoutManager.Instance;
            if (layout == null || layout.Machines == null) return grid;

            Vector2 floorSize = layout.FloorSize;
            Vector3 floorCentre = layout.GridOrigin;
            float halfW = floorSize.x / 2f;
            float halfD = floorSize.y / 2f;

            // GridOrigin is the top-left of the machine area; compute the actual centre.
            float centreX = floorCentre.x + ((layout.LayoutCols - 1) * layout.MachineSpacingX) / 2f;
            float centreZ = floorCentre.z - ((layout.LayoutRows - 1) * layout.RowPitch) / 2f;

            // Channel 0: Machines
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

            // Channel 1: Jobs
            foreach (JobData job in FactoryOrchestrator.Instance.Jobs.AllJobs)
            {
                if (job.State == JobState.Exited) continue;

                Vector3 pos = GetJobWorldPosition(job, layout);
                int gx, gy;
                WorldToGrid(pos, centreX, centreZ, halfW, halfD, out gx, out gy);
                if (!InBounds(gx, gy)) continue;

                float progress = job.TotalOperations > 0
                    ? 1f - ((float)job.CompletedOps / job.TotalOperations)
                    : 0.5f;

                // Multiple jobs can share a cell; max keeps the strongest signal.
                float existing = GetGrid(grid, 1, gx, gy);
                SetGrid(grid, 1, gx, gy, Mathf.Max(existing, progress));
            }

            // Channel 2: AGVs
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
                    else val = 0.75f;

                    float existing = GetGrid(grid, 2, gx, gy);
                    SetGrid(grid, 2, gx, gy, Mathf.Max(existing, val));
                }
            }

            return grid;
        }

        /**
         * @brief Builds the scheduling matrix of shape (MaxJobs, 2 * MaxMachines, 3).
         * @details Rows are jobs padded/truncated to MaxJobs. The first MaxMachines columns
         *          represent the current operation; the next MaxMachines columns represent
         *          the next operation. Channel 0 is normalized processing time (0 if
         *          ineligible), channel 1 is the completion flag, channel 2 is the machine
         *          busy flag.
         * @return Flattened scheduling matrix.
         */
        private float[] BuildSchedulingMatrix()
        {
            float[] matrix = new float[SchedulingLength];

            IReadOnlyList<JobData> jobs = FactoryOrchestrator.Instance.Jobs.AllJobs;
            FactoryLayoutManager layout = FactoryLayoutManager.Instance;
            if (layout == null) return matrix;

            float maxProcTime = FactoryOrchestrator.Instance.CurrentConfig != null ? FactoryOrchestrator.Instance.CurrentConfig.MaxProcTime : 90f;

            int colWidth = 2 * MaxMachines;

            for (int j = 0; j < Mathf.Min(jobs.Count, MaxJobs); j++)
            {
                JobData job = jobs[j];

                // Current operation slice (columns 0 to MaxMachines-1).
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
                        matrix[baseIdx + 1] = 0f;
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
                            matrix[baseIdx + 1] = 1f;
                        }
                    }
                }

                // Next operation slice (columns MaxMachines to 2*MaxMachines-1).
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

        /**
         * @brief Builds the 10-dimensional global scalar feature vector.
         * @details Encodes simulation time, job state fractions, machine/AGV utilization,
         *          queue pressure, and normalized decision index.
         * @return Float array of length GlobalScalarLength.
         */
        private float[] BuildGlobalScalars()
        {
            float[] s = new float[GlobalScalarLength];
            JobStore store = FactoryOrchestrator.Instance.Jobs;
            FactoryLayoutManager layout = FactoryLayoutManager.Instance;

            int totalJobs = Mathf.Max(store.JobCount, 1);
            int totalMachines = layout != null ? layout.MachineCount : 1;

            // [0] Normalized simulation time against a rough horizon estimate.
            float horizon = FactoryOrchestrator.Instance.CurrentConfig != null
                ? FactoryOrchestrator.Instance.CurrentConfig.MaxProcTime * (FactoryOrchestrator.Instance.CurrentConfig.MaxOpsPerJob)
                : 500f;
            s[0] = Mathf.Clamp01((float)FactoryOrchestrator.Instance.SimTime / horizon);

            // [1] Overall completion ratio.
            s[1] = (float)store.CountInState(JobState.Exited) / totalJobs;

            // [2] Fraction of jobs currently processing.
            s[2] = (float)store.CountInState(JobState.Processing) / totalJobs;

            // [3] Fraction of jobs waiting for pickup or in transit.
            s[3] = (float)(store.CountInState(JobState.WaitingForPickup) +
                           store.CountInState(JobState.InTransit)) / totalJobs;

            // [4] Fraction of jobs queued at machines.
            s[4] = (float)store.CountInState(JobState.Queued) / totalJobs;

            // [5] Fraction of jobs needing routing decisions.
            s[5] = (float)store.CountInState(JobState.NeedsRouting) / totalJobs;

            // [6] Machine utilization.
            if (layout != null && layout.Machines != null)
            {
                int busy = 0;
                foreach (PhysicalMachine m in layout.Machines)
                    if (!m.IsIdle) busy++;
                s[6] = (float)busy / Mathf.Max(totalMachines, 1);
            }

            // [7] AGV utilization.
            if (AGVPool.Instance != null && AGVPool.Instance.AllAGVs.Count > 0)
            {
                int activeAgvs = 0;
                foreach (AGVController agv in AGVPool.Instance.AllAGVs)
                    if (!agv.IsIdle) activeAgvs++;
                s[7] = (float)activeAgvs / AGVPool.Instance.AllAGVs.Count;
            }

            // [8] Queue pressure: max queue load across all machines, normalized.
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

            // [9] Normalized decision index. Each op generates ~2 decisions (routing + dispatch).
            int totalOps = 0;
            foreach (JobData job in store.AllJobs)
                totalOps += job.TotalOperations;
            s[9] = Mathf.Clamp01((float)FactoryOrchestrator.Instance.DecisionCount / Mathf.Max(totalOps * 2, 1));

            return s;
        }

        /**
         * @brief Copies the layout manager's precomputed machine-to-machine distance matrix.
         * @return Flattened distance matrix of length DistanceLength.
         */
        private float[] BuildDistanceMatrix()
        {
            float[] dist = new float[DistanceLength];

            FactoryLayoutManager layout = FactoryLayoutManager.Instance;
            if (layout == null || layout.DistanceMatrixFlat == null) return dist;

            int copyLen = Mathf.Min(layout.DistanceMatrixFlat.Length, dist.Length);
            System.Array.Copy(layout.DistanceMatrixFlat, dist, copyLen);

            return dist;
        }

        /**
         * @brief Builds the 6-dimensional event flag vector.
         * @param req The current decision request.
         * @return Binary/one-hot float array indicating decision type and resource state.
         */
        private float[] BuildEventFlags(DecisionRequest req)
        {
            float[] flags = new float[EventFlagLength];

            // [0] Dispatch decision flag.
            flags[0] = req.Type == DecisionType.Dispatch ? 1f : 0f;

            // [1] Routing decision flag.
            flags[1] = req.Type == DecisionType.Routing ? 1f : 0f;

            // [2] At least one AGV is idle.
            if (AGVPool.Instance != null)
                flags[2] = AGVPool.Instance.GetIdleAGV() != null ? 1f : 0f;

            // [3] At least one machine is idle and has a dispatchable job queued.
            if (FactoryLayoutManager.Instance != null)
            {
                foreach (PhysicalMachine m in FactoryLayoutManager.Instance.Machines)
                {
                    if (m.IsIdle && FactoryOrchestrator.Instance.Jobs.HasDispatchableJob(m.MachineId))
                    {
                        flags[3] = 1f;
                        break;
                    }
                }
            }

            // [4] Past the halfway point of all operations.
            flags[4] = FactoryOrchestrator.Instance.Jobs.CountInState(JobState.Exited) > FactoryOrchestrator.Instance.Jobs.JobCount / 2 ? 1f : 0f;

            // [5] Any machine's AlmostDoneFlag is active.
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

        /**
         * @brief Applies Gaussian noise and dropout to observation streams for robustness.
         * @param spatial Spatial grid stream.
         * @param scheduling Scheduling matrix stream.
         * @param scalars Global scalar stream.
         */
        private void ApplyDomainRandomization(float[] spatial, float[] scheduling, float[] scalars)
        {
            AddGaussianNoise(spatial, NoiseStdDev);
            AddGaussianNoise(scheduling, NoiseStdDev * 0.5f);
            AddGaussianNoise(scalars, NoiseStdDev * 0.25f);

            ApplyDropout(spatial, DropoutRate);
            ApplyDropout(scheduling, DropoutRate * 0.5f);
            // Scalars are low-dimensional; every element matters, so no dropout.
        }

        /**
         * @brief Adds Gaussian noise to non-zero elements of the given array in place.
         * @param data Array to perturb.
         * @param stdDev Standard deviation of the Gaussian distribution.
         */
        private static void AddGaussianNoise(float[] data, float stdDev)
        {
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] == 0f) continue;
                data[i] += SampleGaussian() * stdDev;
            }
        }

        /**
         * @brief Zeros out elements of the array with a given probability.
         * @param data Array to apply dropout to.
         * @param rate Probability of zeroing each element.
         */
        private static void ApplyDropout(float[] data, float rate)
        {
            for (int i = 0; i < data.Length; i++)
            {
                if (Random.value < rate)
                    data[i] = 0f;
            }
        }

        /**
         * @brief Samples a single value from a standard normal distribution using Box-Muller.
         * @return A random float with mean 0 and standard deviation 1.
         */
        private static float SampleGaussian()
        {
            float u1 = 1f - Random.value;
            float u2 = Random.value;
            return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Sin(2f * Mathf.PI * u2);
        }

        /**
         * @brief Concatenates multiple float arrays into a single contiguous array.
         * @param streams Arrays to concatenate in order.
         * @return Single flattened array containing all stream data.
         */
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

        /**
         * @brief Converts a world-space XZ position into grid cell coordinates.
         * @param worldPos World-space position to convert.
         * @param centreX X coordinate of the factory floor centre.
         * @param centreZ Z coordinate of the factory floor centre.
         * @param halfW Half-width of the factory floor on the X axis.
         * @param halfD Half-depth of the factory floor on the Z axis.
         * @param gx Output grid X index, clamped to valid range.
         * @param gy Output grid Y index, clamped to valid range.
         */
        private void WorldToGrid(Vector3 worldPos, float centreX, float centreZ,
                                  float halfW, float halfD, out int gx, out int gy)
        {
            float nx = (worldPos.x - centreX + halfW) / (2f * halfW);
            float nz = (worldPos.z - centreZ + halfD) / (2f * halfD);

            gx = Mathf.Clamp(Mathf.FloorToInt(nx * SpatialGridSize), 0, SpatialGridSize - 1);
            gy = Mathf.Clamp(Mathf.FloorToInt(nz * SpatialGridSize), 0, SpatialGridSize - 1);
        }

        /**
         * @brief Checks whether grid coordinates fall within the spatial grid.
         * @param gx Grid X index.
         * @param gy Grid Y index.
         * @return True if both indices are within bounds.
         */
        private static bool InBounds(int gx, int gy)
        {
            return gx >= 0 && gx < SpatialGridSize && gy >= 0 && gy < SpatialGridSize;
        }

        /**
         * @brief Writes a value into the spatial grid using channel-major indexing.
         * @param grid Flat grid array.
         * @param channel Channel index.
         * @param gx Grid X index.
         * @param gy Grid Y index.
         * @param value Value to write.
         */
        private static void SetGrid(float[] grid, int channel, int gx, int gy, float value)
        {
            grid[channel * (SpatialGridSize * SpatialGridSize) + gy * SpatialGridSize + gx] = value;
        }

        /**
         * @brief Reads a value from the spatial grid using channel-major indexing.
         * @param grid Flat grid array.
         * @param channel Channel index.
         * @param gx Grid X index.
         * @param gy Grid Y index.
         * @return The stored value at the given cell.
         */
        private static float GetGrid(float[] grid, int channel, int gx, int gy)
        {
            return grid[channel * (SpatialGridSize * SpatialGridSize) + gy * SpatialGridSize + gx];
        }

        /**
         * @brief Resolves a job's current world-space position based on its lifecycle state.
         * @details Prefers the visual transform when available. Otherwise falls back to the
         *          machine location, assigned AGV position, or the incoming belt position
         *          depending on the job's state.
         * @param job The job whose position is requested.
         * @param layout The factory layout manager providing machine and belt references.
         * @return World-space position of the job.
         */
        private Vector3 GetJobWorldPosition(JobData job, FactoryLayoutManager layout)
        {
            if (job.Visual != null && job.Visual.gameObject.activeInHierarchy)
                return job.Visual.transform.position;

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
                    return layout.IncomingBeltPosition;
            }

            return layout.IncomingBeltPosition;
        }
    }
}