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

        // Observation schema v2 (2026-09-24). v1 had an 8-machine x first-20-jobs scheduling matrix and an
        // 8x8 distance matrix, which on the 15-machine floor dropped machines 8-14 and went blank once
        // the first 20 jobs exited. v2 uses per-entity tables sized for the largest planned floor (100
        // machines, E4); the Python side encodes them with set pooling, so one network serves any floor
        // up to MaxMachines. Keep env/config.py in sync.
        public const int MaxMachines = 100;
        public const int MachineFeatures = 16;
        public const int MachineTableLength = MaxMachines * MachineFeatures;

        public const int MaxJobs = 256;   // 64 until 2026-09-25: randomized-family WIP peaks reached 165
        public const int JobFeatures = 17;
        public const int JobTableLength = MaxJobs * JobFeatures;

        public const int GlobalScalarLength = 16;
        public const int EventFlagLength = 6;

        public static int TotalObservationSize =>
            SpatialLength + MachineTableLength + JobTableLength + GlobalScalarLength + EventFlagLength;

        // Half-saturation scales for Squash() (sim-seconds, or counts).
        private const float TimeScale = 300f;     // operation / repair / waiting durations
        private const float AgeScale = 1200f;     // job age and remaining work (several operations)
        private const float HorizonScale = 7200f; // sim time
        private const float CountScale = 5f;      // queue lengths, candidate counts, eligible machines
        private const float OpsScale = 3f;        // remaining operations
        private const float WipScale = 30f;       // active jobs

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
            float[] machineTable = BuildMachineTable(currentDecision);
            float[] jobTable = BuildJobTable(currentDecision);
            float[] globalScalars = BuildGlobalScalars(currentDecision);
            float[] eventFlags = BuildEventFlags(currentDecision);

            // ApplyDomainRandomization(spatialGrid, jobTable, globalScalars);

            return FlattenStreams(spatialGrid, machineTable, jobTable, globalScalars, eventFlags);
        }

        /**
         * @brief Builds the 64x64x3 spatial occupancy grid.
         * @details Channel 0 encodes machine status (idle=0.25, down=0.5, processing=0.75, finished=1.0).
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
                if (!m.IsAvailableForWork) val = 0.5f;   // down: a repairing machine is otherwise IsIdle
                else if (m.FinishedFlag) val = 1.0f;
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
         * @brief Bounded, scale-free squash for non-negative magnitudes: x / (x + scale), in [0, 1).
         * @details Durations and counts have no fixed upper bound (op lengths and WIP vary per
         *          episode), so min-max normalisation against a config maximum either saturates or
         *          wastes range. Half-saturation at @p scale keeps typical values mid-range.
         */
        private static float Squash(double x, float scale)
        {
            if (x <= 0.0) return 0f;
            return (float)(x / (x + scale));
        }

        /**
         * @brief One-hot index of a job state among the five pre-exit states, or -1 for Exited.
         */
        private static int StateSlot(JobState s) => s switch
        {
            JobState.NeedsRouting => 0,
            JobState.WaitingForPickup => 1,
            JobState.InTransit => 2,
            JobState.Queued => 3,
            JobState.Processing => 4,
            _ => -1,
        };

        /**
         * @brief Builds the machine table, shape (MaxMachines, MachineFeatures), one row per machine
         *        in machine-id order, zero-padded past the floor's machine count.
         * @details Features (all in [0, 1]):
         *   [0]  present (1 for a real machine; padding rows are all zero)
         *   [1-5] primary type one-hot (Mill, Lathe, Weld, Inspect, Assemble)
         *   [6]  busy (processing an operation)
         *   [7]  operational (0 while Failed or Repairing)
         *   [8]  remaining repair time, squashed (TimeScale)
         *   [9]  remaining processing time of the current operation, squashed (TimeScale)
         *   [10] committed work: queued + in-transit/awaiting-pickup jobs targeted here, squashed (TimeScale)
         *   [11] jobs physically queued here, squashed (CountScale)
         *   [12] cumulative utilisation, busy / operational time so far (the MMUR signal)
         *   [13] candidate for the current decision (routing: eligible and operational; dispatch: the machine dispatching)
         *   [14] routing only: processing time of the focus job's current op on this machine, squashed (TimeScale)
         *   [15] routing only: distance from the focus job's position to this machine / floor diagonal
         * The focus job is DecisionRequest.JobId: with several routable jobs and no rule known yet (agent
         * mode) it is the oldest routable job, and the action's rule may route another candidate instead
         * (FactoryOrchestrator.ExecuteRoutingDecision); those candidates' own signals are in the job table.
         */
        private float[] BuildMachineTable(DecisionRequest req)
        {
            float[] table = new float[MachineTableLength];
            FactoryOrchestrator orch = FactoryOrchestrator.Instance;
            FactoryLayoutManager layout = FactoryLayoutManager.Instance;
            if (layout == null || layout.Machines == null || orch == null) return table;

            JobStore store = orch.Jobs;
            var queuedCount = new Dictionary<int, int>();
            foreach (JobData job in store.AllJobs)
                if (job.State == JobState.Queued && job.LocationMachineId >= 0)
                    queuedCount[job.LocationMachineId] = queuedCount.TryGetValue(job.LocationMachineId, out int c) ? c + 1 : 1;

            var candidates = new HashSet<int>();
            JobData focus = null;
            Vector3 focusPos = Vector3.zero;
            if (req != null && req.Type == DecisionType.Routing)
            {
                if (req.CandidateMachineIds != null) candidates.UnionWith(req.CandidateMachineIds);
                focus = store.Get(req.JobId);
                if (focus != null) focusPos = GetJobWorldPosition(focus, layout);
            }
            else if (req != null && req.Type == DecisionType.Dispatch)
            {
                candidates.Add(req.MachineId);
            }

            Vector2 floor = layout.FloorSize;
            float diag = Mathf.Max(floor.magnitude, 1f);

            int n = Mathf.Min(layout.Machines.Count, MaxMachines);
            for (int i = 0; i < n; i++)
            {
                PhysicalMachine m = layout.Machines[i];
                int b = i * MachineFeatures;
                table[b + 0] = 1f;
                int type = (int)m.PrimaryType;
                if (type >= 0 && type < 5) table[b + 1 + type] = 1f;
                table[b + 6] = m.IsIdle ? 0f : 1f;
                table[b + 7] = m.IsAvailableForWork ? 1f : 0f;
                table[b + 8] = Squash(m.RemainingRepairTime, TimeScale);
                table[b + 9] = Squash(m.RemainingProcessingTime, TimeScale);
                table[b + 10] = Squash(store.GetMachineLoad(m.MachineId), TimeScale);
                table[b + 11] = Squash(queuedCount.TryGetValue(m.MachineId, out int q) ? q : 0, CountScale);
                table[b + 12] = Mathf.Clamp01(orch.MachineUtilization(m.MachineId));
                table[b + 13] = candidates.Contains(m.MachineId) ? 1f : 0f;
                if (focus != null && focus.CurrentOpIndex < focus.TotalOperations
                    && focus.EligibleMachinesPerOp[focus.CurrentOpIndex].TryGetValue(m.MachineId, out float pt))
                {
                    table[b + 14] = Squash(pt, TimeScale);
                    table[b + 15] = Mathf.Clamp01(Vector3.Distance(focusPos, m.transform.position) / diag);
                }
            }
            return table;
        }

        /**
         * @brief Builds the job table, shape (MaxJobs, JobFeatures), over ACTIVE jobs only
         *        (arrived and not exited), zero-padded.
         * @details The store keeps every job ever spawned, so the old matrix (first MaxJobs entries
         *          of AllJobs) went blank once the first jobs exited. Rows are filled in priority
         *          order so truncation drops the least relevant jobs: the routing focus job, then
         *          the other decision candidates (routable jobs / the dispatching machine's queue),
         *          then every other active job oldest-first. Row order carries no meaning beyond
         *          that; the Python side pools over rows.
         *   [0]  present
         *   [1-5] state one-hot (NeedsRouting, WaitingForPickup, InTransit, Queued, Processing)
         *   [6]  deferred (every eligible machine for its next op is down)
         *   [7]  age since arrival, squashed (AgeScale)
         *   [8]  time in current state, squashed (TimeScale)
         *   [9]  remaining operations incl. current, squashed (OpsScale)
         *   [10] current op's minimum processing time over eligible machines, squashed (TimeScale)
         *   [11] remaining work: sum of per-op minimum processing times, squashed (AgeScale)
         *   [12] eligible machines for the current op, squashed (CountScale)
         *   [13] fraction of those eligible machines currently operational
         *   [14] routing focus job (DecisionRequest.JobId; see BuildMachineTable on agent mode)
         *   [15] decision candidate (routing: routable job; dispatch: queued at the dispatching machine)
         *   [16] dispatch only: processing time on the dispatching machine, squashed (TimeScale)
         */
        private float[] BuildJobTable(DecisionRequest req)
        {
            float[] table = new float[JobTableLength];
            FactoryOrchestrator orch = FactoryOrchestrator.Instance;
            FactoryLayoutManager layout = FactoryLayoutManager.Instance;
            if (orch == null || layout == null) return table;

            JobStore store = orch.Jobs;
            double now = orch.SimTime;

            int focusId = -1;
            var candidateIds = new HashSet<int>();
            int dispatchMachine = -1;
            if (req != null && req.Type == DecisionType.Routing)
            {
                focusId = req.JobId;
                if (req.JobCandidateIds != null) candidateIds.UnionWith(req.JobCandidateIds);
                candidateIds.Add(req.JobId);
            }
            else if (req != null && req.Type == DecisionType.Dispatch)
            {
                dispatchMachine = req.MachineId;
                if (req.QueuedJobIds != null) candidateIds.UnionWith(req.QueuedJobIds);
            }

            // AllJobs is in arrival order, so a stable partition keeps "oldest first" within each tier.
            var rows = new List<JobData>();
            var rest = new List<JobData>();
            JobData focus = null;
            foreach (JobData job in store.AllJobs)
            {
                if (job.State == JobState.Exited) continue;
                if (job.JobId == focusId) focus = job;
                else if (candidateIds.Contains(job.JobId)) rows.Add(job);
                else rest.Add(job);
            }
            if (focus != null) rows.Insert(0, focus);
            rows.AddRange(rest);

            int n = Mathf.Min(rows.Count, MaxJobs);
            for (int r = 0; r < n; r++)
            {
                JobData job = rows[r];
                int b = r * JobFeatures;
                table[b + 0] = 1f;
                int slot = StateSlot(job.State);
                if (slot >= 0) table[b + 1 + slot] = 1f;
                table[b + 6] = store.DeferredJobIds.Contains(job.JobId) ? 1f : 0f;
                table[b + 7] = Squash(now - job.ArrivalTime, AgeScale);
                table[b + 8] = Squash(now - job.StateEntryTime, TimeScale);
                table[b + 9] = Squash(job.TotalOperations - job.CurrentOpIndex, OpsScale);

                if (job.CurrentOpIndex < job.TotalOperations)
                {
                    var eligible = job.EligibleMachinesPerOp[job.CurrentOpIndex];
                    float minPt = float.MaxValue;
                    int up = 0;
                    foreach (var kvp in eligible)
                    {
                        if (kvp.Value < minPt) minPt = kvp.Value;
                        PhysicalMachine m = layout.GetMachine(kvp.Key);
                        if (m != null && m.IsAvailableForWork) up++;
                    }
                    if (eligible.Count > 0)
                    {
                        table[b + 10] = Squash(minPt, TimeScale);
                        table[b + 12] = Squash(eligible.Count, CountScale);
                        table[b + 13] = (float)up / eligible.Count;
                    }
                    table[b + 11] = Squash(DispatchingEngine.GetRemainingWork(job.JobId, store), AgeScale);
                }

                table[b + 14] = job.JobId == focusId ? 1f : 0f;
                table[b + 15] = candidateIds.Contains(job.JobId) ? 1f : 0f;
                if (dispatchMachine >= 0 && job.State == JobState.Queued && job.LocationMachineId == dispatchMachine)
                    table[b + 16] = Squash(job.GetProcessingTime(dispatchMachine), TimeScale);
            }
            return table;
        }

        /**
         * @brief Builds the global scalar vector (GlobalScalarLength).
         * @details Floor-level aggregates. The old version normalised time and queue pressure
         *          against MaxProcTime x MaxOpsPerJob (~360 s on the compound scenarios), so both
         *          saturated within minutes; everything here is squashed or a true fraction.
         *   [0]  sim time, squashed (HorizonScale)
         *   [1]  active jobs (WIP), squashed (WipScale)
         *   [2-6] fraction of active jobs in NeedsRouting / WaitingForPickup / InTransit / Queued / Processing
         *   [7]  fraction of machines busy
         *   [8]  fraction of machines down (Failed or Repairing)
         *   [9]  fraction of AGVs busy
         *   [10] mean committed work per machine, squashed (TimeScale)
         *   [11] fraction of active jobs that did not fit in the job table
         *   [12] machine count / MaxMachines
         *   [13] AGVs per machine (clamped to 1)
         *   [14] deferred jobs, squashed (CountScale)
         *   [15] options in the current decision (routing: candidate machines; dispatch: queued jobs), squashed (CountScale)
         */
        private float[] BuildGlobalScalars(DecisionRequest req)
        {
            float[] s = new float[GlobalScalarLength];
            FactoryOrchestrator orch = FactoryOrchestrator.Instance;
            FactoryLayoutManager layout = FactoryLayoutManager.Instance;
            if (orch == null) return s;
            JobStore store = orch.Jobs;

            int[] byState = new int[5];
            int active = 0;
            foreach (JobData job in store.AllJobs)
            {
                int slot = StateSlot(job.State);
                if (slot < 0) continue;
                byState[slot]++;
                active++;
            }

            s[0] = Squash(orch.SimTime, HorizonScale);
            s[1] = Squash(active, WipScale);
            if (active > 0)
                for (int k = 0; k < 5; k++) s[2 + k] = (float)byState[k] / active;

            int machineCount = layout != null && layout.Machines != null ? layout.Machines.Count : 0;
            if (machineCount > 0)
            {
                int busy = 0, down = 0;
                float load = 0f;
                foreach (PhysicalMachine m in layout.Machines)
                {
                    if (!m.IsIdle) busy++;
                    if (!m.IsAvailableForWork) down++;
                    load += store.GetMachineLoad(m.MachineId);
                }
                s[7] = (float)busy / machineCount;
                s[8] = (float)down / machineCount;
                s[10] = Squash(load / machineCount, TimeScale);
                s[12] = Mathf.Clamp01((float)machineCount / MaxMachines);
            }

            if (AGVPool.Instance != null && AGVPool.Instance.AllAGVs.Count > 0)
            {
                int activeAgvs = 0;
                foreach (AGVController agv in AGVPool.Instance.AllAGVs)
                    if (!agv.IsIdle) activeAgvs++;
                s[9] = (float)activeAgvs / AGVPool.Instance.AllAGVs.Count;
                if (machineCount > 0)
                    s[13] = Mathf.Clamp01((float)AGVPool.Instance.AllAGVs.Count / machineCount);
            }

            if (active > MaxJobs) s[11] = (float)(active - MaxJobs) / active;
            s[14] = Squash(store.DeferredJobIds.Count, CountScale);
            int options = req == null ? 0
                : req.Type == DecisionType.Routing ? (req.CandidateMachineIds?.Length ?? 0)
                : (req.QueuedJobIds?.Length ?? 0);
            s[15] = Squash(options, CountScale);
            return s;
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

            // [4] At least one machine is down (Failed or Repairing). Was "more than half of JobCount exited",
            //     which is meaningless once jobs arrive over time (JobCount only counts arrivals so far).
            if (FactoryLayoutManager.Instance != null)
            {
                foreach (PhysicalMachine m in FactoryLayoutManager.Instance.Machines)
                {
                    if (!m.IsAvailableForWork)
                    {
                        flags[4] = 1f;
                        break;
                    }
                }
            }

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