using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation.Jobs;

namespace Assets.Scripts.Simulation
{
    public class ObservationBuilder
    {
        private SimulationBridge _bridge;

        public ObservationBuilder(SimulationBridge bridge)
        {
            _bridge = bridge;
        }

        // Main entry point called by the SchedulingAgent
        public float[] BuildCompleteSnapshot(DecisionRequest currentDecision)
        {
            // 1. Generate individual streams
            float[] spatialGrid = BuildSpatialOccupancyGrid();
            float[] schedulingMatrix = BuildSchedulingMatrix();
            float[] globalScalars = BuildGlobalScalars();
            float[] distanceMatrix = BuildDistanceMatrix();
            float[] eventFlags = BuildEventFlags(currentDecision);

            // 2. Apply Domain Randomization (Noise & Dropout)
            ApplyDomainRandomization(spatialGrid, schedulingMatrix, globalScalars);

            // 3. Serialize into a flat floating-point array for ML-Agents
            return FlattenStreams(spatialGrid, schedulingMatrix, globalScalars, distanceMatrix, eventFlags);
        }

        private float[] BuildSpatialOccupancyGrid()
        {
            // TODO: Create a 64x64x3 array representing the grid.
            // Channel 0: Machine locations/status
            // Channel 1: Job physical locations
            // Channel 2: AGV locations and paths
            float[] grid = new float[64 * 64 * 3];
            return grid;
        }

        private float[] BuildSchedulingMatrix()
        {
            // TODO: Extract dimensions n (jobs) and m (machines) from _bridge.CurrentConfig
            // Populate processing times and completion statuses.
            return new float[0]; // Placeholder
        }

        private float[] BuildGlobalScalars()
        {
            float[] scalars = new float[10];
            scalars[0] = (float)_bridge.SimTime;
            scalars[1] = _bridge.Jobs.CountInState(JobState.Exited) / (float)_bridge.Jobs.JobCount;
            // TODO: Add utilization, queue pressure, etc.
            return scalars;
        }

        private float[] BuildDistanceMatrix()
        {
            return new float[64]; // TODO: Pull from TrafficZoneManager or FactoryLayoutManager
        }

        private float[] BuildEventFlags(DecisionRequest req)
        {
            float[] flags = new float[6];
            flags[0] = req.Type == DecisionType.Dispatch ? 1f : 0f;
            flags[1] = req.Type == DecisionType.Routing ? 1f : 0f;
            // TODO: Add other triggers (machine breakdown, AGV battery low, etc.)
            return flags;
        }

        private void ApplyDomainRandomization(float[] spatial, float[] scheduling, float[] scalars)
        {
            // TODO: Inject Gaussian noise and simulate sensor dropout
        }

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
    }
}