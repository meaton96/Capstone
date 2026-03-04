using System;

namespace Scheduling.Data
{
    /// <summary>
    /// Direct deserialization target for the Taillard JSON format.
    /// </summary>
    [Serializable]
    public class TaillardInstance
    {
        public string name;
        public int[][] duration_matrix;   // [job][operation] → processing time
        public int[][] machines_matrix;    // [job][operation] → machine index
        public TaillardMetadata metadata;

        public int JobCount => duration_matrix.Length;
        public int MachineCount => duration_matrix[0].Length; // square for Taillard

        /// <summary>
        /// Get the ordered operations for a specific job.
        /// Returns (machineId, processingTime) pairs in operation order.
        /// </summary>
        public (int machine, int duration)[] GetJobOperations(int jobIndex)
        {
            int opCount = machines_matrix[jobIndex].Length;
            var ops = new (int, int)[opCount];
            for (int i = 0; i < opCount; i++)
            {
                ops[i] = (machines_matrix[jobIndex][i], duration_matrix[jobIndex][i]);
            }
            return ops;
        }
    }

    [Serializable]
    public class TaillardMetadata
    {
        public int optimum;
        public int upper_bound;
        public int lower_bound;
        public string reference;
    }
}
