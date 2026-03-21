using System;
namespace Assets.Scripts.Scheduling.Data
{
    /// @brief Direct deserialization target for the Taillard JSON benchmark format.
    ///
    /// @details Field names match the JSON keys exactly to allow zero-configuration
    /// deserialization via Unity's @c JsonUtility or @c System.Text.Json.
    /// After deserialization, @ref JobCount and @ref MachineCount are derived from the
    /// matrix dimensions, and @ref GetJobOperations provides the typed operation pairs
    /// consumed by @ref DESSimulator.LoadInstance.
    [Serializable]
    public class TaillardInstance
    {
        /// @brief Benchmark instance identifier, e.g. @c "ta001".
        public string name;

        /// @brief Processing time matrix indexed as @c [job][operation].
        /// @details Each entry is the number of time units required for that operation.
        /// Must be rectangular (all rows the same length) for @ref MachineCount to be valid.
        public int[][] duration_matrix;

        /// @brief Machine assignment matrix indexed as @c [job][operation].
        /// @details Each entry is the zero-based index of the machine that must process
        /// that operation. Dimensions must match @ref duration_matrix exactly.
        public int[][] machines_matrix;

        /// @brief Known bounds and bibliographic reference for this instance.
        /// @details May be @c null if the JSON source omits the metadata block.
        public TaillardMetadata metadata;

        /// @brief Number of jobs in the instance, derived from the row count of @ref duration_matrix.
        public int JobCount => duration_matrix.Length;

        /// @brief Number of machines in the instance, derived from the column count of @ref duration_matrix.
        /// @details Taillard benchmark instances are always square (jobs × machines = machines × machines),
        /// so @c duration_matrix[0].Length is a reliable column count. Behaviour is undefined
        /// if @ref duration_matrix is empty or jagged.
        public int MachineCount => duration_matrix[0].Length;

        /// @brief Returns the ordered operation sequence for a single job as typed value tuples.
        ///
        /// @details Zips @ref machines_matrix and @ref duration_matrix for the given job index
        /// into an array of @c (machine, duration) pairs in operation order. This is the
        /// primary data accessor used by @ref DESSimulator.LoadInstance when constructing
        /// @ref Operation objects.
        ///
        /// @param jobIndex Zero-based index of the job to query. Must be in the range
        /// @c [0, JobCount).
        ///
        /// @returns An array of @c (int machine, int duration) tuples, one per operation,
        /// in the order they must be executed.
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

    /// @brief Known optimality bounds and bibliographic metadata for a @ref TaillardInstance.
    ///
    /// @details Populated from the JSON metadata block alongside the problem matrices.
    /// Used for post-simulation validation: comparing @ref DESSimulator.Makespan against
    /// @ref optimum or @ref upper_bound confirms correctness of the dispatching rules.
    /// All fields default to @c 0 if absent from the JSON source.
    [Serializable]
    public class TaillardMetadata
    {
        /// @brief Proven optimal makespan for this instance, or @c 0 if unknown.
        public int optimum;

        /// @brief Best known upper bound on the makespan, i.e. the lowest makespan
        /// achieved by any published algorithm for this instance.
        public int upper_bound;

        /// @brief Theoretical lower bound on the makespan, below which no valid
        /// schedule can exist.
        public int lower_bound;

        /// @brief Bibliographic reference identifying the source of the bound values,
        /// e.g. @c "Taillard 1993".
        public string reference;
    }
}