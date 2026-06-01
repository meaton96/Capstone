using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Stochastic;

namespace Assets.Scripts.Simulation.Types
{
    /// @brief Configuration parameters for a Flexible Job Shop Scheduling Problem (FJSSP) instance.
    ///
    /// @details Defines all static parameters governing job shop generation: machine topology,
    ///          processing time ranges, operation counts, AGV fleet size, and stochastic
    ///          disruption settings. Used by @c ConfigLoader to parse from JSON and by
    ///          @c FJSSPJobGenerator to create problem instances.
    ///
    /// @see ConfigLoader
    /// @see FJSSPJobGenerator
    /// @see StochasticConfig
    public class FJSSPConfig
    {
        // ── Identity ──

        /// @brief Human-readable label used in logs and CSV output (e.g. "20j_15m_baseline").
        public string Name = "unnamed";

        // ── Random seed ──

        /// @brief Seed for deterministic job shop generation. Default: 42.
        public int Seed = 42;

        // ── Problem scale ──

        /// @brief Number of jobs to generate per episode.
        public int JobCount;

        /// @brief Number of machines of each type in the layout.
        public int MachinesPerType;

        /// @brief Ordered array of machine types defining the floor layout.
        /// @details Length equals @c TotalMachines. Each entry specifies the type of the
        ///          corresponding machine on the floor.
        public MachineType[] MachineTypeLayout;

        // ── Processing time parameters ──

        /// @brief Minimum processing time (uniform fallback).
        /// @details Used only when a machine type has no entry in @c ProcTimeParams.
        public float MinProcTime;

        /// @brief Maximum processing time (uniform fallback).
        /// @details Used only when a machine type has no entry in @c ProcTimeParams.
        public float MaxProcTime;

        // ── Operation count range ──

        /// @brief Minimum number of operations per job.
        public int MinOpsPerJob;

        /// @brief Maximum number of operations per job.
        public int MaxOpsPerJob;

        /// @brief Maximum arrival time for initial job batch (time window).
        /// @details Controls how spread out initial job releases are.
       // public float MaxArrivalTime;

        // ── AGV fleet ──

        /// @brief Number of AGVs in the fleet for material transport.
        public int AGVCount;

        // ── Flexibility ──

        /// @brief Probability [0,1] that a machine gains each non-primary type as a
        /// secondary capability during floor construction.
        /// @details 0 = fully typed (default, backward-compatible).
        ///          1 = fully flexible (every machine processes every operation type).
        public float MachineFlexibilityProbability = 0f;

        // ── Scheduling policy ──

        /// @brief Default dispatching rule applied when no agent policy is active.
        public DispatchingRule dispatchingRule = DispatchingRule.SRT_SRWT;

        // ── Processing time distributions ──

        /// @brief Per-machine-type normal distribution parameters (mu, sigma) for processing time sampling.
        ///
        /// @details When populated, @c FJSSPJobGenerator will sample processing times from
        /// N(mu, sigma) for each type. Types not present in this dictionary fall back to
        /// a uniform sample within [MinProcTime, MaxProcTime].
        public Dictionary<MachineType, (float mu, float sigma)> ProcTimeParams
            = new Dictionary<MachineType, (float mu, float sigma)>();

        // ── Stochastic disruptions ──

        /// @brief Optional stochastic disruption parameters.
        /// @details Null = fully deterministic episode. Non-null activates @c StochasticEventManager
        ///          for the subset of disruption types flagged in the config. Deserialised from
        ///          the optional "stochastic" block in batch JSON configs.
        public StochasticConfig Stochastic = null;

        // ── Computed properties ──

        /// @brief Total number of machines in this configuration.
        /// @details Equals the length of @c MachineTypeLayout, or 0 if layout is null.
        public int TotalMachines => MachineTypeLayout?.Length ?? 0;

        // ── Methods ──

        /// @brief Returns a deep clone with a new seed.
        ///
        /// @param newSeed  Random seed for the cloned configuration.
        /// @returns         Deep clone with identical parameters except for the updated seed.
        ///
        /// @details Used by @c HeadlessBatchRunner to generate varied instances across
        ///          repeated runs while keeping all other parameters identical.
        ///          Note: @c Stochastic is shared by reference (not deep-cloned).
        public FJSSPConfig CloneWithSeed(int newSeed)
        {
            return new FJSSPConfig
            {
                Name = Name,
                Seed = newSeed,
                JobCount = JobCount,
                MachinesPerType = MachinesPerType,
                MachineTypeLayout = (MachineType[])MachineTypeLayout.Clone(),
                MinProcTime = MinProcTime,
                MaxProcTime = MaxProcTime,
                MinOpsPerJob = MinOpsPerJob,
                MaxOpsPerJob = MaxOpsPerJob,
                AGVCount = AGVCount,
                dispatchingRule = dispatchingRule,
                ProcTimeParams = new Dictionary<MachineType, (float mu, float sigma)>(ProcTimeParams),
                Stochastic = Stochastic,
                MachineFlexibilityProbability = MachineFlexibilityProbability,
            };
        }
    }
}
