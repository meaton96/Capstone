using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation.Types
{
    /// @brief Legacy configuration class for Flexible Job Shop Scheduling Problem instances.
    ///
    /// @details Retained for backward compatibility with older batch configs and historical data.
    ///          Superseded by @c FJSSPConfig, which adds machine flexibility probability and
    ///          stochastic disruption support. All new code should use @c FJSSPConfig.
    ///
    /// @deprecated Use @c FJSSPConfig instead. This class lacks machine flexibility and
    ///             stochastic disruption parameters.
    /// @see FJSSPConfig
    public class FJSSPConfigOld
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
        public float MaxArrivalTime;

        // ── AGV fleet ──

        /// @brief Number of AGVs in the fleet for material transport.
        public int AGVCount;

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

        // ── Computed properties ──

        /// @brief Total number of machines in this configuration.
        /// @details Equals the length of @c MachineTypeLayout, or 0 if layout is null.
        public int TotalMachines => MachineTypeLayout?.Length ?? 0;
    }
}
