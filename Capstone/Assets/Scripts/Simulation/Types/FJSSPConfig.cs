using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Stochastic;

namespace Assets.Scripts.Simulation.Types
{
    public class FJSSPConfig
    {
        /// @brief Human-readable label used in logs and CSV output (e.g. "20j_15m_baseline").
        public string Name = "unnamed";
        public int Seed = 42;
        public int JobCount;
        public int MachinesPerType;
        public MachineType[] MachineTypeLayout;
        public float MinProcTime;               // fallback only — used when a type has no ProcTimeParams entry
        public float MaxProcTime;               // fallback only
        public int MinOpsPerJob;
        public int MaxOpsPerJob;
        public float MaxArrivalTime;
        public int AGVCount;

        public DispatchingRule dispatchingRule = DispatchingRule.SRT_SRWT;

        /// @brief Per-machine-type normal distribution parameters (mu, sigma) for processing time sampling.
        ///
        /// @details When populated, FJSSPJobGenerator will sample processing times from
        /// N(mu, sigma) for each type. Types not present fall back to Uniform[MinProcTime, MaxProcTime].
        public Dictionary<MachineType, (float mu, float sigma)> ProcTimeParams
            = new Dictionary<MachineType, (float mu, float sigma)>();

        /// @brief Optional stochastic disruption parameters.
        /// Null = fully deterministic episode. Non-null activates StochasticEventManager
        /// for the subset of disruption types flagged in the config.
        /// Deserialised from the optional "stochastic" block in batch JSON configs.
        public StochasticConfig Stochastic = null;

        /// @brief Total number of machines in this configuration.
        public int TotalMachines => MachineTypeLayout?.Length ?? 0;

        /// @brief Returns a deep clone with a new seed. Used by HeadlessBatchRunner
        /// per-repeat to vary job generation while keeping all other parameters identical.
        public FJSSPConfig CloneWithSeed(int newSeed)
        {
            return new FJSSPConfig
            {
                Name           = Name,
                Seed           = newSeed,
                JobCount       = JobCount,
                MachinesPerType = MachinesPerType,
                MachineTypeLayout = (MachineType[])MachineTypeLayout.Clone(),
                MinProcTime    = MinProcTime,
                MaxProcTime    = MaxProcTime,
                MinOpsPerJob   = MinOpsPerJob,
                MaxOpsPerJob   = MaxOpsPerJob,
                MaxArrivalTime = MaxArrivalTime,
                AGVCount       = AGVCount,
                dispatchingRule = dispatchingRule,
                ProcTimeParams = new Dictionary<MachineType, (float mu, float sigma)>(ProcTimeParams),
                Stochastic     = Stochastic,   // intentionally shared — stochastic params don't vary per-repeat
            };
        }
    }
}
