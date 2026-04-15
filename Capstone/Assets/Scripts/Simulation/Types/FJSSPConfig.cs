using System.Collections.Generic;
using Assets.Scripts.Simulation.Machines;

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

        /// @brief Per-machine-type normal distribution parameters (mu, sigma) for processing time sampling.
        ///
        /// @details When populated, @c FJSSPJobGenerator will sample processing times from
        /// N(mu, sigma) for each type. Types not present in this dictionary fall back to
        /// a uniform sample within [MinProcTime, MaxProcTime].
        public Dictionary<MachineType, (float mu, float sigma)> ProcTimeParams
            = new Dictionary<MachineType, (float mu, float sigma)>();

        /// @brief Total number of machines in this configuration.
        public int TotalMachines => MachineTypeLayout?.Length ?? 0;
    }
}