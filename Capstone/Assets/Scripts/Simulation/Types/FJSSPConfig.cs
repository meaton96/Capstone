using Assets.Scripts.Simulation.Machines;

namespace Assets.Scripts.Simulation.Types
{
    public class FJSSPConfig
    {
        /// @brief Human-readable label used in logs and CSV output (e.g. "20j_15m_baseline").
        public string Name = "unnamed";
        public int Seed = 42;
        public int JobCount;                    // 50
        public int MachinesPerType;             // 3 → total 15
        public MachineType[] MachineTypeLayout; // the ordered list of all 15 machines
        public float MinProcTime;               // e.g. 15f
        public float MaxProcTime;               // e.g. 90f
        public int MinOpsPerJob;                // e.g. 3
        public int MaxOpsPerJob;                // e.g. 7
        public float MaxArrivalTime;            // spread dynamic arrivals across this window

        /// @brief Total number of machines in this configuration.
        public int TotalMachines => MachineTypeLayout?.Length ?? 0;
    }
}