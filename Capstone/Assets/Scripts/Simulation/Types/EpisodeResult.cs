namespace Assets.Scripts.Simulation.Types
{
    /// @brief Summary statistics produced when an episode ends.
    public class EpisodeResult
    {
        public string InstanceName;
        public string RuleName;
        public double Makespan;
        public double OptimalMakespan;
        public int TotalJobs;
        public int TotalOperations;
        public int CompletedJobs;
        public int DecisionPoints;
        public double TotalReward;
        public int[] PerMachineDecisions;
        public int EpisodeFailures = 0;
        public float TotalRepairTime = 0f;


        public int AGVCount;

        /// @brief Percentage deviation of achieved makespan from the known optimum.
        public double OptimalityGap => OptimalMakespan > 0
            ? (Makespan - OptimalMakespan) / OptimalMakespan * 100.0
            : 0;
    }
}
