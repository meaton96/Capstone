namespace Assets.Scripts.Simulation.Types
{
    /// @brief Result returned by @c SimulationBridge.Step() after applying a dispatching rule.
    ///
    /// @details Encapsulates the reward signal, episode termination flag, and the next
    ///          decision context for the agent. Populated by the simulation after each
    ///          discrete event (machine completion or AGV delivery).
    ///
    /// @see SimulationBridge.Step
    /// @see DecisionRequest
    public struct StepResult
    {
        /// @brief Reward signal for the agent, computed from the elapsed makespan delta
        ///        since the previous step. Negative or zero values are typical (penalty for elapsed time).
        public float Reward;

        /// @brief True when the episode has ended (all jobs completed or no valid moves remain).
        public bool Done;

        /// @brief The next decision context if a new decision is available, or null if the episode is done.
        /// @details Populated when @c Done is false and a dispatch or routing decision is immediately required.
        public DecisionRequest NextDecision;

        /// @brief Current makespan (completion time of the last finished operation) at the time this step was resolved.
        public double CurrentMakespan;

        /// @brief Total number of operations completed across all jobs at this step.
        /// @details Increments monotonically from 0 to @c FJSSPConfig.JobCount × average operations per job.
        public int OperationsCompleted;
    }
}
