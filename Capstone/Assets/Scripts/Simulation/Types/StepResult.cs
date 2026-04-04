namespace Assets.Scripts.Simulation.Types
{
    /// @brief Result returned by @c SimulationBridge.Step() after applying a dispatching rule.
    public struct StepResult
    {
        /// @brief Reward signal for the agent based on elapsed makespan delta.
        public float Reward;

        /// @brief True when the episode has ended (all jobs complete).
        public bool Done;

        /// @brief The next decision context, if one is immediately available.
        public DecisionRequest NextDecision;

        /// @brief Makespan at the time this step was resolved.
        public double CurrentMakespan;

        /// @brief Total operations completed across all jobs at this step.
        public int OperationsCompleted;
    }
}