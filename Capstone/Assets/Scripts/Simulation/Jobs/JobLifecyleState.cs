namespace Assets.Scripts.Simulation.Jobs
{
    /// @brief Lifecycle states a job passes through from creation to completion.
    public enum JobLifecycleState
    {
        NotStarted,
        Queued,
        Processing,
        WaitingForTransport,
        InTransit,
        Complete,
    }
}