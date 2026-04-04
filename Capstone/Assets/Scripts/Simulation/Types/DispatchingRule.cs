namespace Assets.Scripts.Simulation.Types
{
    public enum DispatchingRule
    {
        SPT_SMPT,   ///< Shortest Processing Time — primary sort by @ref Operation.Duration ascending.
        SPT_SRWT,   ///< Shortest Processing Time with Shortest Remaining Work Time secondary metric.
        LPT_MMUR,   ///< Longest Processing Time — primary sort by @ref Operation.Duration descending.
        LPT_SMPT,   ///< Longest Processing Time with Smallest Most Urgent Remaining Time secondary metric.
        SRT_SRWT,   ///< Shortest Remaining Time — primary sort by total remaining work ascending.
        SRT_SMPT,   ///< Shortest Remaining Time with Smallest Most Urgent Remaining Time secondary metric.
        LRT_MMUR,   ///< Longest Remaining Time — primary sort by total remaining work descending. Equivalent to @c most_work_remaining in job_shop_lib.
        SDT_SRWT,   ///< Smallest Due Time — FIFO based on arrival order. Equivalent to @c first_come_first_served in job_shop_lib.
    }
}