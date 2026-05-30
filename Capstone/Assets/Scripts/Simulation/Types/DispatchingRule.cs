namespace Assets.Scripts.Simulation.Types
{
    /// @brief Predefined dispatching rules for prioritising jobs in the dispatching engine.
    ///
    /// @details Each rule defines a sorting criterion used by @c DispatchingEngine to order
    ///          candidate jobs when an idle machine becomes available. Multi-criteria rules
    ///          use a primary sort key with a secondary tiebreaker.
    ///          - SPT (Shortest Processing Time): favours quick completions, reducing WIP.
    ///          - LPT (Longest Processing Time): defers long operations, useful for throughput.
    ///          - SRT/LRT (Shortest/Longest Remaining Time): uses cumulative remaining work.
    ///          - SDT (Smallest Due Time): arrival-order FIFO, equivalent to FCFS.
    ///          - Random: unweighted random selection (useful for exploration).
    public enum DispatchingRule
    {
        /// @brief Shortest Processing Time — primary sort by @c Operation.Duration ascending.
        /// @details Ties broken by Smallest Most Urgent Remaining Time.
        SPT_SMPT,

        /// @brief Shortest Processing Time with Shortest Remaining Work Time secondary metric.
        /// @details Primary: duration ascending. Secondary: remaining work ascending.
        SPT_SRWT,

        /// @brief Longest Processing Time — primary sort by @c Operation.Duration descending.
        /// @details Ties broken by Smallest Most Urgent Remaining Time.
        LPT_MMUR,

        /// @brief Longest Processing Time with Smallest Most Urgent Remaining Time secondary metric.
        /// @details Primary: duration descending. Secondary: urgency ascending.
        LPT_SMPT,

        /// @brief Shortest Remaining Time — primary sort by total remaining work ascending.
        /// @details Favors jobs with the least total work remaining across all operations.
        SRT_SRWT,

        /// @brief Shortest Remaining Time with Smallest Most Urgent Remaining Time secondary metric.
        /// @details Primary: remaining work ascending. Secondary: urgency ascending.
        SRT_SMPT,

        /// @brief Longest Remaining Time — primary sort by total remaining work descending.
        /// @details Equivalent to @c most_work_remaining in job_shop_lib. Favors heavy jobs.
        LRT_MMUR,

        /// @brief Smallest Due Time — FIFO based on arrival order.
        /// @details Equivalent to @c first_come_first_served in job_shop_lib.
        SDT_SRWT,

        /// @brief Random — unweighted random selection of queued jobs.
        /// @details Useful for exploration or baseline comparison.
        Random,
    }
}
