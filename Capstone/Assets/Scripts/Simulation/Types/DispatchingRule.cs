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
    ///          - FIFO_SRWT: FIFO/FCFS — arrival order, oldest job first.
    ///          - MMUR (Minimum Machine Utilization): routes to the candidate machine with the
    ///            lowest cumulative utilization ratio so far this episode — a longer-horizon,
    ///            workload-history signal, distinct from SRWT's instantaneous queued workload.
    ///          - ECT (Earliest Completion Time): routes to the candidate minimising queued work
    ///            (incl. the in-process remainder) + this job's processing time there.
    ///          - TECT: ECT with AGV travel: max(zone-graph travel time, queued work) + processing time.
    ///          - PTWINQ: job priority = processing time + least queued work among the machines
    ///            eligible for the job's next operation (Holthaus &amp; Rajendran PT+WINQ).
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

        /// @brief FIFO/FCFS — arrival order, oldest job first.
        /// @details Equivalent to @c first_come_first_served in job_shop_lib. Was previously
        ///          implemented backwards (picked newest arrival) despite this doc comment and
        ///          the DispatchingEngine inline comment both saying "longest waiting" -- fixed.
        FIFO_SRWT,

        /// @brief Random — unweighted random selection of queued jobs.
        /// @details Useful for exploration or baseline comparison.
        Random,

        // ── Full job-rule x machine-rule catalog (2026-09-26) ─────────────────────────────────
        // Baseline-only: the RL action space is still the 9 rules above (DispatchingEngine.ActionToRule).
        // Appended after Random so serialized int values of the members above are unchanged.
        // Every name is JOB_MACHINE; DispatchingEngine splits it into its two halves.
        //   Job:     SPT, LPT, SRT, LRT, FIFO, PTWINQ (processing time + least work in the next op's queue)
        //   Machine: SMPT, SRWT, MMUR, ECT (least queued work + own processing time),
        //            TECT (ECT with travel: max(travel, queued work) + own processing time)
        SPT_MMUR, SPT_ECT, SPT_TECT,
        LPT_SRWT, LPT_ECT, LPT_TECT,
        SRT_MMUR, SRT_ECT, SRT_TECT,
        LRT_SMPT, LRT_SRWT, LRT_ECT, LRT_TECT,
        FIFO_SMPT, FIFO_MMUR, FIFO_ECT, FIFO_TECT,
        PTWINQ_SMPT, PTWINQ_SRWT, PTWINQ_MMUR, PTWINQ_ECT, PTWINQ_TECT,
    }
}
