using System;
using System.Collections.Generic;
using System.Linq;
namespace Assets.Scripts.Scheduling.Core
{
    /// @brief Composite primary dispatching rules for machine queue selection.
    ///
    /// @details Tiebreaking is always by lowest job ID to match job_shop_lib's default
    /// internal ordering, ensuring identical makespans when the primary sort key is equal.
    /// Each enumerator name encodes its primary heuristic and secondary metric separated
    /// by an underscore, except @ref MOR which has no secondary metric.
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
        MOR,        ///< Most Operations Remaining — primary sort by count of unprocessed operations descending. Equivalent to @c most_operations_remaining in job_shop_lib.
    }

    public static class DispatchingRules
    {
        /// @brief Selects the next operation to process from a machine's waiting queue.
        ///
        /// @details Applies the given @ref DispatchingRule to rank all waiting operations,
        /// with ties broken by lowest job ID to match job_shop_lib's default ordering.
        /// Returns immediately if the queue contains zero or one entries, avoiding
        /// unnecessary sorting overhead. Rules that share a primary heuristic
        /// (e.g. @ref DispatchingRule.SPT_SMPT and @ref DispatchingRule.SPT_SRWT) are
        /// collapsed to the same LINQ ordering since secondary metrics are not yet
        /// distinguished at the operation level.
        ///
        /// @param waitingOps The list of operations currently queued on the machine,
        /// in arrival order. Must not be modified during selection.
        /// @param allJobs The full job array, used to compute remaining-work metrics
        /// for @ref DispatchingRule.SRT_SRWT, @ref DispatchingRule.SRT_SMPT,
        /// @ref DispatchingRule.LRT_MMUR, and @ref DispatchingRule.MOR.
        /// @param rule The dispatching rule that determines the selection priority.
        ///
        /// @returns The @ref Operation that should be started next, or @c null if
        /// @p waitingOps is null or empty.
        public static Operation SelectNext(
            List<Operation> waitingOps,
            Job[] allJobs,
            DispatchingRule rule)
        {
            if (waitingOps == null || waitingOps.Count == 0) return null;
            if (waitingOps.Count == 1) return waitingOps[0];
            return rule switch
            {
                DispatchingRule.SPT_SMPT or DispatchingRule.SPT_SRWT => waitingOps
                                        .OrderBy(op => op.Duration)
                                        .ThenBy(op => op.JobId)
                                        .First(),
                DispatchingRule.LPT_MMUR or DispatchingRule.LPT_SMPT => waitingOps
                                        .OrderByDescending(op => op.Duration)
                                        .ThenBy(op => op.JobId)
                                        .First(),
                DispatchingRule.SRT_SRWT or DispatchingRule.SRT_SMPT => waitingOps
                                        .OrderBy(op => RemainingWork(op, allJobs))
                                        .ThenBy(op => op.JobId)
                                        .First(),
                DispatchingRule.LRT_MMUR => waitingOps
                                        .OrderByDescending(op => RemainingWork(op, allJobs))
                                        .ThenBy(op => op.JobId)
                                        .First(),
                DispatchingRule.MOR => waitingOps
                                        .OrderByDescending(op =>
                                        {
                                            var job = allJobs[op.JobId];
                                            return job.Operations.Length - op.OperationIndex;
                                        })
                                        .ThenBy(op => op.JobId)
                                        .First(),
                DispatchingRule.SDT_SRWT => waitingOps[0],
                _ => waitingOps[0],
            };
        }

        /// @brief Computes the total remaining processing time for a job from a given operation onward.
        ///
        /// @details Sums @ref Operation.Duration for every operation at or after
        /// @p op's index in the job's operation sequence. Used as the sort key for
        /// @ref DispatchingRule.SRT_SRWT, @ref DispatchingRule.SRT_SMPT, and
        /// @ref DispatchingRule.LRT_MMUR.
        ///
        /// @param op The reference operation whose index marks the start of summation.
        /// @param allJobs The full job array used to retrieve the owning job's operation sequence.
        ///
        /// @returns The sum of durations for all remaining operations, including @p op itself.
        private static int RemainingWork(Operation op, Job[] allJobs)
        {
            var job = allJobs[op.JobId];
            int remaining = 0;
            for (int i = op.OperationIndex; i < job.Operations.Length; i++)
                remaining += job.Operations[i].Duration;
            return remaining;
        }
    }
}