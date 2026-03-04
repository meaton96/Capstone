using System;
using System.Collections.Generic;
using System.Linq;

namespace Scheduling.Core
{
    /// <summary>
    /// Composite Primary Dispatching Rules.
    ///
    /// For cross-validation against job_shop_lib, tiebreaking is by lowest
    /// job_id (matching job_shop_lib's default internal ordering).
    /// This ensures identical makespans when the primary sort key is equal.
    /// </summary>
    public enum DispatchingRule
    {
        SPT_SMPT,   // Shortest Processing Time
        SPT_SRWT,
        LPT_MMUR,   // Longest Processing Time
        LPT_SMPT,
        SRT_SRWT,   // Shortest Remaining Time
        SRT_SMPT,
        LRT_MMUR,   // Longest Remaining Time (= most_work_remaining)
        SDT_SRWT,   // FIFO (= first_come_first_served)
        MOR,        // Most Operations Remaining
    }

    public static class DispatchingRules
    {
        /// <summary>
        /// Given a machine's waiting queue, select the next operation to process.
        /// Tiebreaker: lowest job_id (matches job_shop_lib default).
        /// </summary>
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
                                        .First(),// "Most Work Remaining" in job_shop_lib
                DispatchingRule.MOR => waitingOps
                                        .OrderByDescending(op =>
                                        {
                                            var job = allJobs[op.JobId];
                                            return job.Operations.Length - op.OperationIndex;
                                        })
                                        .ThenBy(op => op.JobId)
                                        .First(),// "Most Operations Remaining" in job_shop_lib
                DispatchingRule.SDT_SRWT => waitingOps[0],// FIFO: return first in queue (arrival order)
                _ => waitingOps[0],
            };

        }

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
