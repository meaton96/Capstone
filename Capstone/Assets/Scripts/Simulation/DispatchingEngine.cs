using System;
using System.Collections.Generic;
using System.Linq;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation
{
    /// <summary>
    /// DispatchingEngine provides static methods for evaluating and applying dispatching rules
    /// in the Flexible Job Shop Scheduling Problem (FJSSP) simulation.
    /// It maps action indices to dispatching rules, selects jobs for machines, selects machines
    /// for jobs, and computes job metrics such as remaining work.
    /// </summary>
    public static class DispatchingEngine
    {
        /// <summary>
        /// Array mapping action indices to their corresponding dispatching rules.
        /// Each rule defines a specific priority heuristic for job selection.
        /// Supported dispatching rules in order: SPT_SMPT, SPT_SRWT, LPT_MMUR, LPT_SMPT,
        /// SRT_SRWT, SRT_SMPT, LRT_MMUR, SDT_SRWT, and Random.
        /// </summary>
        private static readonly DispatchingRule[] ActionToRule = new DispatchingRule[]
        {
            DispatchingRule.SPT_SMPT,   // Shortest Processing Time - Machine
            DispatchingRule.SPT_SRWT,   // Shortest Processing Time - Server
            DispatchingRule.LPT_MMUR,   // Longest Processing Time - Machine
            DispatchingRule.LPT_SMPT,   // Longest Processing Time - Server
            DispatchingRule.SRT_SRWT,   // Shortest Remaining Time - Server
            DispatchingRule.SRT_SMPT,   // Shortest Remaining Time - Machine
            DispatchingRule.LRT_MMUR,   // Longest Remaining Time - Machine
            DispatchingRule.SDT_SRWT,   // Shortest Delay Time - Server
            DispatchingRule.Random
        };

        /// <summary>
        /// Gets the total number of dispatching rules available in the engine.
        /// </summary>
        public static int ActionCount => ActionToRule.Length;

        /// <summary>
        /// Retrieves the dispatching rule associated with the given index.
        /// </summary>
        /// <param name="index">Zero-based index of the rule.</param>
        /// <returns>The dispatching rule at the specified index.</returns>
        public static DispatchingRule RuleForIndex(int index) => ActionToRule[index];

        /// <summary>
        /// Retrieves the index of the given dispatching rule within the registered rule array.
        /// </summary>
        /// <param name="rule">The dispatching rule to look up.</param>
        /// <returns>The zero-based index of the rule, or -1 if not found.</returns>
        public static int IndexForRule(DispatchingRule rule) => Array.IndexOf(ActionToRule, rule);

        /// <summary>
        /// Selects the best job for a given machine based on the dispatching rule specified by actionIndex.
        /// Uses the rule to evaluate candidate jobs and returns the job ID that best satisfies the priority criterion.
        /// </summary>
        /// <param name="actionIndex">Index into ActionToRule to determine which dispatching rule to apply.</param>
        /// <param name="machineId">ID of the machine that needs a job assigned.</param>
        /// <param name="jobs">Reference to the JobStore containing all job data.</param>
        /// <param name="simTime">Current simulation time, used for time-based rules such as SDT.</param>
        /// <returns>The selected job ID, or -1 if no dispatchable jobs are available for the machine.</returns>
        public static int SelectJob(int actionIndex, int machineId, JobStore jobs, double simTime)
        {
            DispatchingRule rule = ActionToRule[actionIndex];

            // Random rule requires re-sampling a specific non-random rule at decision time
            if (rule == DispatchingRule.Random)
                rule = ActionToRule[UnityEngine.Random.Range(0, ActionToRule.Length - 1)];

            List<int> queue = jobs.GetDispatchableJobs(machineId);
            if (queue.Count == 0) return -1;
            if (queue.Count == 1) return queue[0];

            return rule switch
            {
                // Shortest Processing Time rules — minimize processing time on the target machine
                DispatchingRule.SPT_SMPT or DispatchingRule.SPT_SRWT
                    => ArgMin(queue, id => jobs.Get(id).GetProcessingTime(machineId)),
                // Longest Processing Time rules — maximize processing time on the target machine
                DispatchingRule.LPT_MMUR or DispatchingRule.LPT_SMPT
                    => ArgMax(queue, id => jobs.Get(id).GetProcessingTime(machineId)),
                // Shortest Remaining Time rules — minimize total remaining work across all operations
                DispatchingRule.SRT_SRWT or DispatchingRule.SRT_SMPT
                    => ArgMin(queue, id => GetRemainingWork(id, jobs)),
                // Longest Remaining Time rule — maximize total remaining work
                DispatchingRule.LRT_MMUR
                    => ArgMax(queue, id => GetRemainingWork(id, jobs)),
                // Shortest Delay Time rule — prioritize jobs that have been waiting the longest
                DispatchingRule.SDT_SRWT
                    => ArgMin(queue, id => (float)(simTime - jobs.Get(id).ArrivalTime)),
                // Fallback — select a random job from the queue
                _ => queue[UnityEngine.Random.Range(0, queue.Count)]
            };
        }

        /// <summary>
        /// Selects the best machine from candidate machines based on the dispatching rule specified by actionIndex.
        /// Evaluates candidates using metrics from the DecisionRequest (e.g., job times, queue lengths).
        /// </summary>
        /// <param name="actionIndex">Index into ActionToRule to determine which dispatching rule to apply.</param>
        /// <param name="req">The decision request containing candidate machine IDs and their associated metrics.</param>
        /// <returns>The selected machine ID from the candidate set.</returns>
        public static int SelectMachine(int actionIndex, DecisionRequest req)
        {
            DispatchingRule rule = ActionToRule[actionIndex];

            // Random rule requires re-sampling a specific non-random rule at decision time
            if (rule == DispatchingRule.Random)
                rule = ActionToRule[UnityEngine.Random.Range(0, ActionToRule.Length - 1)];

            int[] candidates = req.CandidateMachineIds;
            if (candidates.Length == 1) return candidates[0];

            return rule switch
            {
                // Machine-focused rules — select machine with minimum job processing time
                DispatchingRule.SPT_SMPT or DispatchingRule.LPT_SMPT or DispatchingRule.SRT_SMPT
                    => candidates[ArgMinIdx(req.CandidateJobTimes)],
                // Server-focused rules — select machine with minimum queue length
                DispatchingRule.SPT_SRWT or DispatchingRule.SRT_SRWT or DispatchingRule.SDT_SRWT
                    => candidates[ArgMinIdx(req.CandidateQueueLengths)],
                // Machine/Server rules for longest — select machine with maximum queue length
                DispatchingRule.LPT_MMUR or DispatchingRule.LRT_MMUR
                    => candidates[ArgMaxIdx(req.CandidateQueueLengths)],
                // Fallback — select a random machine from candidates
                _ => candidates[UnityEngine.Random.Range(0, candidates.Length)]
            };
        }

        /// <summary>
        /// Computes the total remaining work for a job, defined as the sum of minimum processing times
        /// across all remaining operations (from the current operation to the end).
        /// For each remaining operation, uses the minimum processing time among eligible machines
        /// as the estimated cost for that operation.
        /// </summary>
        /// <param name="jobId">ID of the job to evaluate.</param>
        /// <param name="jobs">Reference to the JobStore containing all job data.</param>
        /// <returns>Total remaining work as a floating-point sum of minimum processing times for all uncompleted operations.</returns>
        public static float GetRemainingWork(int jobId, JobStore jobs)
        {
            JobData j = jobs.Get(jobId);
            if (j == null) return 0f;
            float total = 0f;
            // Sum minimum processing time for each remaining operation
            for (int o = j.CurrentOpIndex; o < j.TotalOperations; o++)
                total += j.EligibleMachinesPerOp[o].Values.Min();
            return total;
        }

        /// <summary>
        /// Finds the element in the list with the minimum score value.
        /// </summary>
        /// <param name="ids">List of integer IDs to evaluate.</param>
        /// <param name="score">Function that computes a float score for each ID.</param>
        /// <returns>The ID with the minimum score.</returns>
        private static int ArgMin(List<int> ids, Func<int, float> score)
        {
            int best = ids[0]; float bestS = float.MaxValue;
            foreach (int id in ids) { float s = score(id); if (s < bestS) { bestS = s; best = id; } }
            return best;
        }

        /// <summary>
        /// Finds the element in the list with the maximum score value.
        /// </summary>
        /// <param name="ids">List of integer IDs to evaluate.</param>
        /// <param name="score">Function that computes a float score for each ID.</param>
        /// <returns>The ID with the maximum score.</returns>
        private static int ArgMax(List<int> ids, Func<int, float> score)
        {
            int best = ids[0]; float bestS = float.MinValue;
            foreach (int id in ids) { float s = score(id); if (s > bestS) { bestS = s; best = id; } }
            return best;
        }

        /// <summary>
        /// Finds the index of the element with the minimum value in a float array.
        /// </summary>
        /// <param name="v">Array of float values.</param>
        /// <returns>Index of the element with the minimum value.</returns>
        private static int ArgMinIdx(float[] v)
        {
            int b = 0;
            for (int i = 1; i < v.Length; i++) if (v[i] < v[b]) b = i;
            return b;
        }

        /// <summary>
        /// Finds the index of the element with the maximum value in a float array.
        /// </summary>
        /// <param name="v">Array of float values.</param>
        /// <returns>Index of the element with the maximum value.</returns>
        private static int ArgMaxIdx(float[] v)
        {
            int b = 0;
            for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
            return b;
        }
    }
}