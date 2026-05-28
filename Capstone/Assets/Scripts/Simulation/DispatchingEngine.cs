using System;
using System.Collections.Generic;
using System.Linq;
using Assets.Scripts.Simulation.Jobs;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation
{
    public static class DispatchingEngine
    {
        private static readonly DispatchingRule[] ActionToRule = new DispatchingRule[]
        {
            DispatchingRule.SPT_SMPT,
            DispatchingRule.SPT_SRWT,
            DispatchingRule.LPT_MMUR,
            DispatchingRule.LPT_SMPT,
            DispatchingRule.SRT_SRWT,
            DispatchingRule.SRT_SMPT,
            DispatchingRule.LRT_MMUR,
            DispatchingRule.SDT_SRWT,
            DispatchingRule.Random
        };

        public static int ActionCount => ActionToRule.Length;
        public static DispatchingRule RuleForIndex(int index) => ActionToRule[index];
        public static int IndexForRule(DispatchingRule rule) => Array.IndexOf(ActionToRule, rule);

        public static int SelectJob(int actionIndex, int machineId, JobStore jobs, double simTime)
        {
            DispatchingRule rule = ActionToRule[actionIndex];

            // Re-roll every decision point — don't just fall through to queue[0]
            if (rule == DispatchingRule.Random)
                rule = ActionToRule[UnityEngine.Random.Range(0, ActionToRule.Length - 1)];

            List<int> queue = jobs.GetDispatchableJobs(machineId);
            if (queue.Count == 0) return -1;
            if (queue.Count == 1) return queue[0];

            return rule switch
            {
                DispatchingRule.SPT_SMPT or DispatchingRule.SPT_SRWT
                    => ArgMin(queue, id => jobs.Get(id).GetProcessingTime(machineId)),
                DispatchingRule.LPT_MMUR or DispatchingRule.LPT_SMPT
                    => ArgMax(queue, id => jobs.Get(id).GetProcessingTime(machineId)),
                DispatchingRule.SRT_SRWT or DispatchingRule.SRT_SMPT
                    => ArgMin(queue, id => GetRemainingWork(id, jobs)),
                DispatchingRule.LRT_MMUR
                    => ArgMax(queue, id => GetRemainingWork(id, jobs)),
                DispatchingRule.SDT_SRWT
                    => ArgMin(queue, id => (float)(simTime - jobs.Get(id).ArrivalTime)),
                _ => queue[UnityEngine.Random.Range(0, queue.Count)]
            };
        }

        public static int SelectMachine(int actionIndex, DecisionRequest req)
        {
            DispatchingRule rule = ActionToRule[actionIndex];

            // Re-roll every decision point
            if (rule == DispatchingRule.Random)
                rule = ActionToRule[UnityEngine.Random.Range(0, ActionToRule.Length - 1)];

            int[] candidates = req.CandidateMachineIds;
            if (candidates.Length == 1) return candidates[0];

            return rule switch
            {
                DispatchingRule.SPT_SMPT or DispatchingRule.LPT_SMPT or DispatchingRule.SRT_SMPT
                    => candidates[ArgMinIdx(req.CandidateJobTimes)],
                DispatchingRule.SPT_SRWT or DispatchingRule.SRT_SRWT or DispatchingRule.SDT_SRWT
                    => candidates[ArgMinIdx(req.CandidateQueueLengths)],
                DispatchingRule.LPT_MMUR or DispatchingRule.LRT_MMUR
                    => candidates[ArgMaxIdx(req.CandidateQueueLengths)],
                _ => candidates[UnityEngine.Random.Range(0, candidates.Length)]
            };
        }

        public static float GetRemainingWork(int jobId, JobStore jobs)
        {
            JobData j = jobs.Get(jobId);
            if (j == null) return 0f;
            float total = 0f;
            for (int o = j.CurrentOpIndex; o < j.TotalOperations; o++)
                total += j.EligibleMachinesPerOp[o].Values.Min();
            return total;
        }

        private static int ArgMin(List<int> ids, Func<int, float> score)
        {
            int best = ids[0]; float bestS = float.MaxValue;
            foreach (int id in ids) { float s = score(id); if (s < bestS) { bestS = s; best = id; } }
            return best;
        }

        private static int ArgMax(List<int> ids, Func<int, float> score)
        {
            int best = ids[0]; float bestS = float.MinValue;
            foreach (int id in ids) { float s = score(id); if (s > bestS) { bestS = s; best = id; } }
            return best;
        }

        private static int ArgMinIdx(float[] v)
        {
            int b = 0;
            for (int i = 1; i < v.Length; i++) if (v[i] < v[b]) b = i;
            return b;
        }

        private static int ArgMaxIdx(float[] v)
        {
            int b = 0;
            for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
            return b;
        }
    }
}