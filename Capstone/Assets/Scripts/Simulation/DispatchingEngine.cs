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
        /// SRT_SRWT, SRT_SMPT, LRT_MMUR, FIFO_SRWT, and Random. These keep catalog indices 0-8 (baselines,
        /// Random's pool); the RL action space is the two heads below (JobHead x MachineHead).
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
            DispatchingRule.FIFO_SRWT,  // FIFO (longest in its current queue) - Server
            DispatchingRule.Random
            // NOTE: Random must stay last -- Resolve draws Random.Range(0, ActionToRule.Length - 1),
            // which relies on this position to exclude itself.
        };

        /// <summary>
        /// Every rule the engine can run: ActionToRule first (so indices 0..ActionCount-1 keep their old
        /// values), then the rest of the job x machine catalog. The agent reaches its rules through
        /// IndexForHeads.
        /// </summary>
        private static readonly DispatchingRule[] AllRules = ActionToRule
            .Concat(((DispatchingRule[])Enum.GetValues(typeof(DispatchingRule))).Where(r => !ActionToRule.Contains(r)))
            .ToArray();

        /// <summary>Job-priority half of a rule: which job to take (dispatch) or route first.</summary>
        private enum JobRule { SPT, LPT, SRT, LRT, FIFO, PTWINQ, EDD, MDD, ATC }

        /// <summary>ATC's look-ahead parameter k (Vepsalainen &amp; Morton 1987 report k of about 1.5-3); the same
        /// constant as env/des_twin/rules.py ATC_K.</summary>
        private const double AtcK = 2.0;

        /// <summary>Machine-selection half of a rule: which candidate machine to route a job to.</summary>
        private enum MachineRule { SMPT, SRWT, MMUR, ECT, TECT }

        /// <summary>(job half, machine half) of every non-Random rule, parsed once from its JOB_MACHINE name.</summary>
        private static readonly Dictionary<DispatchingRule, (JobRule job, MachineRule machine)> Halves =
            AllRules.Where(r => r != DispatchingRule.Random).ToDictionary(r => r, r =>
            {
                string[] parts = r.ToString().Split('_');
                return ((JobRule)Enum.Parse(typeof(JobRule), parts[0]), (MachineRule)Enum.Parse(typeof(MachineRule), parts[1]));
            });

        /// <summary>
        /// RL action branch 0 (job head): the job half the agent picks. Order is the Python contract
        /// (env/config.py JOB_HEAD_RULES); append only.
        /// </summary>
        private static readonly JobRule[] JobHead =
            { JobRule.SPT, JobRule.SRT, JobRule.PTWINQ, JobRule.FIFO, JobRule.EDD, JobRule.MDD, JobRule.ATC };

        /// <summary>
        /// RL action branch 1 (machine head): the machine half the agent picks. Order is the Python contract
        /// (env/config.py MACHINE_HEAD_RULES); append only.
        /// </summary>
        private static readonly MachineRule[] MachineHead = { MachineRule.ECT, MachineRule.TECT, MachineRule.SRWT };

        /// <summary>Catalog index of each (job head, machine head) pair; every pair must exist as a JOB_MACHINE rule.</summary>
        private static readonly int[,] HeadsToIndex = BuildHeadsToIndex();

        private static int[,] BuildHeadsToIndex()
        {
            var table = new int[JobHead.Length, MachineHead.Length];
            for (int j = 0; j < JobHead.Length; j++)
                for (int m = 0; m < MachineHead.Length; m++)
                {
                    DispatchingRule rule = Halves.First(kv => kv.Value == (JobHead[j], MachineHead[m])).Key;
                    table[j, m] = IndexForRule(rule);
                }
            return table;
        }

        /// <summary>Size of RL action branch 0 (job head).</summary>
        public static int JobBranchSize => JobHead.Length;

        /// <summary>Size of RL action branch 1 (machine head).</summary>
        public static int MachineBranchSize => MachineHead.Length;

        /// <summary>Catalog index (for Step) of the rule an RL action (job head, machine head) stands for.</summary>
        public static int IndexForHeads(int jobHead, int machineHead) => HeadsToIndex[jobHead, machineHead];

        /// <summary>The RL head indices that reproduce a catalog rule; false if a half is not in its head.</summary>
        public static bool TryHeadsForRule(DispatchingRule rule, out int jobHead, out int machineHead)
        {
            jobHead = machineHead = -1;
            if (!Halves.TryGetValue(rule, out var halves)) return false;
            jobHead = Array.IndexOf(JobHead, halves.job);
            machineHead = Array.IndexOf(MachineHead, halves.machine);
            return jobHead >= 0 && machineHead >= 0;
        }

        /// <summary>
        /// Which RL heads can change the outcome of this decision. Dispatch uses only the job half, and only
        /// with more than one job queued. Routing uses the job half when the pool holds more than one job,
        /// and the machine half when the job has more than one candidate machine. With a pool &gt; 1 the
        /// job is picked at Step time, so the machine head counts whenever the pool does (the focus job's
        /// candidates do not tell us the picked job's).
        /// </summary>
        public static (bool job, bool machine) HeadsThatMatter(DecisionRequest req)
        {
            if (req.Type == DecisionType.Dispatch)
                return ((req.QueuedJobIds?.Length ?? 0) > 1, false);

            bool poolChoice = !req.JobSelectedByRule && (req.JobCandidateIds?.Length ?? 0) > 1;
            return (poolChoice, poolChoice || (req.CandidateMachineIds?.Length ?? 0) > 1);
        }

        /// <summary>
        /// Number of legacy rules in ActionToRule (the pool Random draws from). Not the RL action space any more:
        /// the agent acts with two branches, JobBranchSize x MachineBranchSize (see IndexForHeads).
        /// </summary>
        public static int ActionCount => ActionToRule.Length;

        /// <summary>
        /// Retrieves the dispatching rule associated with the given index (RL action or catalog index).
        /// </summary>
        /// <param name="index">Zero-based index of the rule.</param>
        /// <returns>The dispatching rule at the specified index.</returns>
        public static DispatchingRule RuleForIndex(int index) => AllRules[index];

        /// <summary>
        /// Retrieves the index of the given dispatching rule: its RL action index for the 9 legacy rules,
        /// otherwise its catalog index (>= ActionCount).
        /// </summary>
        /// <param name="rule">The dispatching rule to look up.</param>
        /// <returns>The zero-based index of the rule, or -1 if not found.</returns>
        public static int IndexForRule(DispatchingRule rule) => Array.IndexOf(AllRules, rule);

        /// <summary>
        /// Names of an action's (job, machine) halves for the decision log (audit G5). Does not draw from
        /// the random stream: Random is reported as "Random" for both halves, not as the rule it resampled.
        /// </summary>
        public static (string job, string machine) DescribeAction(int index)
        {
            if (index < 0 || index >= AllRules.Length) return ("?", "?");
            DispatchingRule rule = AllRules[index];
            if (!Halves.TryGetValue(rule, out var h)) return (rule.ToString(), rule.ToString());
            return (h.job.ToString(), h.machine.ToString());
        }

        /// <summary>
        /// Resolves an index to its (job, machine) halves. Random re-samples one of the 8 legacy non-random
        /// rules per call, as before.
        /// </summary>
        private static (JobRule job, MachineRule machine) Resolve(int index)
        {
            DispatchingRule rule = AllRules[index];
            if (rule == DispatchingRule.Random)
                rule = ActionToRule[UnityEngine.Random.Range(0, ActionToRule.Length - 1)];
            return Halves[rule];
        }

        /// <summary>
        /// Selects the best job for a given machine based on the dispatching rule specified by actionIndex.
        /// Uses the rule to evaluate candidate jobs and returns the job ID that best satisfies the priority criterion.
        /// </summary>
        /// <param name="actionIndex">RL action or catalog index (see RuleForIndex).</param>
        /// <param name="machineId">ID of the machine that needs a job assigned.</param>
        /// <param name="jobs">Reference to the JobStore containing all job data.</param>
        /// <param name="simTime">Current simulation time, used for time-based rules such as FIFO.</param>
        /// <returns>The selected job ID, or -1 if no dispatchable jobs are available for the machine.</returns>
        public static int SelectJob(int actionIndex, int machineId, JobStore jobs, double simTime)
        {
            JobRule rule = Resolve(actionIndex).job;

            List<int> queue = jobs.GetDispatchableJobs(machineId);
            if (queue.Count == 0) return -1;
            if (queue.Count == 1) return queue[0];

            return RankJobs(rule, queue, jobs, simTime, id => jobs.Get(id).GetProcessingTime(machineId));
        }

        /// <summary>
        /// Selects which job gets the next routing decision, when multiple jobs are simultaneously
        /// ready (state NeedsRouting) — the job-priority half of a rule (SPT/LPT/SRT/LRT/FIFO/PTWINQ),
        /// applied at the point of routing-eligibility rather than only at machine-side dispatch.
        /// </summary>
        /// <remarks>
        /// Machine-agnostic proxies replace SelectJob's per-machine stats, since the target
        /// machine hasn't been chosen yet at this point: SPT/LPT/PTWINQ use the job's minimum processing
        /// time across its eligible machines for the current op (GetMinEligibleProcTime) in place
        /// of processing time at one specific machine; SRT/LRT/FIFO reuse GetRemainingWork and the
        /// time-in-queue formula unchanged, since neither depends on a specific machine.
        /// </remarks>
        /// <param name="actionIndex">RL action or catalog index (see RuleForIndex).</param>
        /// <param name="readyJobIds">IDs of jobs currently ready for a routing decision (has &gt;=1 available eligible machine).</param>
        /// <param name="jobs">Reference to the JobStore containing all job data.</param>
        /// <param name="simTime">Current simulation time, used for FIFO.</param>
        /// <returns>The selected job ID, or -1 if readyJobIds is empty.</returns>
        public static int SelectRoutingJob(int actionIndex, List<int> readyJobIds, JobStore jobs, double simTime)
        {
            JobRule rule = Resolve(actionIndex).job;

            if (readyJobIds.Count == 0) return -1;
            if (readyJobIds.Count == 1) return readyJobIds[0];

            return RankJobs(rule, readyJobIds, jobs, simTime, id => GetMinEligibleProcTime(id, jobs));
        }

        /// <summary>
        /// Applies a job-priority rule to a candidate set (shared by dispatch and routing-job selection).
        /// </summary>
        /// <param name="procTime">The processing-time stat for SPT/LPT/PTWINQ: time on this machine
        /// (dispatch) or the minimum over eligible machines (routing).</param>
        private static int RankJobs(JobRule rule, List<int> ids, JobStore jobs, double simTime, Func<int, float> procTime)
        {
            switch (rule)
            {
                // Shortest / longest processing time
                case JobRule.SPT: return ArgMin(ids, procTime);
                case JobRule.LPT: return ArgMax(ids, procTime);
                // Shortest / longest remaining work across all remaining operations
                case JobRule.SRT: return ArgMin(ids, id => GetRemainingWork(id, jobs));
                case JobRule.LRT: return ArgMax(ids, id => GetRemainingWork(id, jobs));
                // FIFO — the job that has waited longest in its current queue. StateEntryTime is when the job entered
                // its current state, which for these candidates is the queue they are in: Queued at this machine
                // (dispatch) or NeedsRouting in the routing pool (routing). A job sent back to the pool (machine
                // failure, AGV hand-back) rejoins it at the back. Until 2026-10-02 this ranked by time since shop
                // arrival (ArrivalTime), i.e. first in system, first served.
                case JobRule.FIFO: return ArgMax(ids, id => (float)(simTime - jobs.Get(id).StateEntryTime));
                // PT+WINQ — processing time plus the least queued work among the machines that can take the
                // job's next operation (0 on its last op): favours short jobs headed for idle machines.
                case JobRule.PTWINQ:
                {
                    Dictionary<int, float> loads = jobs.GetAllMachineLoads();
                    return ArgMin(ids, id => procTime(id) + WorkInNextQueue(jobs.Get(id), loads));
                }
                // Due-date rules (ported from env/des_twin/rules.py, which the rq2-twin-due screen ran). d = due date
                // (+infinity when the job has none), w = remaining work (SRT's quantity), t = simTime, p = procTime.
                // Scores are doubles, as in the twin. EDD: earliest due date.
                case JobRule.EDD: return ArgMinD(ids, id => jobs.Get(id).DueDate);
                // MDD (modified due date, Baker & Bertrand): max(d, t + w) -- EDD while a job can still finish on
                // time, SRT once it cannot.
                case JobRule.MDD:
                    return ArgMinD(ids, id => Math.Max((double)jobs.Get(id).DueDate, simTime + GetRemainingWork(id, jobs)));
                // ATC (apparent tardiness cost, Vepsalainen & Morton), highest first: (1/p) exp(-max(0, d - t - w) /
                // (k p_mean)), p_mean over the candidates. Without their waiting-time look-ahead in the slack.
                case JobRule.ATC:
                {
                    double pSum = 0;
                    foreach (int id in ids) pSum += procTime(id);
                    double scale = AtcK * Math.Max(pSum / ids.Count, 1e-6);
                    return ArgMaxD(ids, id =>
                    {
                        double p = Math.Max((double)procTime(id), 1e-6);
                        double slack = jobs.Get(id).DueDate - simTime - GetRemainingWork(id, jobs);
                        return Math.Exp(-Math.Max(0.0, slack) / scale) / p;
                    });
                }
                default: return ids[UnityEngine.Random.Range(0, ids.Count)];
            }
        }

        /// <summary>
        /// WINQ for a flexible next operation: the least load (GetMachineLoad) among the machines eligible
        /// for the job's operation after the current one; 0 when the current operation is its last.
        /// </summary>
        private static float WorkInNextQueue(JobData job, Dictionary<int, float> loads)
        {
            int next = job.CurrentOpIndex + 1;
            if (next >= job.TotalOperations) return 0f;
            float best = float.MaxValue;
            foreach (int m in job.EligibleMachinesPerOp[next].Keys)
                if (loads.TryGetValue(m, out float l) && l < best) best = l;
            if (best == float.MaxValue)
            {
                // GetAllMachineLoads has an entry for every machine on the floor, so this means the next
                // operation's eligible machines are not on the floor at all: broken state, not "no queue"
                // (audit A3). Keep 0 so the rule still ranks, but say so.
                Logging.SimLogger.Error($"[DispatchingEngine] PTWINQ: job {job.JobId} op {next} has no eligible machine " +
                                "on the floor; WINQ taken as 0.");
                return 0f;
            }
            return best;
        }

        /// <summary>
        /// Minimum processing time for a job's current operation across its eligible machines —
        /// the best-case cost proxy used by SelectRoutingJob's SPT/LPT scoring, before a specific
        /// machine has been chosen. Mirrors the per-op convention already used by GetRemainingWork.
        /// </summary>
        /// <param name="jobId">ID of the job to evaluate.</param>
        /// <param name="jobs">Reference to the JobStore containing all job data.</param>
        /// <returns>Minimum processing time across eligible machines for the current operation.</returns>
        public static float GetMinEligibleProcTime(int jobId, JobStore jobs)
        {
            JobData j = jobs.Get(jobId);
            if (j == null) return 0f;
            return j.EligibleMachinesPerOp[j.CurrentOpIndex].Values.Min();
        }

        /// <summary>
        /// Selects the best machine from candidate machines based on the dispatching rule specified by actionIndex.
        /// Evaluates candidates using metrics from the DecisionRequest (e.g., job times, queue lengths).
        /// </summary>
        /// <param name="actionIndex">RL action or catalog index (see RuleForIndex).</param>
        /// <param name="req">The decision request containing candidate machine IDs and their associated metrics.</param>
        /// <returns>The selected machine ID from the candidate set.</returns>
        public static int SelectMachine(int actionIndex, DecisionRequest req)
        {
            MachineRule rule = Resolve(actionIndex).machine;

            int[] candidates = req.CandidateMachineIds;
            if (candidates.Length == 1) return candidates[0];

            switch (rule)
            {
                // Machine-focused rules — select machine with minimum job processing time
                case MachineRule.SMPT: return candidates[ArgMinIdx(req.CandidateJobTimes)];
                // Server-focused rules — select machine with minimum queued workload (SRWT)
                case MachineRule.SRWT: return candidates[ArgMinIdx(req.CandidateQueueLengths)];
                // Minimum Machine Utilization Rule — select machine with the lowest cumulative
                // utilization ratio so far (distinct signal from SRWT's instantaneous queued
                // workload: a machine can be idle right now yet have run hot all episode, or
                // vice versa).
                case MachineRule.MMUR: return candidates[ArgMinIdx(req.CandidateUtilization)];
                // Earliest completion: queued work (incl. in-process remainder) + this job's time there.
                case MachineRule.ECT:
                {
                    var ect = new float[candidates.Length];
                    for (int i = 0; i < ect.Length; i++) ect[i] = req.CandidateQueueLengths[i] + req.CandidateJobTimes[i];
                    return candidates[ArgMinIdx(ect)];
                }
                // Travel-aware earliest completion: the job can start once it has arrived AND the machine has
                // cleared its queue. Without travel estimates (null) this is plain ECT. With a travel price λ > 0 the
                // trip is also charged as AGV time the rest of the floor loses (+ λ x travel), so a far machine has to
                // save more queueing to be chosen; λ = 0 leaves the score exactly as before.
                case MachineRule.TECT:
                {
                    var ect = new float[candidates.Length];
                    for (int i = 0; i < ect.Length; i++)
                    {
                        float travel = req.CandidateTravelTimes != null ? req.CandidateTravelTimes[i] : 0f;
                        ect[i] = Math.Max(travel, req.CandidateQueueLengths[i]) + req.CandidateJobTimes[i];
                        if (req.TravelPrice > 0f && travel < float.MaxValue)
                            ect[i] += req.TravelPrice * travel;
                    }
                    return candidates[ArgMinIdx(ect)];
                }
                default: return candidates[UnityEngine.Random.Range(0, candidates.Length)];
            }
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

        /// <summary>ArgMin over double scores (the due-date rules); ties go to the first candidate.</summary>
        private static int ArgMinD(List<int> ids, Func<int, double> score)
        {
            int best = ids[0]; double bestS = double.MaxValue;
            foreach (int id in ids) { double s = score(id); if (s < bestS) { bestS = s; best = id; } }
            return best;
        }

        /// <summary>ArgMax over double scores (ATC); ties go to the first candidate.</summary>
        private static int ArgMaxD(List<int> ids, Func<int, double> score)
        {
            int best = ids[0]; double bestS = double.NegativeInfinity;
            foreach (int id in ids) { double s = score(id); if (s > bestS) { bestS = s; best = id; } }
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