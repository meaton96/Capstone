using System;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Types;
using System.Diagnostics;

namespace Assets.Scripts.Simulation.Jobs
{
    public class FJSSPJobGenerator
    {
        private static readonly MachineType[] AllTypes = (MachineType[])Enum.GetValues(typeof(MachineType));

        // ── Per-type processing-time distributions ────────────────────────────────
        //
        //  Group 1 – Automated / Subtractive (G-code, fast, tight σ)
        //    Mill    N(μ=15, σ=2)
        //    Lathe   N(μ=12, σ=2)
        //
        //  Group 2 – Semi-Automated / Thermal (fixturing + heat cycle, wider σ)
        //    Weld    N(μ=30, σ=6)
        //
        //  Group 3 – Quality Assurance (fast gateway, but can become a bottleneck)
        //    Inspect N(μ=8,  σ=3)
        //
        //  Group 4 – Manual / Additive (longest, most unpredictable)
        //    Assemble N(μ=45, σ=12)
        //
        //  Times are clamped to [1, ∞) so no op ever takes zero or negative time.
        // ─────────────────────────────────────────────────────────────────────────
        private static readonly Dictionary<MachineType, (float mu, float sigma)> ProcTimeParams =
            new Dictionary<MachineType, (float mu, float sigma)>
            {
                { MachineType.Mill,     (mu: 180f, sigma:  20f) },
                { MachineType.Lathe,    (mu: 150f, sigma:  20f) },
                { MachineType.Weld,     (mu: 300f, sigma:  50f) },
                { MachineType.Inspect,  (mu: 120f, sigma:  20f) },
                { MachineType.Assemble, (mu: 480f, sigma:  80f) },
            };

        /// <summary>
        /// Returns a sample from N(μ, σ) clamped to [minValue, ∞).
        /// Uses the Box-Muller transform; both UnityEngine.Random calls consume
        /// values in (0, 1] so the log is never called on zero.
        /// </summary>
        private static float SampleNormal(float mu, float sigma, float minValue = 1f)
        {
            // Box-Muller: two uniform samples → one standard-normal variate
            float u1 = 1f - UnityEngine.Random.value; // (0, 1]
            float u2 = 1f - UnityEngine.Random.value;
            float z = Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
            return Mathf.Max(minValue, mu + sigma * z);
        }

        /// <summary>
        /// Returns a processing-time sample for the given machine type drawn from
        /// that type's calibrated N(μ, σ) distribution.
        /// Falls back to the config's flat range if the type is not in the table
        /// (future-proofs against new enum values added before the table is updated).
        /// </summary>
        private static float SampleProcTime(MachineType type, FJSSPConfig config)
        {
            if (ProcTimeParams.TryGetValue(type, out var p))
                return SampleNormal(p.mu, p.sigma);

            // Fallback: uniform draw from config bounds (preserves old behaviour)
            return UnityEngine.Random.Range(config.MinProcTime, config.MaxProcTime);
        }

        public static FJSSPJobDefinition[] Generate(FJSSPConfig config,
                                                     Dictionary<MachineType, List<int>> machinesByType)
        {
            var jobs = new FJSSPJobDefinition[config.JobCount];

            for (int j = 0; j < config.JobCount; j++)
            {
                int opCount = UnityEngine.Random.Range(config.MinOpsPerJob, config.MaxOpsPerJob + 1);
                var opSequence = GenerateOpSequence(opCount);
                opCount = opSequence.Length; // GenerateOpSequence may clamp up to typeCount

                var eligible = new Dictionary<int, float>[opCount];
                for (int o = 0; o < opCount; o++)
                {
                    eligible[o] = new Dictionary<int, float>();
                    foreach (int machineId in machinesByType[opSequence[o]])
                    {
                        // Each machine of the required type gets its own independent
                        // draw so that same-type machines have different queue pressures,
                        // giving the _SRWT router a real signal to exploit.
                        float procTime = SampleProcTime(opSequence[o], config);
                        eligible[o][machineId] = procTime;
                    }
                }

                jobs[j] = new FJSSPJobDefinition
                {
                    JobId = j,
                    ArrivalTime = UnityEngine.Random.Range(0f, config.MaxArrivalTime),
                    OperationSequence = opSequence,
                    EligibleMachinesPerOp = eligible
                };
            }

            Array.Sort(jobs, (a, b) => a.ArrivalTime.CompareTo(b.ArrivalTime));
            return jobs;
        }

        /// <summary>
        /// Builds an operation type sequence of length <paramref name="opCount"/>.
        /// 
        /// Strategy:
        ///   1. Guarantee every MachineType appears at least once (base pass).
        ///   2. Fill remaining slots by randomly picking types for a second visit,
        ///      chosen proportionally so high-demand types appear more often.
        ///   3. Fisher-Yates shuffle the full list.
        ///   4. Repair any consecutive duplicates by swapping with a later non-equal slot.
        /// 
        /// This ensures the agent always sees every machine type in training,
        /// while the randomized extras create routing pressure on specific types
        /// that varies episode to episode.
        /// </summary>
        private static MachineType[] GenerateOpSequence(int opCount)
        {
            int typeCount = AllTypes.Length;

            // opCount should be at least typeCount from config, but clamp defensively
            opCount = Mathf.Max(opCount, typeCount);

            var sequence = new List<MachineType>(opCount);

            // Pass 1: one of each type guaranteed
            foreach (MachineType t in AllTypes)
                sequence.Add(t);

            // Pass 2: fill remaining slots — each type can appear at most twice total
            int remaining = opCount - typeCount;
            var secondVisitPool = new List<MachineType>(AllTypes); // each type eligible once more

            for (int i = 0; i < remaining && secondVisitPool.Count > 0; i++)
            {
                int pick = UnityEngine.Random.Range(0, secondVisitPool.Count);
                sequence.Add(secondVisitPool[pick]);
                secondVisitPool.RemoveAt(pick);
            }

            // Pass 3: Fisher-Yates shuffle
            for (int i = sequence.Count - 1; i > 0; i--)
            {
                int j = UnityEngine.Random.Range(0, i + 1);
                (sequence[i], sequence[j]) = (sequence[j], sequence[i]);
            }

            // Pass 4: repair consecutive duplicates
            // Swap each offender with the next non-equal element further in the list
            for (int i = 0; i < sequence.Count - 1; i++)
            {
                if (sequence[i] == sequence[i + 1])
                {
                    // Find the nearest later slot that breaks the tie
                    bool repaired = false;
                    for (int k = i + 2; k < sequence.Count; k++)
                    {
                        if (sequence[k] != sequence[i])
                        {
                            (sequence[i + 1], sequence[k]) = (sequence[k], sequence[i + 1]);
                            repaired = true;
                            break;
                        }
                    }

                    // Edge case: all remaining are the same type — accept it
                    // This only happens if opCount > typeCount * 2, which config should prevent
                    if (!repaired) break;
                }
            }

            return sequence.ToArray();
        }
    }
}