using System;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation.Jobs
{
    /// @brief Utility class for generating Flexible Job Shop Scheduling Problem (FJSSP) instances.
    ///
    /// @details Provides methods to generate a series of jobs with randomized arrival times,
    /// operation sequences, and machine eligibility based on configured probability distributions.
    public class FJSSPJobGenerator
    {
        private static readonly MachineType[] AllTypes = (MachineType[])Enum.GetValues(typeof(MachineType));

        /// @brief Samples a value from a normal distribution N(mu, sigma) clamped to a minimum.
        ///
        /// @param mu The mean of the distribution.
        /// @param sigma The standard deviation.
        /// @param minValue The lower bound for the returned sample (defaults to 1.0).
        ///
        /// @return A float value sampled via the Box-Muller transform.
        ///
        /// @details Uses @c UnityEngine.Random.value to generate uniform samples in (0, 1]
        /// to ensure the logarithmic component of the transform remains valid.
        private static float SampleNormal(float mu, float sigma, float minValue = 1f)
        {
            float u1 = 1f - UnityEngine.Random.value;
            float u2 = 1f - UnityEngine.Random.value;
            float z = Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
            return Mathf.Max(minValue, mu + sigma * z);
        }

        /// @brief Calculates the processing time for a specific machine type.
        ///
        /// @param type The @c MachineType being sampled.
        /// @param config The simulation configuration containing per-type N(mu,sigma) params
        ///               and fallback uniform bounds.
        ///
        /// @return A sampled processing time.
        ///
        /// @details Uses @c config.ProcTimeParams for the type when available; otherwise
        /// falls back to a uniform sample in [config.MinProcTime, config.MaxProcTime].
        private static float SampleProcTime(MachineType type, FJSSPConfig config)
        {
            if (config.ProcTimeParams != null && config.ProcTimeParams.TryGetValue(type, out var p))
                return SampleNormal(p.mu, p.sigma);

            return UnityEngine.Random.Range(config.MinProcTime, config.MaxProcTime);
        }

        /// @brief Generates an array of job definitions based on the provided configuration.
        ///
        /// @param config The @c FJSSPConfig defining job counts, operation counts, and
        ///               per-type processing time distributions.
        /// @param machinesByType A mapping of machine types to their physical instance IDs.
        ///
        /// @return An array of @c FJSSPJobDefinition objects sorted by @c ArrivalTime.
        ///
        /// @details For each operation in a job's sequence, this method assigns independent
        /// processing times for every eligible machine instance. This creates the "flexible"
        /// aspect of the FJSSP, providing signals for the router to exploit.
        public static FJSSPJobDefinition[] Generate(FJSSPConfig config,
                                                    Dictionary<MachineType, List<int>> machinesByType)
        {
            var jobs = new FJSSPJobDefinition[config.JobCount];

            for (int j = 0; j < config.JobCount; j++)
            {
                int opCount = UnityEngine.Random.Range(config.MinOpsPerJob, config.MaxOpsPerJob + 1);
                var opSequence = GenerateOpSequence(opCount);
                opCount = opSequence.Length;

                var eligible = new Dictionary<int, float>[opCount];
                for (int o = 0; o < opCount; o++)
                {
                    eligible[o] = new Dictionary<int, float>();
                    foreach (int machineId in machinesByType[opSequence[o]])
                    {
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

        /// @brief Constructs a randomized sequence of machine types for a job's operations.
        ///
        /// @param opCount The desired number of operations in the sequence.
        ///
        /// @return An array of @c MachineType values representing the production path.
        ///
        /// @details Ensures every available @c MachineType appears at least once in the
        /// sequence to guarantee full coverage during training/simulation. Remaining slots
        /// are filled with random second visits. The final list is shuffled and "repaired"
        /// to prevent the same machine type from appearing consecutively.
        private static MachineType[] GenerateOpSequence(int opCount)
        {
            int typeCount = AllTypes.Length;
            opCount = Mathf.Max(opCount, typeCount);

            var sequence = new List<MachineType>(opCount);

            foreach (MachineType t in AllTypes)
                sequence.Add(t);

            int remaining = opCount - typeCount;
            var secondVisitPool = new List<MachineType>(AllTypes);

            for (int i = 0; i < remaining && secondVisitPool.Count > 0; i++)
            {
                int pick = UnityEngine.Random.Range(0, secondVisitPool.Count);
                sequence.Add(secondVisitPool[pick]);
                secondVisitPool.RemoveAt(pick);
            }

            // Fisher-Yates shuffle
            for (int i = sequence.Count - 1; i > 0; i--)
            {
                int j = UnityEngine.Random.Range(0, i + 1);
                (sequence[i], sequence[j]) = (sequence[j], sequence[i]);
            }

            // Repair consecutive duplicates
            for (int i = 0; i < sequence.Count - 1; i++)
            {
                if (sequence[i] == sequence[i + 1])
                {
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
                    if (!repaired) break;
                }
            }

            return sequence.ToArray();
        }
    }
}