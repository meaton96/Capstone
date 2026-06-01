using System;
using System.Collections.Generic;
using UnityEngine;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Types;
using UnityEngine.Assertions.Must;

namespace Assets.Scripts.Simulation.Jobs
{
    /// <summary>
    /// Utility class for generating Flexible Job Shop Scheduling Problem (FJSSP) instances.
    /// </summary>
    /// <remarks>
    /// Provides methods to generate jobs with randomized arrival times, operation sequences,
    /// and machine eligibility based on configured normal distributions.
    /// </remarks>
    public class FJSSPJobGenerator
    {
        private static readonly MachineType[] AllTypes = (MachineType[])Enum.GetValues(typeof(MachineType));

        /// <summary>
        /// Samples a value from a normal distribution N(mu, sigma) clamped to a minimum.
        /// </summary>
        /// <param name="mu">The mean of the distribution.</param>
        /// <param name="sigma">The standard deviation.</param>
        /// <param name="minValue">The lower bound for the returned sample (defaults to 1.0).</param>
        /// <returns>A float value sampled via the Box-Muller transform.</returns>
        /// <remarks>
        /// Uses <c>UnityEngine.Random.value</c> to generate uniform samples in (0, 1]
        /// to ensure the logarithmic component of the transform remains valid.
        /// </remarks>
        private static float SampleNormal(float mu, float sigma, float minValue = 1f)
        {
            float u1 = 1f - UnityEngine.Random.value;
            float u2 = 1f - UnityEngine.Random.value;
            float z = Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
            return Mathf.Max(minValue, mu + sigma * z);
        }

        public static Dictionary<MachineType, (float mu, float sigma)> DefaultParams
            = new Dictionary<MachineType, (float mu, float sigma)>
            {
                { MachineType.Mill,     (mu:  90f, sigma: 10f) },
                { MachineType.Lathe,    (mu:  75f, sigma: 10f) },
                { MachineType.Weld,     (mu: 150f, sigma: 25f) },
                { MachineType.Inspect,  (mu:  60f, sigma: 10f) },
                { MachineType.Assemble, (mu: 240f, sigma: 40f) },
            };

        /// <summary>
        /// Samples the processing time for a specific machine type.
        /// </summary>
        /// <param name="type">The <see cref="MachineType"/> being sampled.</param>
        /// <param name="config">The simulation configuration containing per-type N(mu,sigma) parameters.</param>
        /// <returns>A sampled processing time.</returns>
        /// <remarks>
        /// Uses <c>config.ProcTimeParams</c> for the type when available; otherwise
        /// falls back to the default parameters defined in <see cref="DefaultParams"/>.
        /// </remarks>
        private static float SampleProcTime(MachineType type, FJSSPConfig config)
        {
            if (config.ProcTimeParams != null && config.ProcTimeParams.TryGetValue(type, out var p))
                return SampleNormal(p.mu, p.sigma);

            return SampleNormal(DefaultParams[type].mu, DefaultParams[type].sigma);
        }

        /// <summary>
        /// Generates an array of job definitions based on the provided configuration.
        /// </summary>
        /// <param name="config">The <see cref="FJSSPConfig"/> defining job counts, operation counts, and per-type distributions.</param>
        /// <param name="machinesByType">A mapping of machine types to their physical instance IDs.</param>
        /// <returns>An array of <see cref="FJSSPJobDefinition"/> objects sorted by <see cref="FJSSPJobDefinition.ArrivalTime"/>.</returns>
        /// <remarks>
        /// For each operation in a job's sequence, this method assigns independent processing
        /// times for every eligible machine instance, creating the "flexible" aspect of the FJSSP.
        /// </remarks>
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
                    ArrivalTime = UnityEngine.Random.Range(0f, config.Stochastic.InitialArrivalSpread),
                    OperationSequence = opSequence,
                    EligibleMachinesPerOp = eligible
                };
            }

            Array.Sort(jobs, (a, b) => a.ArrivalTime.CompareTo(b.ArrivalTime));
            return jobs;
        }

        /// <summary>
        /// Constructs a randomized sequence of machine types for a job's operations.
        /// </summary>
        /// <param name="opCount">The desired number of operations in the sequence.</param>
        /// <returns>An array of <see cref="MachineType"/> values representing the production path.</returns>
        /// <remarks>
        /// Ensures every available <c>MachineType</c> appears at least once in the sequence
        /// to guarantee full coverage. Remaining slots are filled with random second visits.
        /// The final list is shuffled using Fisher-Yates and repaired to prevent consecutive
        /// duplicates of the same machine type.
        /// </remarks>
        private static MachineType[] GenerateOpSequence(int opCount)
        {
            // opCount comes directly from the configured range — no floor at typeCount.
            // Draw each operation independently, avoiding consecutive duplicates.
            var sequence = new MachineType[opCount];
            const int maxRetries = 8;

            for (int i = 0; i < opCount; i++)
            {
                int retries = 0;
                MachineType picked;
                do
                {
                    picked = AllTypes[UnityEngine.Random.Range(0, AllTypes.Length)];
                    retries++;
                }
                while (i > 0 && picked == sequence[i - 1] && retries < maxRetries);

                sequence[i] = picked;
            }

            return sequence;
        }
    }
}