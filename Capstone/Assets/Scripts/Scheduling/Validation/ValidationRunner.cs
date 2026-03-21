using System;
using System.Collections.Generic;
using Assets.Scripts.Scheduling.Data;
using Assets.Scripts.Scheduling.Core;

namespace Assets.Scripts.Scheduling.Validation
{
    /// @brief Validates the @ref DESSimulator against Taillard JSP benchmark instances.
    ///
    /// @details Provides three levels of validation:
    /// - @ref RunSingle — simulate one instance with one @ref DispatchingRule and measure the gap to optimum.
    /// - @ref RunAllRules — sweep every @ref DispatchingRule and return results sorted by makespan.
    /// - @ref ValidateConstraints — verify correctness invariants (valid times, precedence,
    ///   no machine overlap, makespan above lower bound).
    ///
    /// Usage outside Unity:
    /// @code
    /// var runner = new ValidationRunner();
    /// runner.RunValidation(taillardInstance);
    /// @endcode
    ///
    /// Usage inside Unity:
    /// @code
    /// // Call from a MonoBehaviour via the TaillardValidator component.
    /// @endcode
    public class ValidationRunner
    {
        /// @brief Encapsulates the outcome of a single simulation run against a benchmark instance.
        ///
        /// @details Captures the instance identity, all known bound values from
        /// @ref TaillardMetadata, the achieved makespan, the rule used, and the
        /// optimality gap so results from different rules can be compared directly.
        public struct ValidationResult
        {
            /// @brief Name of the benchmark instance, sourced from @ref TaillardInstance.name.
            public string InstanceName;

            /// @brief Proven optimal makespan for this instance, sourced from @ref TaillardMetadata.optimum.
            /// @details Used as the denominator in @ref GapPercent. If @c 0, gap calculation will produce
            /// undefined results.
            public int KnownOptimum;

            /// @brief Theoretical lower bound on the makespan, below which no valid schedule can exist.
            public int LowerBound;

            /// @brief Best known upper bound on the makespan from the published literature.
            public int UpperBound;

            /// @brief Makespan achieved by the simulation run, in simulation time units.
            public double Makespan;

            /// @brief The @ref DispatchingRule used to produce this result.
            public DispatchingRule RuleUsed;

            /// @brief Percentage gap between @ref Makespan and @ref KnownOptimum.
            /// @details Computed as @c (Makespan - KnownOptimum) / KnownOptimum * 100.
            /// A value of @c 0 indicates an optimal schedule was found. Negative values
            /// indicate a bug, since no schedule can beat the proven optimum.
            public double GapPercent;

            /// @brief Returns a single-line summary of the result suitable for console logging.
            ///
            /// @returns A formatted string containing instance name, rule, makespan, optimum,
            /// gap percentage, and bound range.
            public override string ToString()
            {
                return $"[{InstanceName}] Rule={RuleUsed}, Makespan={Makespan}, " +
                       $"Optimum={KnownOptimum}, Gap={GapPercent:F2}%, " +
                       $"Bounds=[{LowerBound}, {UpperBound}]";
            }
        }

        /// @brief Simulates a single instance with one dispatching rule and returns a gap analysis.
        ///
        /// @details Constructs a fresh @ref DESSimulator, loads @p instance, sets @p rule as
        /// the active dispatching rule, runs to completion, and packages the result into a
        /// @ref ValidationResult. The gap percentage is computed relative to
        /// @ref TaillardMetadata.optimum and will be infinite or undefined if optimum is @c 0.
        ///
        /// @param instance The benchmark instance to simulate. Must have valid
        /// @ref TaillardInstance.metadata with a non-zero @ref TaillardMetadata.optimum
        /// for gap analysis to be meaningful.
        /// @param rule The @ref DispatchingRule to apply during the simulation run.
        ///
        /// @returns A @ref ValidationResult containing the makespan, gap, and bound values
        /// for this rule and instance combination.
        public ValidationResult RunSingle(TaillardInstance instance, DispatchingRule rule)
        {
            var sim = new DESSimulator();
            sim.LoadInstance(instance);
            sim.ActiveRule = rule;
            double makespan = sim.Run();

            var result = new ValidationResult
            {
                InstanceName = instance.name,
                KnownOptimum = instance.metadata.optimum,
                LowerBound = instance.metadata.lower_bound,
                UpperBound = instance.metadata.upper_bound,
                Makespan = makespan,
                RuleUsed = rule,
                GapPercent = (makespan - instance.metadata.optimum)
                             / (double)instance.metadata.optimum * 100.0
            };

            return result;
        }

        /// @brief Runs every @ref DispatchingRule on an instance and returns results sorted by makespan.
        ///
        /// @details Iterates all values of the @ref DispatchingRule enum via @c Enum.GetValues,
        /// delegates each to @ref RunSingle, and sorts the collected results by
        /// @ref ValidationResult.Makespan ascending so the best-performing heuristic appears first.
        /// Useful for identifying the strongest baseline rule for a given instance before
        /// applying RL-based scheduling.
        ///
        /// @param instance The benchmark instance to run all rules against.
        ///
        /// @returns A list of @ref ValidationResult values, one per @ref DispatchingRule,
        /// sorted by makespan ascending.
        public List<ValidationResult> RunAllRules(TaillardInstance instance)
        {
            var results = new List<ValidationResult>();

            foreach (DispatchingRule rule in Enum.GetValues(typeof(DispatchingRule)))
            {
                results.Add(RunSingle(instance, rule));
            }

            results.Sort((a, b) => a.Makespan.CompareTo(b.Makespan));
            return results;
        }

        /// @brief Verifies correctness invariants of the @ref DESSimulator against a benchmark instance.
        ///
        /// @details Runs a fresh simulation with @p rule and performs four sequential checks.
        /// All violations found across all checks are collected and returned — execution does
        /// not stop at the first error. A correct implementation should always return an empty list.
        ///
        /// The four checks are:
        /// -# **Valid times**: every @ref Operation has non-negative @ref Operation.StartTime
        ///    and @ref Operation.EndTime. A value of @c -1 indicates the operation was never scheduled.
        /// -# **Precedence**: for each job, operation @c i must start no earlier than operation
        ///    @c i-1 ends. Violations indicate a sequencing bug in @ref DESSimulator.
        /// -# **No machine overlap**: no two operations assigned to the same machine may overlap
        ///    in time. Operations are sorted by start time and checked pairwise. Violations indicate
        ///    a machine contention bug.
        /// -# **Makespan lower bound**: the achieved makespan must be at or above
        ///    @ref TaillardMetadata.lower_bound. A violation indicates a bug that produces
        ///    a schedule that is theoretically impossible.
        ///
        /// @note A floating-point epsilon of @c 1e-9 is applied to all time comparisons
        /// to guard against rounding errors in double arithmetic.
        ///
        /// @param instance The benchmark instance to validate against. Must have valid
        /// @ref TaillardMetadata with a meaningful @ref TaillardMetadata.lower_bound.
        /// @param rule The @ref DispatchingRule to use for the validation simulation run.
        ///
        /// @returns A list of human-readable error strings describing each violation found.
        /// Returns an empty list if all constraints are satisfied.
        public List<string> ValidateConstraints(TaillardInstance instance, DispatchingRule rule)
        {
            var errors = new List<string>();
            var sim = new DESSimulator();
            sim.LoadInstance(instance);
            sim.ActiveRule = rule;
            sim.Run();

            // Check 1: All operations have valid times
            foreach (var job in sim.Jobs)
            {
                foreach (var op in job.Operations)
                {
                    if (op.StartTime < 0 || op.EndTime < 0)
                    {
                        errors.Add($"Job {op.JobId} Op {op.OperationIndex}: " +
                                   $"invalid times (start={op.StartTime}, end={op.EndTime})");
                    }
                }
            }

            // Check 2: Precedence constraints
            foreach (var job in sim.Jobs)
            {
                for (int i = 1; i < job.Operations.Length; i++)
                {
                    var prev = job.Operations[i - 1];
                    var curr = job.Operations[i];
                    if (curr.StartTime < prev.EndTime - 1e-9)
                    {
                        errors.Add($"Job {job.Id}: Op {i} starts at {curr.StartTime} " +
                                   $"before Op {i - 1} ends at {prev.EndTime}");
                    }
                }
            }

            // Check 3: No machine overlap
            for (int m = 0; m < sim.Machines.Length; m++)
            {
                var opsOnMachine = new List<Operation>();
                foreach (var job in sim.Jobs)
                {
                    foreach (var op in job.Operations)
                    {
                        if (op.MachineId == m) opsOnMachine.Add(op);
                    }
                }

                opsOnMachine.Sort((a, b) => a.StartTime.CompareTo(b.StartTime));
                for (int i = 1; i < opsOnMachine.Count; i++)
                {
                    var prev = opsOnMachine[i - 1];
                    var curr = opsOnMachine[i];
                    if (curr.StartTime < prev.EndTime - 1e-9)
                    {
                        errors.Add($"Machine {m}: overlap between " +
                                   $"J{prev.JobId}Op{prev.OperationIndex} [{prev.StartTime}-{prev.EndTime}] " +
                                   $"and J{curr.JobId}Op{curr.OperationIndex} [{curr.StartTime}-{curr.EndTime}]");
                    }
                }
            }

            // Check 4: Makespan >= lower bound
            if (sim.Makespan < instance.metadata.lower_bound - 1e-9)
            {
                errors.Add($"Makespan {sim.Makespan} is below the known lower bound " +
                           $"{instance.metadata.lower_bound} — this should be impossible.");
            }

            return errors;
        }
    }
}