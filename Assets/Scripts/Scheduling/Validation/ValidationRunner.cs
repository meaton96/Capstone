using System;
using System.Collections.Generic;
using Scheduling.Data;
using Scheduling.Core;

namespace Scheduling.Validation
{
    /// <summary>
    /// Validates the DES simulator against Taillard benchmark instances.
    /// 
    /// Usage (outside Unity):
    ///     var runner = new ValidationRunner();
    ///     runner.RunValidation(taillardInstance);
    ///
    /// Usage (inside Unity):
    ///     Call from a MonoBehaviour using the TaillardValidator component.
    /// </summary>
    public class ValidationRunner
    {
        public struct ValidationResult
        {
            public string InstanceName;
            public int KnownOptimum;
            public int LowerBound;
            public int UpperBound;
            public double Makespan;
            public DispatchingRule RuleUsed;
            public double GapPercent; // (makespan - optimum) / optimum * 100

            public override string ToString()
            {
                return $"[{InstanceName}] Rule={RuleUsed}, Makespan={Makespan}, " +
                       $"Optimum={KnownOptimum}, Gap={GapPercent:F2}%, " +
                       $"Bounds=[{LowerBound}, {UpperBound}]";
            }
        }

        /// <summary>
        /// Run the simulator on a single instance with a given dispatching rule.
        /// Returns the validation result with gap analysis.
        /// </summary>
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

        /// <summary>
        /// Run all dispatching rules on an instance and return results sorted by makespan.
        /// This helps you find which heuristic works best as a baseline.
        /// </summary>
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

        /// <summary>
        /// Sanity checks that should always pass for a correct DES implementation:
        ///  1. Every operation has a valid start/end time
        ///  2. Precedence: each op starts after its predecessor finishes
        ///  3. No machine overlap: no two ops on the same machine overlap in time
        ///  4. Makespan >= lower bound
        /// </summary>
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
