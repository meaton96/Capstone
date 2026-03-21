using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Scheduling.Data;
using Scheduling.Core;

namespace Scheduling.Validation
{
    /// <summary>
    /// Exports makespan results and full operation schedules in the same format
    /// as the Python reference generator, enabling direct comparison.
    ///
    /// Usage in Unity:
    ///   Call ValidationExporter.RunAndExport() from a MonoBehaviour,
    ///   or run standalone as a console app.
    ///
    /// Output:
    ///   csharp_makespans.csv      — matches reference_makespans.csv columns
    ///   csharp_schedules.json     — matches reference_schedules.json structure
    /// </summary>
    public static class ValidationExporter
    {
        // Map C# dispatching rules to the same short keys used in the Python script
        private static readonly (DispatchingRule rule, string key, string fullName)[] RuleMap =
        {
            (DispatchingRule.SPT_SMPT,  "SPT",  "shortest_processing_time"),
            (DispatchingRule.LPT_MMUR,  "LPT",  "largest_processing_time"),
            (DispatchingRule.LRT_MMUR,  "MWR",  "most_work_remaining"),
            (DispatchingRule.SDT_SRWT,  "FCFS", "first_come_first_served"),
            (DispatchingRule.MOR,       "MOR",  "most_operations_remaining"),
        };

        public static void RunAndExport(string instanceDir, string outputDir)
        {
            var csvLines = new List<string>
            {
                "instance,rule,rule_full,makespan,optimum,gap_pct,num_jobs,num_machines"
            };
            var allSchedules = new Dictionary<string, List<Dictionary<string, object>>>();

            // Find all JSON files in the instance directory
            var jsonFiles = Directory.GetFiles(instanceDir, "*.json")
                                     .OrderBy(f => f)
                                     .ToArray();

            foreach (var filePath in jsonFiles)
            {
                TaillardInstance instance;
                try
                {
                    string json = File.ReadAllText(filePath);
                    instance = JsonConvert.DeserializeObject<TaillardInstance>(json);
                }
                catch (Exception e)
                {
                    Console.WriteLine($"[SKIP] Failed to read {filePath}: {e.Message}");
                    continue;
                }

                if (instance == null || instance.duration_matrix == null || instance.machines_matrix == null)
                {
                    Console.WriteLine($"[SKIP] Null or incomplete data in {Path.GetFileName(filePath)}");
                    continue;
                }

                string name = instance.name ?? Path.GetFileNameWithoutExtension(filePath);

                Console.WriteLine($"\n{"".PadLeft(60, '=')}");
                Console.WriteLine($"Instance: {name}");
                Console.WriteLine($"{"".PadLeft(60, '=')}");



                foreach (var (rule, key, fullName) in RuleMap)
                {
                    var sim = new DESSimulator();
                    sim.LoadInstance(instance);
                    sim.ActiveRule = rule;
                    double makespan = sim.Run();

                    double? gap = null;
                    if (instance.metadata?.optimum > 0)
                    {
                        gap = Math.Round(
                            (makespan - instance.metadata.optimum)
                            / instance.metadata.optimum * 100.0, 4);
                    }

                    Console.WriteLine(
                        $"  {key,-6} -> makespan={makespan,6:F0}  " +
                        $"optimum={instance.metadata?.optimum}  " +
                        $"gap={gap:F1}%");

                    csvLines.Add(string.Join(",",
                        name,
                        key,
                        fullName,
                        makespan,
                        instance.metadata?.optimum.ToString() ?? "",
                        gap?.ToString() ?? "",
                        instance.JobCount,
                        instance.MachineCount
                    ));

                    // Extract full operation schedule for debugging
                    var opList = new List<Dictionary<string, object>>();
                    foreach (var job in sim.Jobs)
                    {
                        foreach (var op in job.Operations)
                        {
                            opList.Add(new Dictionary<string, object>
                            {
                                ["job"] = op.JobId,
                                ["op_index"] = op.OperationIndex,
                                ["machine"] = op.MachineId,
                                ["start"] = op.StartTime,
                                ["end"] = op.EndTime,
                                ["duration"] = op.Duration,
                            });
                        }
                    }

                    opList.Sort((a, b) =>
                    {
                        int cmp = ((double)a["start"]).CompareTo((double)b["start"]);
                        if (cmp != 0) return cmp;
                        cmp = ((int)a["machine"]).CompareTo((int)b["machine"]);
                        if (cmp != 0) return cmp;
                        return ((int)a["job"]).CompareTo((int)b["job"]);
                    });

                    allSchedules[$"{name}_{key}"] = opList;
                }
            }

            // Write CSV
            string csvPath = Path.Combine(outputDir, "csharp_makespans.csv");
            File.WriteAllLines(csvPath, csvLines);
            Console.WriteLine($"\nSaved {csvLines.Count - 1} rows to {csvPath}");

            // Write schedules JSON
            string jsonPath = Path.Combine(outputDir, "csharp_schedules.json");
            File.WriteAllText(jsonPath, JsonConvert.SerializeObject(allSchedules, Formatting.Indented));
            Console.WriteLine($"Saved {allSchedules.Count} schedules to {jsonPath}");
        }
    }
}
