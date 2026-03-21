using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Assets.Scripts.Scheduling.Data;
using Assets.Scripts.Scheduling.Core;

namespace Assets.Scripts.Scheduling.Validation
{
    /// @brief Exports makespan results and full operation schedules in the same format
    /// as the Python reference generator, enabling direct diff against job_shop_lib output.
    ///
    /// @details Produces two output files whose column and key names are kept deliberately
    /// in sync with the Python reference scripts:
    /// - @c csharp_makespans.csv — one row per instance/rule combination, matching the
    ///   column layout of @c reference_makespans.csv.
    /// - @c csharp_schedules.json — one entry per instance/rule combination containing the
    ///   full sorted operation schedule, matching the structure of @c reference_schedules.json.
    ///
    /// Usage inside Unity:
    /// @code
    /// ValidationExporter.RunAndExport(instanceDir, outputDir);
    /// @endcode
    ///
    /// Usage as a standalone console app:
    /// @code
    /// ValidationExporter.RunAndExport("path/to/instances", "path/to/output");
    /// @endcode
    ///
    /// @note Only the five rules present in @ref RuleMap are exported. This subset matches
    /// the rules implemented in the Python reference generator; additional @ref DispatchingRule
    /// values are intentionally excluded to keep the comparison columns aligned.
    public static class ValidationExporter
    {
        /// @brief Mapping from @ref DispatchingRule enum values to the short keys and full names
        /// used by the Python reference generator.
        ///
        /// @details Each entry is a @c (rule, key, fullName) tuple where:
        /// - @c rule is the @ref DispatchingRule enum value used to configure @ref DESSimulator.
        /// - @c key is the abbreviated column identifier written to @c csharp_makespans.csv
        ///   (e.g. @c "SPT"), matching the Python script's short-form keys.
        /// - @c fullName is the snake_case name written to the @c rule_full CSV column
        ///   (e.g. @c "shortest_processing_time"), matching the Python script's full-form keys.
        ///
        /// @note Only rules present in this map are simulated and exported by @ref RunAndExport.
        private static readonly (DispatchingRule rule, string key, string fullName)[] RuleMap =
        {
            (DispatchingRule.SPT_SMPT,  "SPT",  "shortest_processing_time"),
            (DispatchingRule.LPT_MMUR,  "LPT",  "largest_processing_time"),
            (DispatchingRule.LRT_MMUR,  "MWR",  "most_work_remaining"),
            (DispatchingRule.SDT_SRWT,  "FCFS", "first_come_first_served"),
            (DispatchingRule.MOR,       "MOR",  "most_operations_remaining"),
        };

        /// @brief Simulates all Taillard instances in a directory under every mapped rule and
        /// writes the results to CSV and JSON export files.
        ///
        /// @details Processes files in the following sequence:
        /// -# Discovers all @c *.json files in @p instanceDir, sorted alphabetically.
        /// -# For each file, deserializes a @ref TaillardInstance via Newtonsoft.Json.
        ///    Files that fail to parse or produce null/incomplete data are skipped with a
        ///    console warning; processing continues with the next file.
        /// -# For each instance, iterates every entry in @ref RuleMap, running a fresh
        ///    @ref DESSimulator per rule. Computes the optimality gap if
        ///    @ref TaillardMetadata.optimum is non-zero; otherwise leaves the gap column empty.
        /// -# Appends one CSV row per instance/rule combination to an in-memory list.
        /// -# Builds a per-instance/rule operation schedule list, sorted by start time,
        ///    then machine ID, then job ID, and stores it under the key @c "{name}_{key}"
        ///    in the schedules dictionary.
        /// -# Writes @c csharp_makespans.csv to @p outputDir containing all CSV rows.
        /// -# Writes @c csharp_schedules.json to @p outputDir containing all schedules,
        ///    serialized with indented formatting for human readability.
        ///
        /// @par CSV columns
        /// @c instance, @c rule, @c rule_full, @c makespan, @c optimum, @c gap_pct,
        /// @c num_jobs, @c num_machines
        ///
        /// @par Schedule JSON structure
        /// A top-level object keyed by @c "{instanceName}_{ruleKey}". Each value is an array
        /// of operation objects with fields: @c job, @c op_index, @c machine, @c start,
        /// @c end, @c duration.
        ///
        /// @param instanceDir Filesystem path to the directory containing Taillard @c .json files.
        /// Must be a real disk path — not a Unity @c Resources/ virtual path.
        /// @param outputDir Filesystem path to the directory where output files will be written.
        /// Must exist before this method is called; @ref CrossValidator.Start creates it if needed.
        public static void RunAndExport(string instanceDir, string outputDir)
        {
            var csvLines = new List<string>
            {
                "instance,rule,rule_full,makespan,optimum,gap_pct,num_jobs,num_machines"
            };
            var allSchedules = new Dictionary<string, List<Dictionary<string, object>>>();

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

            string csvPath = Path.Combine(outputDir, "csharp_makespans.csv");
            File.WriteAllLines(csvPath, csvLines);
            Console.WriteLine($"\nSaved {csvLines.Count - 1} rows to {csvPath}");

            string jsonPath = Path.Combine(outputDir, "csharp_schedules.json");
            File.WriteAllText(jsonPath, JsonConvert.SerializeObject(allSchedules, Formatting.Indented));
            Console.WriteLine($"Saved {allSchedules.Count} schedules to {jsonPath}");
        }
    }
}