using UnityEngine;
using Newtonsoft.Json;
using Assets.Scripts.Scheduling.Data;
using Assets.Scripts.Scheduling.Core;
using Assets.Scripts.Scheduling.Validation;

namespace Assets.Scripts.Scheduling.Unity
{
    /// @brief Unity entry point for loading and validating Taillard JSP benchmark instances.
    ///
    /// @details Attach to any GameObject in the scene. On @ref Start, or manually via the
    /// @c "Run Validation" context menu entry, the component loads a @ref TaillardInstance
    /// from @c Resources/Instances/, runs constraint checks through @ref ValidationRunner,
    /// executes one or all @ref DispatchingRule variants, and optionally prints a Gantt
    /// schedule to the console via @ref DESSimulator.PrintSchedule.
    ///
    /// @note JSON files must be placed at @c Assets/Resources/Instances/<name>.json and
    /// referenced by filename without the @c .json extension via @ref instanceFileName.
    public class TaillardValidator : MonoBehaviour
    {
        /// @brief Filename of the Taillard JSON instance to load, without the @c .json extension.
        /// @details The file must exist at @c Assets/Resources/Instances/<instanceFileName>.json.
        /// For example, setting this to @c "ta01" loads @c Assets/Resources/Instances/ta01.json.
        [Header("Instance Configuration")]
        [Tooltip("JSON filename in Resources/Instances/ (without .json extension)")]
        public string instanceFileName = "ta01";

        /// @brief When @c true, runs every value in @ref DispatchingRule and logs a
        /// comparative result table. When @c false, runs only @ref singleRule.
        [Header("Validation Settings")]
        [Tooltip("Run all dispatching rules and compare")]
        public bool runAllRules = true;

        /// @brief The dispatching rule used when @ref runAllRules is @c false,
        /// and always used for constraint validation in Step 2 of @ref LoadAndValidate.
        [Tooltip("Single rule to test if runAllRules is false")]
        public DispatchingRule singleRule = DispatchingRule.SPT_SMPT;

        /// @brief When @c true, runs a fresh @ref DESSimulator after validation and
        /// prints a Gantt-style schedule to the console via @ref DESSimulator.PrintSchedule.
        /// @details Uses @ref singleRule regardless of the value of @ref runAllRules.
        [Header("Debug")]
        [Tooltip("Print the full Gantt schedule to console")]
        public bool printSchedule = false;

        /// @brief The deserialized benchmark instance loaded by @ref LoadInstance.
        /// @details @c null until @ref LoadAndValidate completes Step 1 successfully.
        private TaillardInstance _instance;

        /// @brief Unity lifecycle callback — invokes @ref LoadAndValidate on scene start.
        void Start()
        {
            LoadAndValidate();
        }

        /// @brief Loads the configured Taillard instance and runs the full validation pipeline.
        ///
        /// @details Executes four sequential steps:
        /// -# Load the JSON instance from @c Resources/Instances/ via @ref LoadInstance.
        ///    Aborts early if the file is missing or cannot be parsed.
        /// -# Validate scheduling constraints (precedence, no machine overlap, bound checks)
        ///    using @ref ValidationRunner.ValidateConstraints with @ref DispatchingRule.SPT_SMPT.
        ///    All violations are logged as errors via @c Debug.LogError.
        /// -# Run dispatching rules: all rules via @ref ValidationRunner.RunAllRules if
        ///    @ref runAllRules is @c true, otherwise a single rule via @ref ValidationRunner.RunSingle.
        ///    Results are colour-coded green / yellow / red based on gap percentage.
        /// -# Optionally print a Gantt schedule to the console if @ref printSchedule is @c true,
        ///    using a fresh @ref DESSimulator run with @ref singleRule.
        ///
        /// @note Also callable from the Unity Inspector via the @c "Run Validation" context menu
        /// without entering Play Mode.
        [ContextMenu("Run Validation")]
        public void LoadAndValidate()
        {
            _instance = LoadInstance(instanceFileName);
            if (_instance == null) return;

            Debug.Log($"<color=cyan>=== Loaded: {_instance.name} ===</color>");
            Debug.Log($"Jobs: {_instance.JobCount}, Machines: {_instance.MachineCount}");
            Debug.Log($"Known Optimum: {_instance.metadata.optimum}, " +
                      $"Bounds: [{_instance.metadata.lower_bound}, {_instance.metadata.upper_bound}]");

            var runner = new ValidationRunner();

            var errors = runner.ValidateConstraints(_instance, DispatchingRule.SPT_SMPT);
            if (errors.Count == 0)
            {
                Debug.Log("<color=green>PASS: All constraints satisfied (precedence, no overlap, bounds)</color>");
            }
            else
            {
                foreach (var err in errors)
                    Debug.LogError($"CONSTRAINT VIOLATION: {err}");
            }

            if (runAllRules)
            {
                Debug.Log("<color=yellow>--- Running all dispatching rules ---</color>");
                var results = runner.RunAllRules(_instance);
                foreach (var r in results)
                {
                    string color = r.GapPercent < 20 ? "green" : r.GapPercent < 40 ? "yellow" : "red";
                    Debug.Log($"<color={color}>{r}</color>");
                }
            }
            else
            {
                var result = runner.RunSingle(_instance, singleRule);
                Debug.Log(result.ToString());
            }

            if (printSchedule)
            {
                var sim = new DESSimulator();
                sim.LoadInstance(_instance);
                sim.ActiveRule = singleRule;
                sim.Run();
                sim.PrintSchedule();
            }
        }

        /// @brief Loads and deserializes a Taillard JSON instance from the Unity Resources system.
        ///
        /// @details Resolves the asset at @c Resources/Instances/<fileName> using
        /// @c Resources.Load<TextAsset>. If the asset is not found, logs a descriptive
        /// error including the expected file path and returns @c null. If the asset is found
        /// but JSON parsing fails, logs the exception message and returns @c null.
        /// Uses @c Newtonsoft.Json.JsonConvert rather than @c JsonUtility to support
        /// jagged array deserialization required by @ref TaillardInstance.
        ///
        /// @param fileName Filename without extension, e.g. @c "ta01". Must match a
        /// @c TextAsset located at @c Assets/Resources/Instances/<fileName>.json.
        ///
        /// @returns The deserialized @ref TaillardInstance on success, or @c null if the
        /// file is missing or the JSON cannot be parsed.
        private TaillardInstance LoadInstance(string fileName)
        {
            var textAsset = Resources.Load<TextAsset>($"Instances/{fileName}");
            if (textAsset == null)
            {
                Debug.LogError($"Could not find instance file: Resources/Instances/{fileName}.json\n" +
                               $"Make sure the file exists at Assets/Resources/Instances/{fileName}.json");
                return null;
            }
            try
            {
                var instance = JsonConvert.DeserializeObject<TaillardInstance>(textAsset.text);
                return instance;
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Failed to parse JSON: {e.Message}");
                return null;
            }
        }
    }
}