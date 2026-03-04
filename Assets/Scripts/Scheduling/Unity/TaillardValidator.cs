using UnityEngine;
using Newtonsoft.Json;
using Scheduling.Data;
using Scheduling.Core;
using Scheduling.Validation;

namespace Scheduling.Unity
{
    
    public class TaillardValidator : MonoBehaviour
    {
        [Header("Instance Configuration")]
        [Tooltip("JSON filename in Resources/Instances/ (without .json extension)")]
        public string instanceFileName = "ta01";

        [Header("Validation Settings")]
        [Tooltip("Run all dispatching rules and compare")]
        public bool runAllRules = true;

        [Tooltip("Single rule to test if runAllRules is false")]
        public DispatchingRule singleRule = DispatchingRule.SPT_SMPT;

        [Header("Debug")]
        [Tooltip("Print the full Gantt schedule to console")]
        public bool printSchedule = false;

        private TaillardInstance _instance;

        void Start()
        {
            LoadAndValidate();
        }

        [ContextMenu("Run Validation")]
        public void LoadAndValidate()
        {
            // Step 1: Load the JSON instance
            _instance = LoadInstance(instanceFileName);
            if (_instance == null) return;

            Debug.Log($"<color=cyan>=== Loaded: {_instance.name} ===</color>");
            Debug.Log($"Jobs: {_instance.JobCount}, Machines: {_instance.MachineCount}");
            Debug.Log($"Known Optimum: {_instance.metadata.optimum}, " +
                      $"Bounds: [{_instance.metadata.lower_bound}, {_instance.metadata.upper_bound}]");

            var runner = new ValidationRunner();

            // Step 2: Validate constraints (should produce zero errors)
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

            // Step 3: Run dispatching rules
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

            // Step 4: Print Gantt if requested
            if (printSchedule)
            {
                var sim = new DESSimulator();
                sim.LoadInstance(_instance);
                sim.ActiveRule = singleRule;
                sim.Run();
                sim.PrintSchedule();
            }
        }

        /// <summary>
        /// Load a Taillard JSON from Resources/Instances/.
        /// Place your JSON files at: Assets/Resources/Instances/ta01.json, etc.
        /// </summary>
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
