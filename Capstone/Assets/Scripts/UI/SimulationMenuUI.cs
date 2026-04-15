using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;
using TMPro;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Types;
using Assets.Scripts.Simulation;

namespace Assets.Scripts.UI
{

    [System.Serializable]
    public class MachineTypeProcRow
    {
        /// @brief Display label (not editable at runtime — just a hint in the Inspector).
        public string TypeName;
        public TMP_InputField MuInput;
        public TMP_InputField SigmaInput;
    }

    public class SimulationMenuUI : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private SimulationBridge bridge;
        [SerializeField] private SchedulingAgent agent;
        [SerializeField] private GameObject panel;

        // ── General parameters ────────────────────────────────────────────────
        [Header("General Input Fields")]
        [SerializeField] private TMP_InputField jobCountInput;
        [SerializeField] private TMP_InputField machinesPerTypeInput;
        [SerializeField] private TMP_InputField agvCountInput;
        [SerializeField] private TMP_InputField minOpsInput;
        [SerializeField] private TMP_InputField maxOpsInput;
        [SerializeField] private TMP_InputField arrivalWindowInput;
        [SerializeField] private TMP_InputField seedInput;
        //[SerializeField] private TMP_InputField configNameInput;

        // ── Dispatching Rule ──────────────────────────────────────────────────
        [Header("Heuristics")]
        [SerializeField] private TMP_Dropdown dispatchRuleDropdown; // <-- ADDED THIS

        // ── Per-type proc-time distribution ──────────────────────────────────
        /// <summary>
        /// One entry per MachineType, ordered to match the MachineType enum.
        /// E.g. index 0 → Mill, 1 → Lathe, 2 → Weld, 3 → Inspect, 4 → Assemble.
        /// </summary>
        [Header("Per-Type Processing Time (mu / sigma)")]
        [SerializeField] private MachineTypeProcRow[] procTimeRows;

        // ── Optional labels / totals ──────────────────────────────────────────
        [Header("Optional")]
        [SerializeField] private TextMeshProUGUI statusText;
        [SerializeField] private TextMeshProUGUI totalMachinesText;

        [Header("Buttons")]
        [SerializeField] private Button spawnButton;
        [SerializeField] private Button startButton;


        private static readonly (float mu, float sigma)[] DefaultProcParams =
        {
            (90f,  10f),   // Mill
            (75f,  10f),   // Lathe
            (150f, 25f),   // Weld
            (60f,  10f),   // Inspect
            (240f, 40f),   // Assemble
        };

        private MachineType[] allMachineTypes;
        private bool panelVisible = true;

        // ─────────────────────────────────────────────────────────────────────
        //  Unity lifecycle
        // ─────────────────────────────────────────────────────────────────────

        private void Awake()
        {
            allMachineTypes = (MachineType[])Enum.GetValues(typeof(MachineType));
        }

        private void Start()
        {
            if (Application.isBatchMode || (bridge != null && bridge.AutoStartOnPlay))
            {
                HidePanel();
                return;
            }

            PopulateDefaults();
            WireCallbacks();
            RefreshButtonStates();
            ShowPanel();
        }

        private void OnEnable()
        {
            if (bridge != null)
            {
                bridge.OnFactorySpawned.AddListener(OnFactorySpawned);
                bridge.OnEpisodeFinished.AddListener(OnEpisodeFinished);
            }
        }

        private void OnDisable()
        {
            if (bridge != null)
            {
                bridge.OnFactorySpawned.RemoveListener(OnFactorySpawned);
                bridge.OnEpisodeFinished.RemoveListener(OnEpisodeFinished);
            }
        }

        private void Update()
        {
            if (Keyboard.current != null && Keyboard.current.escapeKey.wasPressedThisFrame)
            {
                if (panelVisible) HidePanel();
                else ShowPanel();
            }

            RefreshButtonStates();
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Initialisation
        // ─────────────────────────────────────────────────────────────────────

        /// @brief Fills every input field with sensible starting values.
        private void PopulateDefaults()
        {
            SetInput(jobCountInput, "20");
            SetInput(machinesPerTypeInput, "3");
            SetInput(agvCountInput, "3");
            SetInput(minOpsInput, "3");
            SetInput(maxOpsInput, "7");
            SetInput(arrivalWindowInput, "0");
            SetInput(seedInput, "42");

            // <-- ADDED DROPDOWN POPULATION -->
            if (dispatchRuleDropdown != null)
            {
                dispatchRuleDropdown.ClearOptions();
                var ruleNames = new List<string>(Enum.GetNames(typeof(DispatchingRule)));
                dispatchRuleDropdown.AddOptions(ruleNames);
                dispatchRuleDropdown.value = (int)DispatchingRule.SRT_SRWT; // Sensible default
            }

            // Per-type mu/sigma rows
            if (procTimeRows != null)
            {
                for (int i = 0; i < procTimeRows.Length && i < DefaultProcParams.Length; i++)
                {
                    SetInput(procTimeRows[i].MuInput, DefaultProcParams[i].mu.ToString("F1"));
                    SetInput(procTimeRows[i].SigmaInput, DefaultProcParams[i].sigma.ToString("F1"));
                }
            }

            UpdateTotalMachinesLabel();
        }

        private void WireCallbacks()
        {
            if (spawnButton != null) spawnButton.onClick.AddListener(OnSpawnClicked);
            if (startButton != null) startButton.onClick.AddListener(OnStartClicked);

            // Recompute the total-machine label whenever machines-per-type changes
            if (machinesPerTypeInput != null)
                machinesPerTypeInput.onValueChanged.AddListener(_ => UpdateTotalMachinesLabel());
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Button handlers
        // ─────────────────────────────────────────────────────────────────────

        private void OnSpawnClicked()
        {
            if (bridge == null) return;
            FJSSPConfig config = BuildConfig();
            bridge.LoadConfig(config);
            bridge.SpawnFactory();
        }

        private void OnStartClicked()
        {
            if (agent == null) return;
            agent.ArmAndStart();
            SetStatus("Simulation running…");
            HidePanel();
        }

        private void RefreshButtonStates()
        {
            if (bridge == null) return;

            if (spawnButton != null)
                spawnButton.interactable = !bridge.IsEpisodeActive;

            if (startButton != null)
            {
                bool canStart = bridge.IsFactoryReady && !bridge.IsEpisodeActive;
                startButton.interactable = canStart;
                startButton.gameObject.SetActive(canStart);
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Bridge events
        // ─────────────────────────────────────────────────────────────────────

        private void OnFactorySpawned()
        {
            SetStatus($"Factory ready — {bridge.CurrentConfig.TotalMachines} machines. Press Start.");
        }

        private void OnEpisodeFinished(EpisodeResult result)
        {
            SetStatus($"<color=#00FF00>SUCCESS!</color>\n" +
                      $"Final Makespan: <b>{result.Makespan:F1}s</b>\n" +
                      $"Decisions Made: <b>{result.DecisionPoints}</b>");
            ShowPanel();
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Config builder
        // ─────────────────────────────────────────────────────────────────────

        private FJSSPConfig BuildConfig()
        {
            int mpt = ParseInt(machinesPerTypeInput, 3);

            var layout = new MachineType[allMachineTypes.Length * mpt];
            for (int t = 0; t < allMachineTypes.Length; t++)
                for (int m = 0; m < mpt; m++)
                    layout[t * mpt + m] = allMachineTypes[t];

            string name = $"{ParseInt(jobCountInput, 20)}j_{layout.Length}m";

            // Build per-type proc-time params from the UI rows
            var procParams = new System.Collections.Generic.Dictionary<MachineType, (float mu, float sigma)>();
            if (procTimeRows != null)
            {
                for (int i = 0; i < procTimeRows.Length && i < allMachineTypes.Length; i++)
                {
                    float mu = ParseFloat(procTimeRows[i].MuInput, DefaultProcParams[i].mu);
                    float sigma = ParseFloat(procTimeRows[i].SigmaInput, DefaultProcParams[i].sigma);
                    procParams[allMachineTypes[i]] = (mu, sigma);
                }
            }

            DispatchingRule selectedRule = DispatchingRule.SRT_SRWT;
            if (dispatchRuleDropdown != null)
            {
                selectedRule = (DispatchingRule)dispatchRuleDropdown.value;
            }

            return new FJSSPConfig
            {
                Name = name,
                Seed = ParseInt(seedInput, 42),
                JobCount = ParseInt(jobCountInput, 20),
                MachinesPerType = mpt,
                MachineTypeLayout = layout,
                AGVCount = ParseInt(agvCountInput, 5),
                MinOpsPerJob = ParseInt(minOpsInput, 3),
                MaxOpsPerJob = ParseInt(maxOpsInput, 7),
                MaxArrivalTime = ParseFloat(arrivalWindowInput, 0f),
                ProcTimeParams = procParams,
                dispatchingRule = selectedRule,
                // Fallback uniform bounds — only used for types missing from procParams
                MinProcTime = 1f,
                MaxProcTime = 30f,
            };
        }

        // ─────────────────────────────────────────────────────────────────────
        //  UI helpers
        // ─────────────────────────────────────────────────────────────────────

        private void ShowPanel()
        {
            panelVisible = true;
            if (panel != null) panel.SetActive(true);
        }

        private void HidePanel()
        {
            panelVisible = false;
            if (panel != null) panel.SetActive(false);
        }

        private void SetStatus(string msg)
        {
            if (statusText != null) statusText.text = msg;
        }

        private void UpdateTotalMachinesLabel()
        {
            if (totalMachinesText == null) return;
            int mpt = ParseInt(machinesPerTypeInput, 1);
            int typeCount = allMachineTypes.Length;
            int total = typeCount * mpt;
            totalMachinesText.text = $"{total} machines ({typeCount} types × {mpt})";
        }

        private static void SetInput(TMP_InputField field, string value)
        {
            if (field != null) field.text = value;
        }

        private static int ParseInt(TMP_InputField field, int fallback)
        {
            if (field == null) return fallback;
            return int.TryParse(field.text, out int v) ? v : fallback;
        }

        private static float ParseFloat(TMP_InputField field, float fallback)
        {
            if (field == null) return fallback;
            return float.TryParse(field.text,
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture,
                out float v) ? v : fallback;
        }
    }
}