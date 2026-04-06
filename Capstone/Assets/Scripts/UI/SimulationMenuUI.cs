using System;
using System.Linq;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;
using TMPro;
using Assets.Scripts.Simulation.Machines;
using Assets.Scripts.Simulation.Types;

namespace Assets.Scripts.Simulation.UI
{
    /// @brief Wires a pre-built UI panel to the SimulationBridge lifecycle.
    ///
    /// @details Drop your Canvas sliders, labels, and buttons into the inspector.
    ///          This script configures slider ranges, hooks up button clicks,
    ///          and manages panel visibility. No programmatic UI creation.
    ///
    /// Workflow:
    ///   App starts → panel visible → user tweaks sliders →
    ///   Spawn Factory → inspect layout → Start Simulation → panel hides →
    ///   episode ends → panel reappears.
    public class SimulationMenuUI : MonoBehaviour
    {
        // ─────────────────────────────────────────────────────────
        //  Inspector References
        // ─────────────────────────────────────────────────────────

        [Header("Bridge")]
        [SerializeField] private SimulationBridge bridge;

        [Header("Panel Root")]
        [SerializeField] private GameObject panel;

        [Header("Sliders")]
        [SerializeField] private Slider jobCountSlider;
        [SerializeField] private Slider machinesPerTypeSlider;
        [SerializeField] private Slider minProcTimeSlider;
        [SerializeField] private Slider maxProcTimeSlider;
        [SerializeField] private Slider minOpsSlider;
        [SerializeField] private Slider maxOpsSlider;
        [SerializeField] private Slider arrivalWindowSlider;

        [Header("Value Labels (TMP)")]
        [SerializeField] private TextMeshProUGUI jobCountText;
        [SerializeField] private TextMeshProUGUI machinesPerTypeText;
        [SerializeField] private TextMeshProUGUI minProcTimeText;
        [SerializeField] private TextMeshProUGUI maxProcTimeText;
        [SerializeField] private TextMeshProUGUI minOpsText;
        [SerializeField] private TextMeshProUGUI maxOpsText;
        [SerializeField] private TextMeshProUGUI arrivalWindowText;

        [Header("Buttons")]
        [SerializeField] private Button spawnButton;
        [SerializeField] private Button startButton;

        [Header("Optional")]
        [SerializeField] private TextMeshProUGUI statusText;
        [SerializeField] private TextMeshProUGUI totalMachinesText;
        [SerializeField] private Slider seedSlider;
        [SerializeField] private TextMeshProUGUI seedText;
        [SerializeField] private TMP_InputField configNameField;

        // ─────────────────────────────────────────────────────────
        //  Internal State
        // ─────────────────────────────────────────────────────────

        private MachineType[] allMachineTypes;
        private bool panelVisible = true;

        // ─────────────────────────────────────────────────────────
        //  Unity Lifecycle
        // ─────────────────────────────────────────────────────────

        private void Awake()
        {
            allMachineTypes = (MachineType[])Enum.GetValues(typeof(MachineType));
        }

        private void Start()
        {
            if (Application.isBatchMode)
            {
                if (panel != null) panel.SetActive(false);
                return;
            }

            ConfigureSliders();
            WireCallbacks();
            RefreshButtonStates();
            ShowPanel();
        }

        private void OnEnable()
        {
            if (bridge != null)
            {
                bridge.OnFactorySpawned?.AddListener(OnFactorySpawned);
                bridge.OnEpisodeFinished?.AddListener(OnEpisodeFinished);
            }
        }

        private void OnDisable()
        {
            if (bridge != null)
            {
                bridge.OnFactorySpawned?.RemoveListener(OnFactorySpawned);
                bridge.OnEpisodeFinished?.RemoveListener(OnEpisodeFinished);
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

        // ─────────────────────────────────────────────────────────
        //  Slider Configuration
        // ─────────────────────────────────────────────────────────

        private void ConfigureSliders()
        {
            SetupSlider(jobCountSlider, 1, 200, 20, true, jobCountText);
            SetupSlider(machinesPerTypeSlider, 1, 10, 3, true, machinesPerTypeText, _ => UpdateTotalMachines());
            SetupSlider(minProcTimeSlider, 1, 300, 15, true, minProcTimeText, _ => ClampMinMax(minProcTimeSlider, maxProcTimeSlider, minProcTimeText, maxProcTimeText));
            SetupSlider(maxProcTimeSlider, 1, 300, 90, true, maxProcTimeText, _ => ClampMinMax(minProcTimeSlider, maxProcTimeSlider, minProcTimeText, maxProcTimeText));
            SetupSlider(minOpsSlider, 1, 15, 3, true, minOpsText, _ => ClampMinMax(minOpsSlider, maxOpsSlider, minOpsText, maxOpsText));
            SetupSlider(maxOpsSlider, 1, 15, 7, true, maxOpsText, _ => ClampMinMax(minOpsSlider, maxOpsSlider, minOpsText, maxOpsText));
            SetupSlider(arrivalWindowSlider, 0, 600, 0, true, arrivalWindowText);

            if (seedSlider != null)
                SetupSlider(seedSlider, 1, 9999, 42, true, seedText);

            UpdateTotalMachines();
        }

        private void SetupSlider(Slider slider, float min, float max, float defaultVal,
                                  bool wholeNumbers, TMP_Text label,
                                  UnityEngine.Events.UnityAction<float> extraCallback = null)
        {
            if (slider == null) return;

            slider.minValue = min;
            slider.maxValue = max;
            slider.wholeNumbers = wholeNumbers;
            slider.value = defaultVal;

            slider.onValueChanged.AddListener(v => UpdateLabel(label, v, wholeNumbers));
            if (extraCallback != null)
                slider.onValueChanged.AddListener(extraCallback);

            UpdateLabel(label, defaultVal, wholeNumbers);
        }

        private void UpdateLabel(TMP_Text label, float value, bool wholeNumbers)
        {
            if (label == null) return;
            label.text = wholeNumbers ? $"{(int)value}" : $"{value:F1}";
        }

        private void ClampMinMax(Slider minSlider, Slider maxSlider, TMP_Text minLabel, TMP_Text maxLabel)
        {
            if (minSlider == null || maxSlider == null) return;

            if (minSlider.value > maxSlider.value)
                maxSlider.value = minSlider.value;

            UpdateLabel(minLabel, minSlider.value, minSlider.wholeNumbers);
            UpdateLabel(maxLabel, maxSlider.value, maxSlider.wholeNumbers);
        }

        private void UpdateTotalMachines()
        {
            if (totalMachinesText == null || machinesPerTypeSlider == null) return;
            int typeCount = allMachineTypes.Length;
            int total = typeCount * (int)machinesPerTypeSlider.value;
            totalMachinesText.text = $"{total} machines ({typeCount} types x {(int)machinesPerTypeSlider.value})";
        }

        // ─────────────────────────────────────────────────────────
        //  Button Wiring
        // ─────────────────────────────────────────────────────────

        private void WireCallbacks()
        {
            if (spawnButton != null)
                spawnButton.onClick.AddListener(OnSpawnClicked);

            if (startButton != null)
                startButton.onClick.AddListener(OnStartClicked);
        }

        private void RefreshButtonStates()
        {
            if (bridge == null) return;

            if (spawnButton != null)
                spawnButton.interactable = !bridge.IsEpisodeActive;

            if (startButton != null)
            {
                startButton.interactable = bridge.IsFactoryReady && !bridge.IsEpisodeActive;
                startButton.gameObject.SetActive(bridge.IsFactoryReady && !bridge.IsEpisodeActive);
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Button Handlers
        // ─────────────────────────────────────────────────────────

        private void OnSpawnClicked()
        {
            if (bridge == null) return;

            FJSSPConfig config = BuildConfig();
            bridge.LoadConfig(config);
            bridge.SpawnFactory();
        }

        private void OnStartClicked()
        {
            if (bridge == null) return;

            bridge.StartSimulationInteractive();
            SetStatus("Simulation running...");
            HidePanel();
        }

        // ─────────────────────────────────────────────────────────
        //  Bridge Events
        // ─────────────────────────────────────────────────────────

        private void OnFactorySpawned()
        {
            SetStatus($"Factory ready — {bridge.CurrentConfig.TotalMachines} machines. Press Start.");
        }

        private void OnEpisodeFinished(EpisodeResult result)
        {
            SetStatus($"Done — makespan: {result.Makespan:F1}s, decisions: {result.DecisionPoints}");
            ShowPanel();
        }

        // ─────────────────────────────────────────────────────────
        //  Panel Visibility
        // ─────────────────────────────────────────────────────────

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

        // ─────────────────────────────────────────────────────────
        //  Config Builder
        // ─────────────────────────────────────────────────────────

        private FJSSPConfig BuildConfig()
        {
            int mpt = (int)machinesPerTypeSlider.value;

            var layout = new MachineType[allMachineTypes.Length * mpt];
            for (int t = 0; t < allMachineTypes.Length; t++)
                for (int m = 0; m < mpt; m++)
                    layout[t * mpt + m] = allMachineTypes[t];

            string name = configNameField != null && !string.IsNullOrEmpty(configNameField.text)
                ? configNameField.text
                : $"{(int)jobCountSlider.value}j_{layout.Length}m";

            return new FJSSPConfig
            {
                Name = name,
                Seed = seedSlider != null ? (int)seedSlider.value : 42,
                JobCount = (int)jobCountSlider.value,
                MachinesPerType = mpt,
                MachineTypeLayout = layout,
                MinProcTime = minProcTimeSlider.value,
                MaxProcTime = maxProcTimeSlider.value,
                MinOpsPerJob = (int)minOpsSlider.value,
                MaxOpsPerJob = (int)maxOpsSlider.value,
                MaxArrivalTime = arrivalWindowSlider.value,
            };
        }
    }
}