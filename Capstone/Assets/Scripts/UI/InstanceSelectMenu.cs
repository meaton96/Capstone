using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using Assets.Scripts.Simulation;
using Assets.Scripts.Scheduling.Data;
using Newtonsoft.Json;
using Assets.Scripts.Logging;

namespace Assets.Scripts.UI
{
    /// @brief Startup menu that lets the user pick a Taillard instance before
    ///        the simulation begins.
    ///
    /// @details On @c Start, all @c TextAsset files under @c Resources/Instances
    ///          are loaded and their names populate a @c TMP_Dropdown. When the
    ///          user confirms their selection the chosen asset is pushed to
    ///          @c SimulationBridge.TaillardJson and @c StartEpisode() is called.
    ///
    public class InstanceSelectMenu : MonoBehaviour
    {
        [Header("References")]
        [Tooltip("The SimulationBridge to configure and start.")]
        [SerializeField] private SimulationBridge bridge;

        [Header("UI")]
        [Tooltip("Dropdown populated with the names of every instance found in Resources/Instances.")]
        [SerializeField] private TMP_Dropdown instanceDropdown;

        [Tooltip("Button that confirms the selection and starts the simulation.")]
        [SerializeField] private Button startButton;

        [Tooltip("Optional label that shows metadata for the currently highlighted instance.")]
        [SerializeField] private TextMeshProUGUI previewText;

        [Tooltip("Root panel to hide once the episode starts.")]
        [SerializeField] private GameObject menuPanel;

        private List<TextAsset> loadedInstances = new List<TextAsset>();

        // ─────────────────────────────────────────────────────────
        //  Unity Lifecycle
        // ─────────────────────────────────────────────────────────

        /// @brief Loads all instance assets from @c Resources/Instances and
        ///        populates the dropdown. Disables the start button if none are found.
        private void Start()
        {
            LoadInstances();
            PopulateDropdown();

            if (startButton != null)
            {
                startButton.onClick.AddListener(OnStartClicked);
                startButton.interactable = loadedInstances.Count > 0;
            }

            if (instanceDropdown != null)
            {
                instanceDropdown.onValueChanged.AddListener(OnSelectionChanged);
            }

            // Show preview for the first entry immediately
            if (loadedInstances.Count > 0)
            {
                UpdatePreview(0);
            }
        }

        /// @brief Removes UI listeners to prevent stale callbacks after destruction.
        private void OnDestroy()
        {
            if (startButton != null)
                startButton.onClick.RemoveListener(OnStartClicked);

            if (instanceDropdown != null)
                instanceDropdown.onValueChanged.RemoveListener(OnSelectionChanged);
        }

        /// @brief Re-shows the menu and refreshes the dropdown for a new selection.
        public void Show()
        {
            if (menuPanel != null) menuPanel.SetActive(true);
        }

        // ─────────────────────────────────────────────────────────
        //  UI Callbacks
        // ─────────────────────────────────────────────────────────

        /// @brief Pushes the selected @c TextAsset to the bridge and starts the episode.
        /// @details Hides the menu panel so the factory floor is fully visible during play.
        public void OnStartClicked()
        {
            if (bridge == null)
            {
                SimLogger.Error("[InstanceSelectMenu] No SimulationBridge assigned.");
                return;
            }

            int index = instanceDropdown != null ? instanceDropdown.value : 0;

            if (index < 0 || index >= loadedInstances.Count)
            {
                SimLogger.Error("[InstanceSelectMenu] Selected index is out of range.");
                return;
            }

            //bridge.TaillardJson = loadedInstances[index];
            SimLogger.Medium($"[InstanceSelectMenu] Selected instance: {loadedInstances[index].name}");
            bridge.StartEpisode();

            if (menuPanel != null)
                menuPanel.SetActive(false);
        }

        /// @brief Refreshes the preview label when the dropdown selection changes.
        /// @param index  The newly selected dropdown index.
        private void OnSelectionChanged(int index)
        {
            UpdatePreview(index);
        }

        // ─────────────────────────────────────────────────────────
        //  Helpers
        // ─────────────────────────────────────────────────────────

        /// @brief Loads every @c TextAsset found under @c Resources/Instances.
        private void LoadInstances()
        {
            loadedInstances.Clear();

            TextAsset[] found = Resources.LoadAll<TextAsset>("Instances");

            if (found == null || found.Length == 0)
            {
                Debug.LogWarning("[InstanceSelectMenu] No instances found in Resources/Instances.");
                return;
            }

            loadedInstances.AddRange(found);
            SimLogger.Medium($"[InstanceSelectMenu] Loaded {loadedInstances.Count} instance(s).");
        }

        /// @brief Clears and repopulates the dropdown with the loaded instance names.
        private void PopulateDropdown()
        {
            if (instanceDropdown == null) return;

            instanceDropdown.ClearOptions();

            List<string> options = new List<string>();
            foreach (TextAsset asset in loadedInstances)
                options.Add(asset.name);

            instanceDropdown.AddOptions(options);
        }

        /// @brief Parses the instance at @p index and writes a summary to @c previewText.
        /// @param index  Index into @c loadedInstances to preview.
        private void UpdatePreview(int index)
        {
            if (previewText == null) return;
            if (index < 0 || index >= loadedInstances.Count) return;

            try
            {
                TaillardInstance instance = JsonConvert.DeserializeObject<TaillardInstance>(
                    loadedInstances[index].text);

                if (instance == null)
                {
                    previewText.text = "Failed to parse instance.";
                    return;
                }

                string optimumStr = instance.metadata.optimum > 0
                    ? instance.metadata.optimum.ToString()
                    : "unknown";

                previewText.text =
                    $"{instance.Name}\n" +
                    $"Jobs: {instance.JobCount}   Machines: {instance.MachineCount}\n" +
                    $"Optimum: {optimumStr}";
            }
            catch
            {
                previewText.text = "Could not read instance metadata.";
            }
        }
    }
}