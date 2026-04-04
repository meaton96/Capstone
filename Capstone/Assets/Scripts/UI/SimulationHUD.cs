using UnityEngine;
using TMPro;
using Assets.Scripts.Simulation;
using UnityEngine.UI;
using Assets.Scripts.Simulation.Jobs;

namespace Assets.Scripts.UI
{
    /// @brief Heads-up display that polls @c SimulationBridge each frame and
    ///        surfaces key episode metrics to the player.
    ///
    /// @details Displays current simulation time, the last dispatching rule applied,
    /// total decision count, and a completed-jobs tally. Also owns a time-scale
    /// slider that directly modifies @c Time.timeScale.
    public class SimulationHUD : MonoBehaviour
    {
        [Header("UI Text References")]
        [SerializeField] private TextMeshProUGUI timeText;
        [SerializeField] private TextMeshProUGUI ruleText;
        [SerializeField] private TextMeshProUGUI jobsText;
        [SerializeField] private TextMeshProUGUI decisionsText;

        [Header("Session Controls")]
        [SerializeField] private Button stopButton;
        [SerializeField] private InstanceSelectMenu instanceSelectMenu;

        [Header("Time Controls")]
        [SerializeField] private Slider timeScaleSlider;
        [SerializeField] private TextMeshProUGUI timeScaleValueText;

        // ─────────────────────────────────────────────────────────
        //  Unity Lifecycle
        // ─────────────────────────────────────────────────────────

        /// @brief Syncs the slider's initial value with @c Time.timeScale and
        ///        registers the value-changed listener.
        private void Start()
        {
            if (timeScaleSlider != null)
            {
                timeScaleSlider.value = Time.timeScale;
                timeScaleSlider.onValueChanged.AddListener(OnTimeScaleChanged);
                UpdateScaleText(Time.timeScale);
            }

            if (stopButton != null)
                stopButton.onClick.AddListener(OnStopClicked);
        }

        /// @brief Removes the slider listener to prevent stale callbacks after destruction.
        private void OnDestroy()
        {
            if (timeScaleSlider != null)
            {
                timeScaleSlider.onValueChanged.RemoveListener(OnTimeScaleChanged);
            }

            if (stopButton != null)
                stopButton.onClick.RemoveListener(OnStopClicked);
        }

        /// @brief Stops the active episode and returns to the instance select menu.
        private void OnStopClicked()
        {
            if (SimulationBridge.Instance != null)
                SimulationBridge.Instance.StopEpisode();

            if (instanceSelectMenu != null)
                instanceSelectMenu.Show();
        }

        /// @brief Polls the bridge and refreshes all HUD labels once per frame.
        /// @details Exits early if no active episode is running.
        private void Update()
        {
            if (SimulationBridge.Instance == null || !SimulationBridge.Instance.IsEpisodeActive)
                return;

            if (timeText != null)
                timeText.text = $"SIM TIME: {SimulationBridge.Instance.SimTime:F1}s";

            if (ruleText != null)
                ruleText.text = $"LAST RULE: {SimulationBridge.Instance.LastAppliedRule}";

            if (decisionsText != null)
                decisionsText.text = $"DECISIONS: {SimulationBridge.Instance.DecisionCount}";

            if (jobsText != null
                && SimulationBridge.Instance.JobManager != null
                && SimulationBridge.Instance.JobManager.IsInitialized)
            {
                int totalJobs = SimulationBridge.Instance.JobManager.JobCount;
                int completedJobs = 0;

                foreach (var tracker in SimulationBridge.Instance.JobManager.JobTrackers)
                {
                    if (tracker.State == JobLifecycleState.Complete)
                        completedJobs++;
                }

                jobsText.text = $"JOBS DONE: {completedJobs} / {totalJobs}";
            }
        }

        // ─────────────────────────────────────────────────────────
        //  Time Scale Controls
        // ─────────────────────────────────────────────────────────

        /// @brief Slider callback that applies @p newScale to @c Time.timeScale.
        /// @param newScale  The value chosen by the user on the slider.
        public void OnTimeScaleChanged(float newScale)
        {
            Time.timeScale = newScale;
            UpdateScaleText(newScale);
        }

        /// @brief Refreshes the speed label to reflect the current time scale.
        /// @param scale  The time scale value to display.
        private void UpdateScaleText(float scale)
        {
            if (timeScaleValueText != null)
                timeScaleValueText.text = $"SPEED: {scale:F1}x";
        }
    }
}