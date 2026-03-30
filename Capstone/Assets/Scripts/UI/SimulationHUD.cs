using UnityEngine;
using TMPro;
using Assets.Scripts.Simulation;
using UnityEngine.UI;

namespace Assets.Scripts.UI
{
    public class SimulationHUD : MonoBehaviour
    {
        [Header("UI Text References")]
        [SerializeField] private TextMeshProUGUI timeText;
        [SerializeField] private TextMeshProUGUI ruleText;
        [SerializeField] private TextMeshProUGUI jobsText;
        [SerializeField] private TextMeshProUGUI decisionsText;
        [Header("Time Controls")]
        [SerializeField] private Slider timeScaleSlider;
        [SerializeField] private TextMeshProUGUI timeScaleValueText;

        private void Start()
        {
            // Set up the slider if it has been assigned in the Inspector
            if (timeScaleSlider != null)
            {
                // Sync the slider's starting value with the current game time scale
                timeScaleSlider.value = Time.timeScale;

                // Add a listener so the method is called automatically when you drag the slider
                timeScaleSlider.onValueChanged.AddListener(OnTimeScaleChanged);

                // Initialize the text label
                UpdateScaleText(Time.timeScale);
            }
        }
        private void OnDestroy()
        {
            if (timeScaleSlider != null)
            {
                timeScaleSlider.onValueChanged.RemoveListener(OnTimeScaleChanged);
            }
        }
        private void UpdateScaleText(float scale)
        {
            if (timeScaleValueText != null)
            {
                timeScaleValueText.text = $"SPEED: {scale:F1}x";
            }
        }

        // This method fires whenever the slider is moved
        public void OnTimeScaleChanged(float newScale)
        {
            // Unity's built-in way to speed up, slow down, or pause time
            Time.timeScale = newScale;
            UpdateScaleText(newScale);
        }

        private void Update()
        {
            // Don't update if the bridge isn't ready or the episode is over
            if (SimulationBridge.Instance == null || !SimulationBridge.Instance.IsEpisodeActive)
                return;

            // 1. Current Simulation Time (Makespan)
            if (timeText != null)
            {
                timeText.text = $"SIM TIME: {SimulationBridge.Instance.SimTime:F1}s";
            }

            // 2. Last Applied Dispatching Rule
            if (ruleText != null)
            {
                ruleText.text = $"LAST RULE: {SimulationBridge.Instance.LastAppliedRule}";
            }

            // 3. Decisions Made
            if (decisionsText != null)
            {
                decisionsText.text = $"DECISIONS: {SimulationBridge.Instance.DecisionCount}";
            }

            // 4. Job Progress
            if (jobsText != null && SimulationBridge.Instance.JobManager != null && SimulationBridge.Instance.JobManager.IsInitialized)
            {
                int totalJobs = SimulationBridge.Instance.JobManager.JobCount;
                int completedJobs = 0;

                // Quickly tally up how many jobs are totally finished
                foreach (var tracker in SimulationBridge.Instance.JobManager.JobTrackers)
                {
                    if (tracker.State == JobLifecycleState.Complete)
                    {
                        completedJobs++;
                    }
                }

                jobsText.text = $"JOBS DONE: {completedJobs} / {totalJobs}";
            }
        }
    }
}