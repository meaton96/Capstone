using UnityEngine;
using TMPro; // Make sure you have TextMeshPro imported!
using Assets.Scripts.Simulation;

namespace Assets.Scripts.UI
{
    public class SimulationHUD : MonoBehaviour
    {
        [Header("UI Text References")]
        [SerializeField] private TextMeshProUGUI timeText;
        [SerializeField] private TextMeshProUGUI ruleText;
        [SerializeField] private TextMeshProUGUI jobsText;
        [SerializeField] private TextMeshProUGUI decisionsText;

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