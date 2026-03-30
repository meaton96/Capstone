// using UnityEngine;

// namespace Assets.Scripts.Simulation
// {
//     // ═════════════════════════════════════════════════════════════
//     //  Supporting types
//     // ═════════════════════════════════════════════════════════════

//     /// @brief Snapshot of the DES state at a decision point, presented to the
//     /// agent (or fixed rule) so it can choose an action.
//     [Serializable]
//     public struct DecisionRequest
//     {
//         /// @brief ID of the machine that just became free and has a non-empty queue.
//         public int MachineId;

//         /// @brief Current simulation time.
//         public double SimTime;

//         /// @brief IDs of jobs waiting in the machine's queue.
//         public int[] QueuedJobIds;

//         /// @brief Processing times of the queued operations (parallel to QueuedJobIds).
//         public double[] QueuedDurations;

//         /// @brief How many decisions have been made so far this episode.
//         public int DecisionIndex;

//         /// @brief Total jobs in the instance.
//         public int TotalJobs;

//         /// @brief Jobs completed so far.
//         public int CompletedJobs;



//         // TODO Phase 2: Add full observation dict (grid, sched_matrix, scalars,
//         //               distances, event_flags) once ObservationBuilder exists.
//     }

//     /// @brief Returned by @ref SimulationBridge.Step after the agent's action is applied.
//     [Serializable]
//     public struct StepResult
//     {
//         /// @brief Reward signal for this step.
//         public float Reward;

//         /// @brief True if the episode is finished (all jobs complete).
//         public bool Done;

//         /// @brief The next decision request, if not done. Check @ref Done first.
//         public DecisionRequest NextDecision;

//         /// @brief Current makespan estimate (max end-time so far).
//         public double CurrentMakespan;

//         /// @brief Number of operations completed so far.
//         public int OperationsCompleted;
//     }

//     /// @brief Final episode statistics, emitted via @ref SimulationBridge.OnEpisodeFinished.
//     [Serializable]
//     public struct EpisodeResult
//     {
//         public string InstanceName;
//         public string RuleName;
//         public double Makespan;
//         public double OptimalMakespan;
//         public int TotalJobs;
//         public int TotalOperations;
//         public int CompletedJobs;
//         public int DecisionPoints;
//         public double TotalReward;
//         public int[] PerMachineDecisions;

//         /// @brief Percentage gap from optimal: (makespan - optimal) / optimal × 100.
//         public double OptimalityGap =>
//             OptimalMakespan > 0
//                 ? (Makespan - OptimalMakespan) / OptimalMakespan * 100.0
//                 : 0;
//     }
// }