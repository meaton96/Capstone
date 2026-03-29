# Unity Simulation API Summary

This document summarizes the core API and functionality for the 3D Factory Simulation, focusing on the communication between the Discrete Event Simulator (DES) and the Unity visual layer.

---

## 1. SimulationBridge.cs (The Orchestrator)
The `SimulationBridge` is the central "brain" of the simulation. It follows the **Gymnasium (OpenAI Gym) pattern**, managing the episode lifecycle and bridging the gap between raw simulation data and Unity visuals.

### Key Structs
* **`DecisionRequest`**: A snapshot of the state when the simulation pauses. Includes `MachineId`, `QueuedJobIds`, and `SimTime`.
* **`StepResult`**: Returned after an action is taken. Contains the `Reward`, `Done` status, and the `NextDecision`.
* **`EpisodeResult`**: Final statistics (Makespan, Optimality Gap, Total Reward) emitted when an episode finishes.

### Core API
* **`StartEpisode(TextAsset json)`**: Resets the DES, loads a Taillard instance, instructs `FactoryLayoutManager` to build the floor, and advances to the first decision point.
* **`Step(int action)`**: 
    * Takes an integer (0-7) representing a Priority Dispatching Rule (PDR).
    * Applies the decision to the DES.
    * Advances time until the next conflict or the end of the episode.
    * Calculates and returns the reward.
* **`RunEpisodeWithFixedRule(TextAsset json, int ruleIndex)`**: A synchronous version of the simulation used for fast batch validation without visual overhead.

### Action Space Mapping
The Bridge maps integer actions to specific **Dispatching Rules**:
| Index | Rule | Index | Rule |
| :--- | :--- | :--- | :--- |
| **0** | SPT-SMPT | **4** | LRT-SMPT |
| **1** | SPT-SRWT | **5** | LRT-MMUR |
| **2** | LPT-MMUR | **6** | SRT-SMPT |
| **3** | SRT-SRWT | **7** | SDT-SRWT |

---

## 2. FactoryLayoutManager.cs (Spatial Logic)
Responsible for the physical arrangement of machines and providing spatial data to the RL observation space.

### Core API
* **`BuildFloor(DESSimulator simulator)`**: The primary entry point. Spawns `MachineVisual` prefabs based on the number of machines in the DES.
* **`GetMachineVisual(int machineId)`**: Returns the specific visual component for a machine ID. Used by the Bridge to trigger flashes or UI updates.
* **`DistanceMatrix` / `DistanceMatrixFlat`**: Computes an $N \times N$ matrix of Euclidean distances between machines.
    * `DistanceMatrixFlat` provides a normalized 64-D (8x8) vector specifically for the RL agent's observation space.
* **`SetCustomLayout(Vector3[] positions)`**: Allows external scripts to define specific machine coordinates instead of using the default grid logic.

---

## 3. MachineVisual.cs (Visual State)
A component attached to each machine prefab that manages its own UI, animations, and material properties.

### Core API
* **`Initialise(int id, Machine coreMachineRef)`**: Links the 3D object to its logical counterpart in the DES.
* **`SetState(MachineState newState)`**: Updates the machine's color (e.g., Green for Idle, Yellow for Busy, Red for Failed).
* **`BeginOperation / CompleteOperation`**: High-level wrappers that start/stop the overhead progress bar and log history.
* **`UpdateProgress(float currentSimTime)`**: Manually syncs the overhead UI slider with the current simulation clock.
* **`EnqueueJob / DequeueJob`**: Manually handles the visual "slots" behind the machine where jobs wait in line.
* **`RecordDecisionPoint(...)`**: Triggers a visual "flash" (Step 8) and logs the decision details to the machine's internal history log for debugging.

---

## Workflow Diagram: The Decision Loop


1.  **Advance**: The Bridge runs the DES until a machine has multiple jobs in its queue.
2.  **Request**: `OnDecisionRequired` event is fired with a `DecisionRequest`.
3.  **Action**: An RL Agent or `RunFixedRuleCoroutine` provides an action index.
4.  **Apply**: The Bridge calls `Step(action)`, which tells the machine which job to process next.
5.  **Visualize**: The Bridge captures all events that happened during the "jump" in time and tells `MachineVisuals` to play back the operations.