# Project Context: DRL-Based Dynamic Flexible Job Shop Scheduling (DFJSP)

This project addresses the **sim-to-real gap** in industrial scheduling by integrating high-fidelity 3D physics with Deep Reinforcement Learning. Unlike traditional discrete event simulators (DES) that treat factories as abstract logical nodes, this framework utilizes a game engine to model continuous physical constraints.

## 1. The Simulation Environment (Unity)

* **Engine:** Built in **Unity 6.3** utilizing native **PhysX** and **NavMesh** pathfinding.
* **Emergent Physicality:** AGV kinematics, robotic acceleration limits, battery depletion, and spatial congestion are emergent properties of the environment rather than static parameters.
* **Logistics Optimization:** Implements **predictive AGV pre-dispatch**, where an idle vehicle is sent to a machine dock before an operation finishes to mask transport latency.
* **Traffic Management:** Uses a specialized manager for **deadlock-free directed zone routing** to handle fleet movement on the factory floor.

## 2. DRL Formulation

* **Algorithm:** **Proximal Policy Optimization (PPO)** implemented via Stable-Baselines3 and PyTorch 2.1.
* **Reward Function:** A dense reward defined as the **negative normalized makespan delta**:

$$\text{Reward}_t = -\frac{\Delta M_t}{N_{ops}}$$

where $\Delta M_t$ is the increase in makespan estimate and $N_{ops}$ is the number of remaining operations.

* **Action Space (Composite PDRs):** The agent selects from **8 composite Priority Dispatching Rules (PDRs)** to manage combinatorial complexity:
    * **Throughput Focused:** SPT (Shortest Processing Time) paired with SMPT (Shortest Machine Processing Time).
    * **Load Balancing:** LPT (Longest Processing Time) or LRT (Longest Remaining Time) paired with MMUR (Minimum Machine Utilization).
    * **Fairness Focused:** FIFO (First-Come-First-Served, oldest arrival first) paired with SRWT (Shortest Remaining Work).

## 3. Multimodal Neural Architecture

To handle both spatial geometry and scheduling logic, the system fuses five parallel data streams:

| Stream | Format | Description |
| :--- | :--- | :--- |
| **Spatial Occupancy** | $64 \times 64 \times 3$ | A 3-channel tensor mapping coordinates of machines, jobs, and AGVs. |
| **Scheduling Matrix** | $n \times 2m \times 3$ | Tracks processing times and completion flags for operations. |
| **Global Scalars** | 10-D | Encodes overall factory utilization metrics. |
| **Distance Matrix** | 64-D | Flattened pairwise physical distances between nodes. |
| **Event Flags** | 6-D | Signals discrete state changes (e.g., task completion). |

* **Size-Agnostic Inference:** The spatial grid and scheduling matrix are processed by **CNN-SPPF** (Spatial Pyramid Pooling-Fast) encoders, allowing the model to generalize to factory configurations and job counts not seen during training.

## 4. Key Findings & Baselines

* **Benchmarking:** Heuristic floors were established using **Taillard** and **Brandimarte (MK01-MK15)** instances.
* **Rule Sensitivity:** On procedurally generated random instances, PDR choice contributed less than 2% to makespan when AGV transport was the primary bottleneck.
* **Adaptive Necessity:** On structured benchmarks with high contention, no single PDR dominated across all instances, confirming that an adaptive DRL policy is required to select the optimal rule for specific structural contexts.

## 5. Planned Stochastic Extensions

Future work involves testing policy robustness against mid-episode disruptions:

* **Machine/AGV Failures:** Modeled using a **two-parameter Weibull distribution** ($k=1.5$) to simulate realistic wear-out dynamics.
* **Repair Times:** Modeled with a **log-normal distribution** to capture heavy-tailed maintenance data.
* **Dynamic Arrivals:** New jobs arriving via a **homogeneous Poisson process** to shift factory utilization unpredictably.