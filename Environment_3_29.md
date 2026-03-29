This summary provides an overview of the DRL Job-Shop Scheduling architecture. The system uses Proximal Policy Optimization (PPO) to select composite Priority Dispatching Rules (PDRs) based on multi-modal factory observations.

---

## 🏗️ System Architecture Overview

The architecture follows a modular "Encoder-Fusion-Head" pipeline designed to process diverse data types from a factory environment.

| Component | Responsibility |
| :--- | :--- |
| **Encoders** | Process spatial grids, scheduling matrices, and scalar metadata into embeddings. |
| **Fusion Head** | Concatenates and projects all embeddings into a shared 256-D latent space. |
| **Actor-Critic** | Predicts the best PDR action (Actor) and estimates state value (Critic). |
| **Environment** | A Gymnasium-based simulation with support for sensor degradation. |

---

## ⚙️ Configuration (`config.py`)

The system is driven by `dataclasses` that define the dimensions and hyperparameters. Key configurations include:

* **`EnvConfig`**: Defines the $64 \times 64$ spatial grid and job/machine ranges (e.g., 15–20 machines, 50–100 jobs).
* **`EncoderConfig`**: Sets the embedding sizes. The default total concatenated dimension is **464-D**.
* **`PDR_ACTIONS`**: A list of 8 composite rules (e.g., `SPT-SMPT`, `LPT-MMUR`) that the agent chooses from.

---

## 🧠 Neural Network Modules

### 1. Multi-Modal Encoders (`encoder.py`)
This module handles five distinct observation streams:
* **`CNNSPPFEncoder`**: Uses a CNN backbone with **Spatial Pyramid Pooling – Fast (SPPF)** to extract multi-scale features from the `factory_grid` and `sched_matrix`.
* **`MLPEncoder`**: Processes 1-D vectors like `global_scalars`, `distance_matrix`, and `event_flags`.
* **`MultiModalEncoder`**: The master class that runs all sub-encoders and concatenates their outputs.

### 2. Actor-Critic Heads (`actor_critic.py`)
* **`FusionHead`**: A two-layer MLP with **LayerNorm** and **SiLU** activations that compresses the 464-D input to 256-D.
* **`ActorCritic`**: Contains the `actor` (outputs 8 action logits) and `critic` (outputs 1 scalar value).
    * **`act()`**: Used during rollouts to sample actions.
    * **`evaluate()`**: Used during PPO updates to calculate log-probs and entropy.

### 3. The Integrated Network (`network.py`)
The **`SchedulingNetwork`** class acts as the primary API for the agent.
* **Usage**: 
    ```python
    net = SchedulingNetwork()
    action, log_prob, value = net.act(obs_dict)
    ```

---

## 🏭 Environment & Robustness

### Placeholder Environment (`placeholder_env.py`)
Since the Unity simulation might not always be available, the **`PlaceholderSchedulingEnv`** generates synthetic observations and rewards.
* **`VectorizedPlaceholderEnv`**: Runs multiple environments in parallel to speed up data collection.

### Sensor Corruption (`sensor_corruption.py`)
To ensure the model works in "messy" real-world conditions, the **`SensorCorruptionWrapper`** injects:
* **Dropout**: Randomly zeroes out sensors (IoT failure).
* **Gaussian Noise**: Adds jitter to measurements (imprecise AGV localization).

---

## 📈 Training Logic (`train.py` & `rollout_buffer.py`)

The training follows the standard PPO cycle: **Collect → Compute GAE → Update**.

### Rollout Management
The **`RolloutBuffer`** stores transitions and implements **Generalized Advantage Estimation (GAE)**.
* **`compute_gae()`**: Iterates backward through time to calculate advantages ($\hat{A}_t$) and returns.
* **`get_batches()`**: Flattens and shuffles data into mini-batches for the optimizer.

### The Training Loop
The **`train()`** function in `train.py` orchestrates the process:
1.  **Collection**: Runs the agent in parallel environments for `rollout_length` steps.
2.  **Optimization**: Performs `num_epochs` of PPO updates.
3.  **Loss Calculation**: 
    $$Loss = L_{CLIP} + c_1 L_{VF} + c_2 S$$
    *(Where $L_{CLIP}$ is the surrogate policy loss, $L_{VF}$ is value MSE, and $S$ is the entropy bonus).*

---

## 🚀 Usage Guide

To start training with the default settings (4 parallel environments, CPU):

```bash
python train.py --total-timesteps 100000 --num-envs 4 --device cpu
```

**Key Parameters:**
* `--lr`: Learning rate (default `3e-4`).
* `--batch-size`: Size of the PPO mini-batches (default `32`).
* `--rollout-length`: How many steps to take before updating (default `64`).