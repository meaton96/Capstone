# DRL Job-Shop Scheduling Network

A deep reinforcement learning system for job-shop scheduling in a simulated factory environment. The agent learns to select composite Priority Dispatching Rules (PDR) to minimise makespan using Proximal Policy Optimisation (PPO).

## Architecture Overview

The network processes five sensor modalities from the factory floor, fuses them into a shared representation, and outputs both a scheduling policy and a state-value estimate.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Observation Dict                             │
│                                                                     │
│  factory_grid    sched_matrix    global_scalars  distances  events  │
│  (3, 64, 64)    (3, 100, 40)    (10,)           (64,)      (6,)   │
└──────┬───────────────┬──────────────┬──────────────┬─────────┬──────┘
       │               │              │              │         │
       ▼               ▼              ▼              ▼         ▼
  ┌─────────┐    ┌─────────┐    ┌─────────┐   ┌─────────┐ ┌────────┐
  │CNN-SPPF │    │CNN-SPPF │    │   MLP   │   │   MLP   │ │  MLP   │
  │  256-D  │    │  128-D  │    │  32-D   │   │  32-D   │ │  16-D  │
  └────┬────┘    └────┬────┘    └────┬────┘   └────┬────┘ └───┬────┘
       │              │              │              │          │
       └──────────────┴──────────────┴──────────────┴──────────┘
                                     │
                              Concatenate (464-D)
                                     │
                                     ▼
                              ┌─────────────┐
                              │ Fusion Head │
                              │   256-D     │
                              └──────┬──────┘
                                     │
                          ┌──────────┴──────────┐
                          ▼                     ▼
                   ┌─────────────┐       ┌─────────────┐
                   │    Actor    │       │   Critic    │
                   │  8 actions  │       │    V(s)     │
                   └─────────────┘       └─────────────┘
```

## Training Pipeline

Training follows a standard PPO collect-update cycle. Data is generated on-the-fly by the agent interacting with parallel environments — there is no static dataset.

```
┌──────────────────────────────────────────────────────────────┐
│                      Training Loop                           │
│                                                              │
│  1. COLLECT   Agent steps through N parallel environments    │
│               for rollout_length steps, storing transitions  │
│               (obs, action, log_prob, reward, value, done)   │
│               in the RolloutBuffer.                          │
│                                                              │
│  2. COMPUTE   GAE walks backwards through the buffer to      │
│               calculate advantages and discounted returns.   │
│                                                              │
│  3. UPDATE    Mini-batches are shuffled and fed through the  │
│               PPO clipped surrogate objective for num_epochs │
│               passes. Loss = policy + value + entropy.       │
│                                                              │
│  4. REPEAT    Buffer is reset, next rollout begins.          │
└──────────────────────────────────────────────────────────────┘
```

Each observation is treated as an independent, self-contained state (Markov assumption). The architecture is purely feedforward with no recurrent layers, so temporal ordering within the buffer is irrelevant and mini-batches are freely shuffled.

## Sensor Corruption (Domain Randomisation)

To improve sim-to-real transfer, a `SensorCorruptionWrapper` applies per-modality corruption to raw observations before they enter the encoder. This simulates real-world sensor degradation:

| Modality | Dropout (sensor failure) | Noise (measurement error) | Rationale |
|---|---|---|---|
| `factory_grid` | 5% | σ = 0.02 | IoT sensor failures, AGV localisation jitter |
| `sched_matrix` | 2% | σ = 0.01 | MES data loss, processing-time estimation error |
| `global_scalars` | 1% | σ = 0.01 | Aggregated KPIs have some inherent smoothing |
| `distance_matrix` | 0% | σ = 0.01 | Fixed layout, minor measurement jitter |
| `event_flags` | 0% | 0 | Discrete binary states, not continuous readings |

Corruption is applied at the observation level (not inside the network) because it models physical phenomena. It can be toggled off for evaluation via `wrapper.set_enabled(False)`.

## Project Structure

```
├── config.py                  # All dataclass configs (env, encoder, fusion, PPO)
├── train.py                   # PPO training loop entry point
├── rollout_buffer.py          # Transition storage + GAE computation
│
├── models/
│   ├── encoder.py             # Multi-modal encoders (CNN-SPPF, MLP)
│   ├── actor_critic.py        # FusionHead, ActorHead, CriticHead, ActorCritic
│   └── network.py             # SchedulingNetwork (end-to-end wrapper)
│
├── env/
│   ├── placeholder_env.py     # Synthetic Gymnasium env + vectorised wrapper
│   └── sensor_corruption.py   # Observation corruption wrapper
│
└── tests/
    └── test_architecture.py   # Full test suite (shapes, gradients, env, GAE, PPO)
```

## Action Space

The agent selects from 8 composite dispatching rules, each pairing a job-priority rule with a machine-assignment rule:

| Index | Action | Job Rule | Machine Rule |
|---|---|---|---|
| 0 | SPT-SMPT | Shortest Processing Time | Shortest Machine Processing Time |
| 1 | SPT-SRWT | Shortest Processing Time | Shortest Remaining Work Time |
| 2 | LPT-MMUR | Longest Processing Time | Most Machine Utilisation Rate |
| 3 | SRT-SRWT | Shortest Remaining Time | Shortest Remaining Work Time |
| 4 | LRT-SMPT | Longest Remaining Time | Shortest Machine Processing Time |
| 5 | LRT-MMUR | Longest Remaining Time | Most Machine Utilisation Rate |
| 6 | SRT-SMPT | Shortest Remaining Time | Shortest Machine Processing Time |
| 7 | SDT-SRWT | Shortest Due Time | Shortest Remaining Work Time |

## Observation Space

| Component | Shape | Description |
|---|---|---|
| `factory_grid` | (3, 64, 64) | Spatial occupancy grid — channel 0: machines, channel 1: jobs (with intensity), channel 2: AGV positions |
| `sched_matrix` | (3, 100, 40) | Scheduling matrix image — channel 0: machine assignments, channel 1: processing times, channel 2: reserved |
| `global_scalars` | (10,) | Normalised features: sim time, makespan lower bound, jobs waiting/active, failures, throughput, queue length, completion rate, time since last event, overtime flag |
| `distance_matrix` | (64,) | Flattened 8×8 pairwise machine distances |
| `event_flags` | (6,) | Binary indicators for discrete scheduling events |

All values are normalised to [0, 1].

## Quick Start

**Training** (50k steps on CPU for testing):

```bash
python train.py --total-timesteps 50000 --num-envs 4 --device cpu
```

**All arguments:**

```bash
python train.py \
    --total-timesteps 1000000 \
    --num-envs 8 \
    --rollout-length 128 \
    --batch-size 64 \
    --lr 3e-4 \
    --device cuda
```

A checkpoint is saved to `checkpoint.pt` at the end of training, containing model weights, optimiser state, step count, and the full config.

**Running tests:**

```bash
python -m pytest tests/test_architecture.py -v
# or without pytest:
python tests/test_architecture.py
```

## PPO Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `lr` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE smoothing parameter |
| `clip_epsilon` | 0.2 | PPO surrogate clipping range |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `value_coef` | 0.5 | Value loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping norm |
| `num_epochs` | 4 | PPO epochs per rollout |
| `batch_size` | 64 | Mini-batch size |
| `rollout_length` | 128 | Steps per rollout |
| `num_envs` | 8 | Parallel environments |

## Placeholder Environment

The current environment generates synthetic observations and rewards so the full pipeline can be validated end-to-end before the Unity simulation is ready. The reward signal simulates makespan reduction with action-dependent quality and noise, giving the agent a learnable gradient.

When the Unity sim is available, replace `PlaceholderSchedulingEnv` with a wrapper that translates Unity observations into the same dict format. No changes to the network, buffer, or training loop are required — the interface contract (observation dict → action int → reward float) stays the same. The `SensorCorruptionWrapper` can either be kept as an outer wrapper or its corruption logic can be moved into the Unity sim itself.

## Documentation

All source files are annotated for Doxygen. Generate HTML documentation with:

```bash
doxygen Doxyfile
```
