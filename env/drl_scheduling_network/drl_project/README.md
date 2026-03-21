# DRL Job-Shop Scheduling Network

Standalone PyTorch implementation of the multi-modal Actor-Critic architecture
for dynamic job-shop scheduling with AGV routing. Designed to be developed and
tested independently of the Unity simulation.

## Architecture (from thesis diagram)

```
State Space                          Encoder                    Heads
─────────────────────────────────────────────────────────────────────────
Factory Grid   (3×64×64)  ──► CNN-SPPF  ──► 256-D ─┐
Sched Matrix   (3×100×40) ──► CNN-SPPF  ──► 128-D ─┤
Global Scalars (10-D)      ──► MLP      ──►  32-D ─┼─► Concat (464-D)
Distance Mat   (64-D)      ──► MLP      ──►  32-D ─┤   │
Event Flags    (6-D)       ──► MLP      ──►  16-D ─┘   ▼
                                              Domain Randomization
                                                    │
                                              Fusion Head (256-D)
                                                  ┌─┴─┐
                                              Actor   Critic
                                              (8 PDR)  (V)
```

## Action Space: 8 Composite Priority Dispatching Rules

| Index | Job Rule | Machine Rule |
|-------|----------|-------------|
| 0     | SPT      | SMPT        |
| 1     | SPT      | SRWT        |
| 2     | LPT      | MMUR        |
| 3     | SRT      | SRWT        |
| 4     | LRT      | SMPT        |
| 5     | LRT      | MMUR        |
| 6     | SRT      | SMPT        |
| 7     | SDT      | SRWT        |

## Project Structure

```
drl_project/
├── config.py                  # All hyperparams & dimensions
├── train.py                   # PPO training loop
├── rollout_buffer.py          # GAE rollout storage
├── models/
│   ├── encoder.py             # CNN-SPPF + MLP encoders (5 modalities)
│   ├── actor_critic.py        # Fusion head + Actor + Critic
│   └── network.py             # Full SchedulingNetwork end-to-end
├── env/
│   └── placeholder_env.py     # Synthetic Gym env (swap for Unity)
└── tests/
    └── test_architecture.py   # Shape, gradient, checkpoint tests
```

## Quick Start

```bash
# Run all tests
python tests/test_architecture.py

# Short training run (verify pipeline)
python train.py --total-timesteps 10000 --num-envs 4

# Full training
python train.py --total-timesteps 1000000 --num-envs 8
```

## Unity Integration Points

When the Unity simulation is ready, you need to:

1. **Replace `PlaceholderSchedulingEnv`** with a wrapper that:
   - Receives observations from Unity via your connector (gRPC, socket, etc.)
   - Packs them into the same dict format: `factory_grid`, `sched_matrix`,
     `global_scalars`, `distance_matrix`, `event_flags`
   - Sends the selected PDR action index back to Unity
   - Returns the reward from Unity's DES queue metrics

2. **The observation dict contract** (everything else stays the same):
   ```python
   obs = {
       "factory_grid":    np.float32, shape (3, 64, 64),   # Machine/Job/AGV
       "sched_matrix":    np.float32, shape (3, 100, 40),  # n×2m×3
       "global_scalars":  np.float32, shape (10,),          # normalized
       "distance_matrix": np.float32, shape (64,),          # 8×8 flat
       "event_flags":     np.float32, shape (6,),           # binary
   }
   ```

3. **Reward signal** should come from Unity's simulation metrics:
   - Primary: negative makespan (Cmax)
   - Shaping: throughput, tardiness, queue lengths

## Key Design Decisions

- **SPPF over SPP**: Faster with sequential max-pools at different kernel sizes
- **Domain Randomization**: Dropout + Gaussian noise in the fusion layer to
  facilitate sim-to-real transfer
- **LayerNorm everywhere**: More stable than BatchNorm for RL
- **SiLU activation**: Smooth, works well with deep RL
- **Separate actor/critic heads**: Shared encoder but independent final layers

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- NumPy
- Gymnasium
