#!/usr/bin/env bash
##############################################################################
#  run_train_warmup.sh
#
#  PPO training on the compound_scenario chain, with --random-warmup: each
#  episode's floor runs forward under SPT_SMPT to a random phase-boundary
#  offset (drawn from that episode's own seed) before the RL agent takes
#  over, then the agent gets --episode-duration-seconds of real decisions
#  from that mid-cycle state. Combines LONG's full-phase-cycle exposure
#  (why LONG generalized 11x more consistently than SHORT in the prior
#  episode-length comparison, per episode-length-comparison-bimodal-
#  generalization) with SHORT's cheap per-episode decision cost (SHORT
#  always started at t=0, phases 1-3 only, which is the bias this is meant
#  to fix) -- so this run should get many more gradient updates per hour
#  than LONG did, while (unlike SHORT) still training on every phase.
#
#  Needs a Unity build newer than the C# warm-up changes (StochasticConfig.
#  WarmupSeconds, FactoryOrchestrator.InWarmup, ScenarioLoader.
#  ReadDispatchingRule) -- rebuild from the Unity editor first if unsure;
#  compare linux_server/capstone_Data/Managed/Simulation.dll's mtime against
#  your last C# edit.
#
#  Usage:
#    ./run_train_warmup.sh                  # defaults below
#    TOTAL_TIMESTEPS=1000000 ./run_train_warmup.sh
#    RUN_ID=ep_warmup02 TOTAL_TIMESTEPS=250000 ./run_train_warmup.sh
#
#  Stop (Ctrl-C) and resume later -- TOTAL_TIMESTEPS is always the target
#  cumulative step count, so raise it on resume or it'll say there's nothing
#  left to do:
#    RESUME_FROM=results/ep_warmup01/checkpoint.pt TOTAL_TIMESTEPS=1000000 \
#        ./run_train_warmup.sh
#
#  Monitor while it runs:
#    tail -f results/<run-id>/Player-0.log      # Unity side, per-episode lines
#    tensorboard --logdir results                # losses, episode metrics, reward terms
##############################################################################

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# ── Tunables (override via env var, e.g. `TOTAL_TIMESTEPS=1000000 ./run_train_warmup.sh`) ──
RUN_ID="${RUN_ID:-ep_warmup01}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
EPISODE_DURATION_SECONDS="${EPISODE_DURATION_SECONDS:-2700}"   # matches ep_short's own cap, for a direct comparison
WARMUP_RULE="${WARMUP_RULE:-SPT_SMPT}"                          # best-PDR floor state, matches the eval baseline
TRAIN_SEED="${TRAIN_SEED:-0}"                                   # matches ep_short/ep_long's own --train-seed 0 (same init weights)
NUM_ENVS="${NUM_ENVS:-4}"
DEVICE="${DEVICE:-cuda}"
SAVE_EVERY="${SAVE_EVERY:-100}"                                  # in PPO *updates*, not steps -- 100 updates x
                                                                  # NUM_ENVS x rollout_length(64) = 25,600 steps/
                                                                  # checkpoint, matching ep_short/ep_long's cadence
RESUME_FROM="${RESUME_FROM:-}"                                   # path to a checkpoint .pt to continue from

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "[ERROR] .venv not found at repo root — see native-linux-python-setup." >&2
    exit 1
fi
if [[ ! -x "linux_server/capstone.x86_64" ]]; then
    echo "[ERROR] linux_server/capstone.x86_64 not found or not executable." >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "========================================================================"
echo "[Launcher] run-id:                   $RUN_ID"
echo "[Launcher] total-timesteps:          $TOTAL_TIMESTEPS"
echo "[Launcher] episode-duration-seconds: $EPISODE_DURATION_SECONDS (post-warmup window)"
echo "[Launcher] warmup-dispatching-rule:  $WARMUP_RULE"
echo "[Launcher] train-seed:               $TRAIN_SEED"
echo "[Launcher] num-envs:                 $NUM_ENVS"
echo "[Launcher] device:                   $DEVICE"
echo "[Launcher] results/$RUN_ID/ will hold Player logs, checkpoints (every"
echo "[Launcher] $SAVE_EVERY updates = $(( SAVE_EVERY * NUM_ENVS * 64 )) steps), TensorBoard events, and episodes.csv."
if [[ -n "$RESUME_FROM" ]]; then
    echo "[Launcher] resuming from:            $RESUME_FROM"
fi
echo "========================================================================"

RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
    RESUME_ARGS=(--resume-from "$RESUME_FROM")
fi

python env/train.py \
    --unity --unity-path linux_server/capstone.x86_64 --no-graphics \
    --scenario-generator compound --random-warmup --warmup-dispatching-rule "$WARMUP_RULE" \
    --episode-duration-seconds "$EPISODE_DURATION_SECONDS" \
    --reward-spec env/config/rewards/flow_time.json \
    --train-seed "$TRAIN_SEED" \
    --num-envs "$NUM_ENVS" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --save-every "$SAVE_EVERY" \
    --device "$DEVICE" \
    --run-id "$RUN_ID" \
    "${RESUME_ARGS[@]}"
