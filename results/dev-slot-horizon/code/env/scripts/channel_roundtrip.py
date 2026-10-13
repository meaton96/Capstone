"""
@file channel_roundtrip.py
@brief Manual side-channel round trip against a real player: config from Python -> Unity -> telemetry back.

@details Sends CONFIG on EpisodeConfigChannel, plays with action 0 on both heads, and passes once an
EpisodeTelemetryChannel payload comes back for an episode built from CONFIG (its job count, machine count and
stochastic tag). The episode already running when Python connects keeps the player's previous config, so CONFIG
takes effect from the next episode Unity starts.

This launches a Unity player, so it is a script, not a pytest test: it lives outside env/tests (pytest.ini
collects env/tests) and does nothing on import. It closes only the player it started. Never pkill
capstone.x86_64 instead: the other players on the machine are usually experiments.

From the repo root:
    .venv/bin/python env/scripts/channel_roundtrip.py --unity-path linux_server/capstone.x86_64 \\
        --log-file results/dev-channel-roundtrip/Player.log
Exit code 0 = pass, 1 = fail. If Unity rejects the config it stops the player; the reason is in the player log.
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
from mlagents_envs.exception import UnityException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ACTION_BRANCHES  # noqa: E402
from env_wrappers.unity_env import UnitySchedulingEnv  # noqa: E402

## @brief Generated-config message (EpisodeConfigChannel.send_config schema, channels/config_schema.py): 5 jobs on
##        10 machines, 4 AGVs, machine failures on. Keep it a plain literal: test_config_safety.py validates it by
##        parsing this file, so that pytest never imports a script that launches a player.
CONFIG = {
    "name": "channel_roundtrip",
    "seed": 99,
    "jobCount": 5,
    "machinesPerType": 2,
    "machineTypes": ["Mill", "Lathe", "Weld", "Inspect", "Assemble"],
    "minProcTime": 5.0,
    "maxProcTime": 15.0,
    "minOpsPerJob": 2,
    "maxOpsPerJob": 3,
    "agvCount": 4,
    "stochastic": {
        "machineFailuresEnabled": True,
        "weibullK": 1.5,
        "weibullLambda": 2000.0,
        "repairLogMu": 2.0,
        "repairLogSigma": 0.3,
        "agvFailuresEnabled": False,
        "dynamicArrivalsEnabled": False,
    },
}

## @brief Action 0 of each head. Unity never masks it (SchedulingAgent.MaskAllButFirst), so it is always valid.
ACTION = np.zeros(len(ACTION_BRANCHES), dtype=np.int64)

## @brief Seconds to wait for the player to start, and for it to quit on close before it is killed.
TIMEOUT_WAIT = 60


def is_config_episode(result: dict) -> bool:
    """@brief Whether a telemetry result comes from an episode built from CONFIG."""
    return (result.get("jobCount") == CONFIG["jobCount"]
            and result.get("machineCount") == len(CONFIG["machineTypes"]) * CONFIG["machinesPerType"]
            and result.get("stochasticTag") == "mf")


def run(env: UnitySchedulingEnv, max_episodes: int, timeout_s: float) -> bool:
    """@brief Send CONFIG and play until telemetry arrives for an episode built from it. @return True on pass."""
    env.send_config(CONFIG)
    env.reset()   # delivers CONFIG
    print(f"Config sent. The player log should show: [Bridge] Applied Python config: {CONFIG['name']}")

    deadline = time.monotonic() + timeout_s
    episodes, decisions = 0, 0
    while episodes < max_episodes:
        if time.monotonic() > deadline:
            print(f"FAIL: timed out after {timeout_s:.0f} s ({episodes} episode(s) finished)")
            return False
        _, _, done, info = env.step(ACTION)
        decisions += 1
        if not done:
            continue

        episodes += 1
        payload = info["telemetry"]
        if payload is None:
            print(f"FAIL: episode {episodes} ended without a telemetry payload")
            return False
        result = payload.get("result") or {}
        print(f"Episode {episodes} ended after {decisions} decisions: jobs={result.get('jobCount')} "
              f"machines={result.get('machineCount')} stochastic={result.get('stochasticTag')} "
              f"rule={result.get('ruleName')} makespan={result.get('makespan')} "
              f"events={len(payload.get('events') or [])} config_hash={result.get('configHash')}")
        decisions = 0
        if is_config_episode(result):
            print("PASS: telemetry came back for an episode built from the sent config.")
            return True

    print(f"FAIL: no episode built from the sent config in {max_episodes} episodes")
    return False


def close_own_player(env: UnitySchedulingEnv):
    """@brief Close the player this script started, and only that one.

    @details env.close() asks the player to quit and kills it after TIMEOUT_WAIT seconds. If close() itself is
    interrupted (a second Ctrl-C), the player's own process handle is killed here instead.
    """
    process = getattr(env.env, "_process", None)   # mlagents UnityEnvironment's Popen handle for its player
    try:
        env.close()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Side-channel round trip against a Unity player (launches one player; see the module docstring)")
    parser.add_argument("--unity-path", default="linux_server/capstone.x86_64",
                        help="Player executable (default: %(default)s, relative to the working directory)")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="ML-Agents worker id; the player connects on port 5005 + id, which must be free")
    parser.add_argument("--log-file", default=None,
                        help="Player log path (default: Unity's own location)")
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--max-episodes", type=int, default=3,
                        help="Episodes to play before failing; the first one keeps the player's previous config")
    parser.add_argument("--timeout", type=float, default=600.0, help="Wall-clock limit in seconds")
    args = parser.parse_args(argv)

    if args.log_file is not None:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)   # the player exits if it can't open it

    # Without a handler, SIGTERM (kill <pid>, timeout) would skip the finally below and leave the player running.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))

    env = UnitySchedulingEnv(file_name=args.unity_path, worker_id=args.worker_id, no_graphics=True,
                             time_scale=args.time_scale, timeout_wait=TIMEOUT_WAIT, log_file=args.log_file)
    try:
        return 0 if run(env, args.max_episodes, args.timeout) else 1
    except UnityException as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}\n(A player that rejects a config exits with code 3; "
              f"the reason is in its log.)")
        return 1
    finally:
        close_own_player(env)


if __name__ == "__main__":
    sys.exit(main())
