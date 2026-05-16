# test_channels.py
import json
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channels.channels import EpisodeConfigChannel, EpisodeTelemetryChannel

engine_channel = EngineConfigurationChannel()
config_channel = EpisodeConfigChannel()
telemetry_channel = EpisodeTelemetryChannel()

env = UnityEnvironment(
    file_name="/code/linux_server/capstone.x86_64",
    side_channels=[engine_channel, config_channel, telemetry_channel],
    worker_id=0,
    no_graphics=True,
)
engine_channel.set_configuration_parameters(time_scale=20.0)

max_wait = 30
for i in range(max_wait):
    specs = env.behavior_specs
    if len(specs) > 0:
        break
    print(f"Waiting for behavior registration... ({i+1}/{max_wait})")
    env.step()
    time.sleep(0.2)
else:
    print("ERROR: No behavior registered after waiting. Check Unity scene.")
    env.close()
    exit(1)

behavior_name = list(env.behavior_specs.keys())[0]
print(f"Connected. Behavior: {behavior_name}")

# ── Test 1: send a config and see if Unity logs it ───────────────────
test_config = {
    "name": "channel_test",
    "seed": 99,
    "jobCount": 5,
    "machinesPerType": 1,
    "machineTypes": ["Mill", "Lathe", "Weld", "Inspect", "Assemble"],
    "minProcTime": 5.0,
    "maxProcTime": 15.0,
    "minOpsPerJob": 2,
    "maxOpsPerJob": 3,
    "maxArrivalTime": 0.0,
    "agvCount": 2,
}
config_channel.send_config(test_config)
print("Config sent. Check Unity log for: [ConfigChannel] Received config: channel_test")

# ── Reset and run one episode with action=0 ──────────────────────────
env.reset()
print("Reset complete.")

steps = 0
done = False
while not done:
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    if len(terminal_steps) > 0:
        print(f"Episode ended after {steps} decisions.")
        done = True
        break

    if len(decision_steps) > 0:
        action = ActionTuple(discrete=np.array([[0]], dtype=np.int32))
        env.set_actions(behavior_name, action)
        steps += 1

    env.step()

# ── Test 2: check telemetry came back ───────────────────────────────
payload = telemetry_channel.pop_payload()
if payload is None:
    print("FAIL: no telemetry payload received.")
else:
    print(f"PASS: telemetry received.")
    print(f"  events:  {len(payload.get('events', []))}")
    result = payload.get("result")
    if result:
        print(f"  makespan:  {result.get('makespan')}")
        print(f"  jobs:      {result.get('jobCount')}")
        print(f"  rule:      {result.get('ruleName')}")
        print(f"  stochastic: {result.get('stochasticTag')}")
    else:
        print("  WARN: result block missing — is Flush() called in HandleEpisodeFinished?")

env.close()