"""
@file channels.py
@brief Python-side ML-Agents side channel implementations.

Matches the C# GUIDs exactly:
  EpisodeConfigChannel:    b1e2c3d4-f5a6-7890-bcde-f01234567891
  EpisodeTelemetryChannel: c2f3d4e5-a6b7-8901-cdef-012345678902

Usage in UnitySchedulingEnv:
  self.config_channel = EpisodeConfigChannel()
  self.telemetry_channel = EpisodeTelemetryChannel()
  self.env = UnityEnvironment(
      ...,
      side_channels=[self.engine_channel, self.config_channel, self.telemetry_channel]
  )

  # Before each reset, push a config dict:
  self.config_channel.send_config(my_fjssp_config_dict)
  obs = self.reset()

  # After episode ends, read telemetry:
  payload = self.telemetry_channel.pop_payload()
  ttfs = [e["ttf"] for e in payload["events"] if e["type"] == "machine_failure"]
"""

import uuid
import json
from typing import Optional
from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage, OutgoingMessage


class EpisodeConfigChannel(SideChannel):
    """
    Sends a full FJSSPConfig JSON blob to Unity before each episode reset.
    Unity's EpisodeConfigChannel.OnMessageReceived() deserialises it into
    a FJSSPConfig and makes it available via ConsumeConfig().
    """

    CHANNEL_ID = uuid.UUID("b1e2c3d4-f5a6-7890-bcde-f01234567891")

    def __init__(self):
        super().__init__(self.CHANNEL_ID)

    def on_message_received(self, msg: IncomingMessage):
        # Config channel is send-only from Python — Unity never replies on this channel.
        pass

    def send_config(self, config: dict):
        """
        Serialise a config dict and queue it for Unity.
        Call this BEFORE env.reset() so Unity receives it in OnEpisodeBegin.

        Args:
            config: Dict matching FJSSPConfig fields. Minimal required keys:
                    jobCount, machinesPerType, machineTypes, agvCount.
                    Optional: stochastic block, procTimeParams, seed, name, etc.

        Example:
            channel.send_config({
                "name": "30j_15m_mf_low",
                "seed": 42,
                "jobCount": 30,
                "machinesPerType": 3,
                "machineTypes": ["Mill", "Lathe", "Weld", "Inspect", "Assemble"],
                "minProcTime": 15.0,
                "maxProcTime": 60.0,
                "minOpsPerJob": 4,
                "maxOpsPerJob": 6,
                "maxArrivalTime": 0.0,
                "agvCount": 10,
                "stochastic": {
                    "machineFailuresEnabled": True,
                    "weibullK": 1.5,
                    "weibullLambda": 900.0,
                    "repairLogMu": 4.0,
                    "repairLogSigma": 0.5,
                }
            })
        """
        msg = OutgoingMessage()
        msg.write_string(json.dumps(config))
        super().queue_message_to_send(msg)


class EpisodeTelemetryChannel(SideChannel):
    """
    Receives per-episode telemetry JSON from Unity at episode end.
    Unity's EpisodeTelemetryChannel.Flush() sends a payload containing:
      - events: list of failure/repair/arrival events with timing data
      - result: final episode metrics (makespan, reward, rule, stochastic tag)

    Access via pop_payload() after each episode terminates.
    """

    CHANNEL_ID = uuid.UUID("c2f3d4e5-a6b7-8901-cdef-012345678902")

    def __init__(self):
        super().__init__(self.CHANNEL_ID)
        self._payloads = []  # one entry per episode

    def on_message_received(self, msg: IncomingMessage):
        raw = msg.read_string()
        try:
            payload = json.loads(raw)
            self._payloads.append(payload)
        except json.JSONDecodeError as e:
            print(f"[TelemetryChannel] JSON decode error: {e}")

    def pop_payload(self) -> Optional[dict]:
        """
        Returns and removes the most recent episode payload, or None if empty.
        Call after done=True is returned by env.step().

        Returns dict with structure:
        {
            "events": [
                {"type": "machine_failure", "machineId": 2, "ttf": 847.3,
                 "repairDuration": 58.2, "simTime": 847.3},
                {"type": "machine_repair_complete", "machineId": 2,
                 "repairDuration": 58.2, "simTime": 905.5},
                ...
            ],
            "result": {
                "makespan": 1923.4, "jobCount": 30, "machineCount": 15,
                "totalOps": 142, "decisions": 89, "totalReward": -0.342,
                "ruleName": "SPT_SMPT", "stochasticTag": "mf"
            }
        }
        """
        if not self._payloads:
            return None
        return self._payloads.pop(0)

    def pop_all_payloads(self) -> list:
        """Returns and clears all buffered payloads."""
        payloads = list(self._payloads)
        self._payloads.clear()
        return payloads

    def extract_machine_ttfs(self, payload: dict) -> list:
        """Convenience: extract all observed machine TTF values from a payload."""
        if payload is None:
            return []
        return [e["ttf"] for e in payload.get("events", [])
                if e["type"] == "machine_failure"]

    def extract_repair_durations(self, payload: dict) -> list:
        """Convenience: extract all machine repair durations from a payload."""
        if payload is None:
            return []
        return [e["repairDuration"] for e in payload.get("events", [])
                if e["type"] == "machine_failure"]

    def extract_inter_arrivals(self, payload: dict) -> list:
        """Convenience: extract Poisson inter-arrival times from a payload."""
        if payload is None:
            return []
        return [e["interArrivalTime"] for e in payload.get("events", [])
                if e["type"] == "job_arrival"]