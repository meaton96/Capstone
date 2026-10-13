"""Event-based twin of the Unity shop floor (thesis: event-based vs physically simulated environments).

The twin replays a Unity run's jobs and dispatching logic as a next-event simulation and swaps only the
transport model, so its gap to the Unity run isolates what the physical floor adds:

  instant    DES-0: transfers take no time and there are no vehicles (the usual DRL-DFJSP model).
  geometric  DES-1g: N AGVs drive the zone-graph route at constant speed (path length / speed + handshakes).
  kinematic  DES-1k: the same AGVs with Unity's free-flow motion (turning, dock alignment), still with no
             zone reservations, so vehicles never wait for or block each other.

Inputs come from a Unity run with "-destrace" (Logging/DesTwinExport.cs): des_floor.json, des_jobs.json.
"""
from .engine import TwinConfig, run_twin
from .floor import Floor

__all__ = ["Floor", "TwinConfig", "run_twin"]
