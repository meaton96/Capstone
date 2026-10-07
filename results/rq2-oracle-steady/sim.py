"""
@file sim.py
@brief Shared twin helpers for rq2-oracle-steady and rq2-realizable (2026-10-07): settings, scenario generation, and one
       episode with a slot policy (a fixed schedule or a callback on the state at each slot start).

Settings (all: twin, tardiness, failures off, c ~ U[1.75, 2.5], B2 load mix normal 0.6-1.2 (2) : surge 1.0-1.6 (1) :
short fleet 2 AGVs (1)):
  - B2: rq2-twin-fleet's B2 exactly (random warm-up 0-2.5 h, 6 h window, regime blocks of 1.5 h, 900 s slots). The
        setting of every RL result so far; fixed pairs here equal rq2-twin-fleet's.
  - S6: steady state with long regime pieces (base-eplen: the floor needs about a day to level off; 6 h blocks give the
        largest and most persistent chunk-to-chunk differences): warm-up >= 24 h (a segment start in [24 h, ~26.5 h]),
        12 h window, regime blocks of 6 h, 1,800 s slots (rq2-oracle-slotlen: 30-min slots keep 91% of the headroom).
"""
import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "env"))
_spec = importlib.util.spec_from_file_location("sig", REPO / "results/dev-congestion-signal/run.py")
sig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig)
fleet = sig.fleet
PAIRS = list(fleet.PAIRS)

SETTINGS = {
    "B2": {"window": 21600.0, "block": 5400.0, "min_warmup": 0.0, "slot": 900.0, "horizon": 30600.0},
    "S6": {"window": 43200.0, "block": 21600.0, "min_warmup": 86400.0, "slot": 1800.0,
           "horizon": 86400.0 + 43200.0 + 9000.0},
}


def n_slots(setting):
    s = SETTINGS[setting]
    return int(round(s["window"] / s["slot"]))


def scenario(setting, seed):
    from scenarios.randomized import DEFAULT_PARAMS, randomized_generator
    s = SETTINGS[setting]
    if setting == "B2":
        return fleet.scenario("B2", seed)               # identical to rq2-twin-fleet
    params = dataclasses.replace(DEFAULT_PARAMS, failure_probability=0.0, due_date_allowance_range=(1.75, 2.5),
                                 regime_block_seconds=s["block"], horizon_seconds=s["horizon"],
                                 load_mix=fleet.SETTINGS["B2"]["params"]["load_mix"])
    return randomized_generator(s["window"], random_warmup=True, params=params,
                                min_warmup_seconds=s["min_warmup"])(seed)


class Episode:
    """One instance, ready to play many times."""

    def __init__(self, setting, seed):
        from des_twin.scenario import agv_schedule, episode_settings, resolve_jobs
        self.setting, self.seed = setting, seed
        self.cfg = SETTINGS[setting]
        self.sc = scenario(setting, seed)
        self.sched = agv_schedule(self.sc)
        self.warmup, _, self.warm_rule = episode_settings(self.sc)
        self.jobs = resolve_jobs(self.sc, sig.floor())
        self.blocks = self.sc["_meta"]["blocks"]

    def regime_at(self, t):
        for b in self.blocks:
            if b["start"] <= t < b["end"]:
                return b["load_profile"]
        return self.blocks[-1]["load_profile"]

    def play(self, policy, capture=None):
        """policy: list of pairs per slot (the last one holds to the end), or a callable (tw, dec, k, ep) -> pair
        evaluated at each slot's first decision. capture(tw, dec, k) is called there too. Returns window tardiness
        (1000 job-s) and the pairs played per slot."""
        from des_twin import TwinConfig
        from des_twin.engine import Twin
        tw = Twin(sig.floor(), self.jobs, TwinConfig(
            rule=self.warm_rule, transport="kinematic", warmup_seconds=self.warmup,
            episode_duration_seconds=self.cfg["window"], max_sim_seconds=self.warmup + self.cfg["window"] + 200000.0,
            agv_schedule=self.sched))
        slot, n = self.cfg["slot"], n_slots(self.setting)
        gen = tw.agent_decisions()
        t0, played, current = None, [], None
        try:
            dec = gen.send(None)
            while True:
                if t0 is None:
                    t0 = tw.now
                k = int((tw.now - t0) // slot)
                if k >= len(played) and k < n:
                    while len(played) < k:            # a slot with no decision keeps the previous pair
                        played.append(current)
                    if capture is not None:
                        capture(tw, dec, k)
                    current = policy(tw, dec, k, self) if callable(policy) else policy[min(k, len(policy) - 1)]
                    played.append(current)
                pair = current if current is not None else (policy[-1] if not callable(policy) else "MDD-TECT")
                dec = gen.send(tuple(pair.split("-")))
        except StopIteration:
            pass
        t0 = self.warmup if t0 is None else t0
        t1 = tw.now
        tard = sum(fleet.due.overlap(j.due, j.exit_time if j.exit_time is not None else t1, t0, t1)
                   for j in tw.jobs.values() if j.due is not None)
        return tard / 1000.0, played


def state_features(tw):
    """WIP and jobs waiting for an AGV per on-duty AGV at a decision (dev-congestion-signal's strongest)."""
    f, _ = sig.congestion(tw, 0.0, None)
    return {"wip": f["c_wip"], "wait_per_agv": f["c_wait_per_agv"]}


def parse_seeds(text):
    out = []
    for part in str(text).split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out
