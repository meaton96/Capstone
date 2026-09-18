"""
@file compound.py
@brief Seeded variants of the compound_scenario chain scenario, for RL training.

@details
Wraps linux_server/BatchConfigs/Scenarios/generate_scenarios.compound_scenario — already a
pure function of its seed (builds its own random.Random(seed), no shared state, no file I/O)
returning a ScenarioLoader-schema dict — so a fresh ~360-job, 13-phase variant can be built
on the fly for every training episode instead of training on one fixed instance.

See linux_server/BatchConfigs/Scenarios/compound_scenario.json for what one instance looks
like (job counts, phase structure) and env/rewards for why makespan is the wrong reward here
(PDR mean flow time spans 563-974 across rules on the seed=42 instance while makespan barely
moves — the compound scenario differentiates rules on flow time, not completion time).

@par Usage
@code{.sh}
python env/train.py --unity --unity-path linux_server/capstone.x86_64 \\
    --scenario-generator compound --episode-duration-seconds 3000 \\
    --reward-spec env/config/rewards/flow_time.json --train-seed 0 ...
@endcode
"""

import importlib.util
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

## @brief The generator lives in linux_server/, not env/ — it is also the source used to
##        produce the checked-in scenario JSON files (generate_scenarios.py main()).
_GENERATOR_PATH = (Path(__file__).resolve().parent.parent.parent
                   / "linux_server" / "BatchConfigs" / "Scenarios" / "generate_scenarios.py")

_module = None


def _load_module():
    """@brief Import generate_scenarios.py by path (it isn't part of any Python package)."""
    global _module
    if _module is None:
        if not _GENERATOR_PATH.is_file():
            raise FileNotFoundError(f"Scenario generator not found: {_GENERATOR_PATH}")
        spec = importlib.util.spec_from_file_location("_generate_scenarios", _GENERATOR_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _module = module
    return _module


## @brief --scenario-generator NAME -> generate_scenarios.py function name. "compound_v2" adds
##        the speed_trap regime (see generate_scenarios.compound_scenario_v2's docstring) on top
##        of everything compound_scenario has -- additive, not a replacement, so "compound"
##        stays exactly what every prior PDR baseline and training run measured against.
_SCENARIO_FN_NAMES = {"compound": "compound_scenario", "compound_v2": "compound_scenario_v2"}


def compound_variant(seed: int, episode_duration_seconds: Optional[float] = None,
                      warmup_seconds: Optional[float] = None, variant: str = "compound",
                      agv_move_speed: Optional[float] = None,
                      agv_handshake_duration: Optional[float] = None) -> Dict:
    """@brief One seeded compound_scenario (or compound_scenario_v2) instance.

    @param episode_duration_seconds  If set, adds a steady-state time cap (the scenario's
                                     "stochastic": {"episodeDurationSeconds": ...}), cutting the
                                     episode short — truncated, not terminated — with in-flight
                                     jobs recorded as censored, instead of running to its natural
                                     length every episode.
    @param warmup_seconds  If set, adds a warm-start window ("stochastic": {"warmupSeconds": ...}):
                           the floor runs under the scenario's dispatchingRule (see
                           REFERENCE_RULE / phase_start_offsets below) for this many sim-seconds
                           before the RL agent takes over, with no Python round-trips during that
                           window — see FactoryOrchestrator.InWarmup. Lets an episode "begin" with
                           realistic mid-scenario WIP instead of an empty floor.
    @param variant  "compound" (default, ~11,255s/383 jobs) or "compound_v2" (~15,855s/533 jobs,
                    adds the speed_trap regime).
    @param agv_move_speed  If set, overrides the AGV prefab's travel speed (units/sim-second,
                           prefab default 3.5) for this scenario — see FJSSPConfig.AGVMoveSpeed.
                           Faster AGVs shrink physical transit time relative to job processing
                           time without changing the scripted job-generation rate, so machines
                           see jobs actually queue up together more often instead of arriving
                           one-at-a-time spaced out by transit — see
                           dispatch-degeneracy-regime-dependence: on compound_scenario, dispatch
                           decisions were degenerate (<=1 real candidate) 42-77% of the time even
                           in phases sized for sustained contention, only reliably avoided by the
                           instantaneous `burst` phase.
    @param agv_handshake_duration  If set, overrides the AGV prefab's pickup/dropoff handshake
                           time (sim-seconds, prefab default 1.5) — see
                           FJSSPConfig.AGVHandshakeDuration.
    """
    fn = getattr(_load_module(), _SCENARIO_FN_NAMES[variant])
    scenario = fn(seed=seed)
    scenario = dict(scenario)
    stochastic = {}
    if episode_duration_seconds is not None:
        stochastic["episodeDurationSeconds"] = float(episode_duration_seconds)
    if warmup_seconds is not None:
        stochastic["warmupSeconds"] = float(warmup_seconds)
    if stochastic:
        scenario["stochastic"] = stochastic
    if agv_move_speed is not None:
        scenario["agvMoveSpeed"] = float(agv_move_speed)
    if agv_handshake_duration is not None:
        scenario["agvHandshakeDuration"] = float(agv_handshake_duration)
    return scenario


def phase_start_offsets(seed: int, variant: str = "compound") -> List[float]:
    """@brief Phase-boundary start times (sim-seconds) for a compound_scenario variant.

    @details Usable as warm-up cutoffs so an episode "begins" exactly at a regime change
    (e.g. the start of standoff_1, or starvation_1) instead of at an arbitrary point mid-phase.
    Excludes the scenario's final phase (quiet_7_cooldown) — warming up to there would leave
    the agent nothing but a cooldown to act on.
    """
    fn = getattr(_load_module(), _SCENARIO_FN_NAMES[variant])
    scenario = fn(seed=seed)
    return [p["start"] for p in scenario["_phases"][:-1]]


def compound_generator(episode_duration_seconds: Optional[float] = None,
                        random_warmup: bool = False,
                        warmup_dispatching_rule: Optional[str] = None,
                        variant: str = "compound",
                        agv_move_speed: Optional[float] = None,
                        agv_handshake_duration: Optional[float] = None) -> Callable[[int], Dict]:
    """@brief A scenario_generator callable for UnitySchedulingEnv/VectorizedUnityEnv.

    @param random_warmup  If set, each episode's warm-up cutoff is drawn — deterministically,
                          from that episode's own seed, so a repeated seed reproduces the same
                          cutoff — from phase_start_offsets(seed) instead of every episode
                          starting at t=0. Fixes the SHORT-episode bias found in
                          episode-length-comparison-bimodal-generalization: a fixed-length
                          truncated episode that always starts at t=0 only ever sees phases 1-3
                          of the 13-phase cycle; a random phase-aligned start sees all of them
                          across training at SHORT's cheaper episode cost.
    @param warmup_dispatching_rule  DispatchingRule name (e.g. "SPT_SMPT") driving the warm-up
                          window. None uses the scenario JSON's own default (SRT_SRWT, see
                          ScenarioLoader.ReadDispatchingRule).
    @param variant  "compound" or "compound_v2" — see compound_variant. Bound at registry-build
                    time (see REGISTRY below), not passed per-call by train.py/evaluate.py.
    @param agv_move_speed, agv_handshake_duration  See compound_variant.
    """
    def _generate(seed: int) -> Dict:
        warmup = (random.Random(seed).choice(phase_start_offsets(seed, variant))
                  if random_warmup else None)
        scenario = compound_variant(seed, episode_duration_seconds, warmup, variant,
                                     agv_move_speed, agv_handshake_duration)
        if warmup_dispatching_rule is not None:
            scenario["dispatchingRule"] = warmup_dispatching_rule
        return scenario
    return _generate


def _generator_for(variant: str):
    def _factory(episode_duration_seconds=None, random_warmup=False, warmup_dispatching_rule=None,
                 agv_move_speed=None, agv_handshake_duration=None):
        return compound_generator(episode_duration_seconds, random_warmup,
                                   warmup_dispatching_rule, variant=variant,
                                   agv_move_speed=agv_move_speed,
                                   agv_handshake_duration=agv_handshake_duration)
    return _factory


## @brief name -> generator-factory, for a --scenario-generator NAME style CLI flag.
REGISTRY = {name: _generator_for(name) for name in _SCENARIO_FN_NAMES}
