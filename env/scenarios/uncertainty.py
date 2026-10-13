"""
@file uncertainty.py
@brief Scenario plumbing of the twin uncertainty layer (2026-10-11, dev-twin-uncertainty; plan §2.5): an "uncertainty"
       block in the scenario dict (spec + noise seed + draw, and the rush-order pool when E4 / E4n is on), and the
       trace built from it for the twin.

@details The block is twin-only (Unity ignores unknown keys; execution-noise injection into Unity is plan §4, not
built). The noise seed defaults to the scenario's seed (the instance seed), so instances that share a seed share their
noise draw too (train.py --shared-baseline-group keeps its variance reduction). The rush pool is another instance of
the same generator (seed + RUSH_POOL_OFFSET, rq2-oracle-perturbed's OTHER_SEED convention), stored as scenario jobs and
resolved with the scenario.
"""
import copy
from typing import Callable, Dict, Optional

RUSH_POOL_OFFSET = 70000


def uncertainty_block(spec, seed: int, draw: int = 0, pool_jobs=None) -> Dict:
    from des_twin.uncertainty import UncertaintyConfig
    cfg = UncertaintyConfig.from_spec(spec)
    block = {"config": cfg.to_dict(), "noiseSeed": int(seed), "draw": int(draw)}
    if pool_jobs is not None:
        block["rushPool"] = pool_jobs
    return block


def with_uncertainty(generator: Callable[[int], Dict], spec, draw: int = 0,
                     pool_offset: int = RUSH_POOL_OFFSET) -> Callable[[int], Dict]:
    """@brief seed -> scenario callable that adds the uncertainty block to every scenario of @p generator.
    @p spec: anything UncertaintyConfig.from_spec takes, or a callable seed -> spec (domain randomization)."""
    def _generate(seed: int) -> Dict:
        from des_twin.uncertainty import UncertaintyConfig
        scenario = copy.deepcopy(generator(seed))
        s = spec(seed) if callable(spec) else spec
        cfg = UncertaintyConfig.from_spec(s)
        pool = None
        if cfg.active("E4") or cfg.active("E4n"):
            pool = generator(seed + pool_offset)["jobs"]
        scenario["uncertainty"] = uncertainty_block(cfg, scenario.get("seed", seed), draw, pool)
        return scenario
    return _generate


def dr_spec(weights=(0.2, 0.3, 0.3, 0.2), types=None, info: str = "I0", base_seed: int = 0):
    """@brief A per-seed domain-randomization spec for with_uncertainty: each type's level drawn with these weights
    (none / mild / realistic / severe), seeded by the scenario seed."""
    import numpy as np
    from des_twin.uncertainty import MIX_TYPES, UncertaintyConfig

    def spec(seed: int):
        rng = np.random.default_rng([base_seed, int(seed), 4242])
        return UncertaintyConfig.mix(rng, types=types or MIX_TYPES, weights=weights, info=info)
    return spec


def trace_for(scenario: Dict, floor, plan_jobs: Dict, spec=None, seed: Optional[int] = None):
    """@brief The UncertaintyTrace for a resolved scenario: from its "uncertainty" block, or from @p spec (then the
    noise seed is @p seed or the scenario seed); None when neither is given. The rush pool (if any) is resolved on the
    floor like the scenario's own jobs."""
    from des_twin.scenario import resolve_jobs
    from des_twin.uncertainty import UncertaintyConfig, UncertaintyTrace
    block = scenario.get("uncertainty")
    if spec is None and not block:
        return None
    if spec is not None:
        cfg = UncertaintyConfig.from_spec(spec)
        noise_seed = scenario.get("seed", 42) if seed is None else seed
        draw, pool_jobs = 0, (block or {}).get("rushPool")
    else:
        cfg = UncertaintyConfig.from_dict(block["config"])
        noise_seed, draw, pool_jobs = block.get("noiseSeed", scenario.get("seed", 42)), block.get("draw", 0), \
            block.get("rushPool")
    pool = None
    if cfg.active("E4") or cfg.active("E4n"):
        if pool_jobs is None:
            raise ValueError("E4 / E4n need the scenario's uncertainty.rushPool (scenarios.uncertainty.with_uncertainty)")
        pool = resolve_jobs(dict(scenario, jobs=pool_jobs), floor)
    return UncertaintyTrace.build(plan_jobs, floor, cfg, noise_seed, rush_pool=pool, draw=draw)


UNCERTAINTY_HELP = ("twin only, with --scenario-generator: run every instance's world derived from its plan with the "
                    "twin uncertainty layer (des_twin.uncertainty). A level ('mild' / 'realistic' / 'severe' = every "
                    "MIX type at that level), types ('E2d=realistic,O1=realistic[,info=I1]'), or 'dr[:I0|I1]' "
                    "(domain randomization: per instance and type none/mild/realistic/severe with weights "
                    "0.2/0.3/0.3/0.2). The noise seed is the instance seed.")


def spec_from_arg(text):
    """@brief The spec (or a per-seed spec callable) for a --uncertainty CLI value; None for an empty value."""
    if text is None or str(text).strip() == "":
        return None
    t = str(text).strip()
    if t.lower() == "dr" or t.lower().startswith("dr:"):
        info = t.split(":", 1)[1] if ":" in t else "I0"
        return dr_spec(info=info)
    from des_twin.uncertainty import UncertaintyConfig
    UncertaintyConfig.from_spec(t)            # validate now
    return t
