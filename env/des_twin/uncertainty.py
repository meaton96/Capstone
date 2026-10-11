"""
@file uncertainty.py
@brief Twin uncertainty layer (2026-10-11, dev-twin-uncertainty; plan docs/Plans/TWIN_UNCERTAINTY_LAYER_PLAN_2026-10-10.md
       §1-2.1): execution noise (the world differs from the plan) and observation noise (the controllers' view differs
       from the world), as exogenous, seeded traces drawn independently of the policy (common random numbers).

@details
**The plan is primary.** The scenario generator's output (a des_jobs/1 list) is the *plan*: planned arrivals,
planned per-machine durations, due dates. `UncertaintyTrace.build` derives the *world* from it:
- `world_jobs`: the jobs the engine runs (Twin(floor, trace.world_jobs, TwinConfig(uncertainty=trace))): planned jobs
  with their actual arrival (E3), minus cancelled orders, plus rush orders (E4 / E4n). Every job keeps its PLANNED
  durations in Job.ops (rules, observation and planner read planned values) and its plan due date;
- the actual duration of an operation, drawn when it starts: f32(p_plan x b_m(t) x f_jom) (`actual_duration`), with
  f = E1's per-op lognormal factor and b = the machine bias (E2 static x E2d drift x wear);
- observation noise for the view (des_twin.observed): booking delays per (job, op, transition, occurrence) (O1),
  periodic snapshots (O1p), per-machine sensor dropouts (O2), per-op remaining-time estimate factors (O3);
- `plan_view(t, ...)`: the published schedule for the planner / look-ahead (planned arrivals, cancellations known by
  their notice time, P1 refreshes).

**Seeding and CRN.** Every draw is keyed by (noise_seed, draw, purpose, entity...) and never depends on the order in
which the engine asks, so the same (job, op, machine) gets the same factor under every rule / policy / forced choice.
Per-entity scalar draws use a counter-based generator (splitmix64 over the key, then the inverse normal / exponential
CDF); long per-machine paths (OU drift, dropout renewal process) use numpy PCG64 streams seeded per machine and purpose.
Each severity only SCALES standardized variates (sigma x Z) or THRESHOLDS uniforms (event if U < q), so an op's error
has the same sign at every level and the event sets are nested (the late jobs at "mild" are a subset of those at
"realistic"); degradation curves are paired across levels.

**Mean-one convention** (deviation, documented in results/dev-twin-uncertainty/README.md): every multiplicative factor
(E1 job/op factors, E2 static bias, E2d drift, the wear sawtooth, O3) is normalized to mean one, so the expected load
equals the plan's under every type and level; a biased plan is the separate knob `plan_optimism`.

**Exactness.** An all-zero configuration gives factors of exactly 1.0, bias 1.0, delays 0.0 and world_jobs identical
to the plan, so f32(p x 1 x 1) == p and the engine reproduces the noise-free twin bit for bit
(results/dev-twin-uncertainty/check_exact.py).

**Conditional redraw** (`redraw_after`, §1.5): keep what has happened by time t and redraw the rest with seed k, for
rollouts scored over futures that use only current-state knowledge (mode "realizable": the bias continues from the
online estimate) or the true current bias (mode "oracle"). A job not yet arrived whose redrawn arrival would be in
the past (at or before the current tick) arrives at the next tick (next_tick_arrival; fix of 10-11,
results/dev-uncert-clipfix).
"""
import copy
import math
from bisect import bisect_right
from dataclasses import dataclass, field, fields
from statistics import NormalDist

import numpy as np

f32 = np.float32

LEVELS = ("none", "mild", "realistic", "severe")
EXECUTION_TYPES = ("E1", "E2", "E2d", "E3", "E4", "E4n")
OBSERVATION_TYPES = ("O1", "O1p", "O2", "O3")
TYPES = EXECUTION_TYPES + OBSERVATION_TYPES
# The screen's MIX cell: every type at the same level (E4n, not E4; O1, not its snapshot variant).
MIX_TYPES = ("E1", "E2", "E2d", "E3", "E4n", "O1", "O2", "O3")
H = 3600.0

# Presets (plan §1.1 / §1.2; every "realistic" value is uncertain, the levels bracket plausible plants).
PRESETS = {
    "E1": {"none": {"sigma_job": 0.0, "sigma_op": 0.0}, "mild": {"sigma_job": 0.03, "sigma_op": 0.03},
           "realistic": {"sigma_job": 0.08, "sigma_op": 0.10}, "severe": {"sigma_job": 0.15, "sigma_op": 0.20}},
    "E2": {"none": {"sigma_s": 0.0}, "mild": {"sigma_s": 0.05}, "realistic": {"sigma_s": 0.10},
           "severe": {"sigma_s": 0.20}},
    "E2d": {"none": {"sigma_d": 0.0, "tau": 4 * H, "wear": 0.0, "t_tool": 3 * H},
            "mild": {"sigma_d": 0.03, "tau": 4 * H, "wear": 0.0, "t_tool": 3 * H},
            "realistic": {"sigma_d": 0.08, "tau": 4 * H, "wear": 0.0, "t_tool": 3 * H},
            "severe": {"sigma_d": 0.15, "tau": 2 * H, "wear": 0.15, "t_tool": 3 * H}},
    "E3": {"none": {"sigma_a": 0.0, "q": 0.0, "m": 0.0}, "mild": {"sigma_a": 120.0, "q": 0.0, "m": 0.0},
           "realistic": {"sigma_a": 600.0, "q": 0.10, "m": 1800.0},
           "severe": {"sigma_a": 1200.0, "q": 0.25, "m": 3600.0}},
    "E4": {"none": {"r": 0.0, "x": 0.0}, "mild": {"r": 0.03, "x": 0.01}, "realistic": {"r": 0.10, "x": 0.03},
           "severe": {"r": 0.20, "x": 0.08}},
    "E4n": {"none": {"r": 0.0, "x": 0.0}, "mild": {"r": 0.03, "x": 0.03}, "realistic": {"r": 0.10, "x": 0.10},
            "severe": {"r": 0.20, "x": 0.20}},
    "O1": {"none": {"mean": 0.0, "p_extra": 0.0, "mean_extra": 0.0}, "mild": {"mean": 30.0, "p_extra": 0.0, "mean_extra": 0.0},
           "realistic": {"mean": 120.0, "p_extra": 0.10, "mean_extra": 600.0},
           "severe": {"mean": 300.0, "p_extra": 0.20, "mean_extra": 1800.0}},
    "O1p": {"none": {"period": 0.0}, "mild": {"period": 60.0}, "realistic": {"period": 300.0},
            "severe": {"period": 900.0}},
    "O2": {"none": {"rate": 0.0, "mean": 0.0}, "mild": {"rate": 1.0 / (8 * H), "mean": 300.0},
           "realistic": {"rate": 1.0 / (4 * H), "mean": 900.0}, "severe": {"rate": 1.0 / (2 * H), "mean": 1800.0}},
    "O3": {"none": {"sigma_g": 0.0}, "mild": {"sigma_g": 0.05}, "realistic": {"sigma_g": 0.15},
           "severe": {"sigma_g": 0.30}},
}

# Purposes of the keyed draws.
_E1J, _E1O, _E2, _E2D, _WEAR, _E3Z, _E3U, _E3E, _E4R, _E4C, _E4N, _E4RC = 11, 12, 21, 22, 23, 31, 32, 33, 41, 42, 43, 44
_O1, _O1P, _O2, _O3 = 51, 52, 61, 71
_REDRAW = 977
RUSH_ID_OFFSET = 1_000_000
BIAS_DT = 60.0                     # E2d: piecewise constant on a 60 s grid
CANCEL_NOTICE_MAX = 2 * H          # E4: the planner learns of a cancellation U[0, 2 h] before the planned arrival

# Transition codes for the booking-delay keys (job states of des_twin.engine, plus arrival).
ARRIVED = 6

# ── Keyed counter-based draws ───────────────────────────────────────────────────────────────────────────────

_M64 = (1 << 64) - 1
_ND = NormalDist()


def _sm(x):
    """splitmix64 finalizer step."""
    x = (x + 0x9E3779B97F4A7C15) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


def _key(keys):
    h = 0x6A09E667F3BCC909
    for k in keys:
        h = _sm(h ^ (int(k) & _M64))
    return h


def keyed_uniform(*keys):
    """U(0, 1) (open) for the key tuple; the same key always gives the same value."""
    return ((_key(keys) >> 11) + 0.5) * (1.0 / 9007199254740992.0)


def keyed_normal(*keys):
    return _ND.inv_cdf(keyed_uniform(*keys))


def keyed_exp(*keys):
    """Exp(1)."""
    return -math.log(keyed_uniform(*keys))


def _stream(*keys):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(k) & 0xFFFFFFFF for k in keys])))


# ── Configuration ───────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class UncertaintyConfig:
    """@brief Severity per uncertainty type plus the information level and the semantics switches.

    @param levels   {type: level name ("none" / "mild" / "realistic" / "severe") or an explicit parameter dict
                    (missing keys from the "none" preset)}; types not listed are "none". E4 and E4n are exclusive.
    @param info     information level of every controller: "I0" (plan only) or "I1" (plan + online per-machine bias
                    estimate from reported completions: p_hat = p_plan x b_hat everywhere; plan §2.4); "I2" (diagnostic
                    upper bound, not a realizable level: b_hat = the true current machine bias b_m(now))
    @param info_obs with I1: whether the policy's observation also sees the estimates (False: rules and planner only)
    @param predispatch_on  "planned": the almost-done pre-dispatch fires on the controller's estimate of the remaining
                    time (planned countdown x O3 factor (x b_hat at I1)); "actual": on the true countdown (a sensor)
    @param plan_optimism   E[p_actual] / p_plan (1 = unbiased plan)
    @param o1_stamp "event": a booking carries the event's true time (back-dated scan); "booking": the time it is
                    booked (event time + delay), so FIFO's time in queue and the dead-reckoned remaining time are off too
    @param bias_half_life, bias_prior  I1 estimator: exponential forgetting half-life (s) and prior strength
                    (pseudo-observations at bias 1)
    @param cancel_notice  "advance": a cancellation becomes known U[0, 2 h] before the planned arrival; "noshow": at
                    the planned arrival
    @param p1_lead  P1 forecast refresh: the planner learns a planned job's true arrival this many seconds before it
    @param alternate_on  FailureCoordinator.FindBestAlternateMachine reads "view" (default) or "truth"
    """
    levels: dict = field(default_factory=dict)
    info: str = "I0"
    info_obs: bool = True
    predispatch_on: str = "planned"
    plan_optimism: float = 1.0
    o1_stamp: str = "event"
    bias_half_life: float = 7200.0
    bias_prior: float = 2.0
    cancel_notice: str = "advance"
    p1_lead: float = 0.0
    alternate_on: str = "view"

    def __post_init__(self):
        for t in self.levels:
            if t not in TYPES:
                raise ValueError(f"unknown uncertainty type {t!r} (expected one of {TYPES})")
        if self.param("E4", "r") > 0 and self.param("E4n", "r") > 0:
            raise ValueError("E4 and E4n are alternatives (E4n = E4 with x = r): give one of them")
        if self.info not in ("I0", "I1", "I2"):
            raise ValueError(f"info must be I0, I1 or I2, not {self.info!r}")
        if self.predispatch_on not in ("planned", "actual"):
            raise ValueError("predispatch_on must be 'planned' or 'actual'")
        if self.o1_stamp not in ("event", "booking"):
            raise ValueError("o1_stamp must be 'event' or 'booking'")
        if self.cancel_notice not in ("advance", "noshow"):
            raise ValueError("cancel_notice must be 'advance' or 'noshow'")
        if self.alternate_on not in ("view", "truth"):
            raise ValueError("alternate_on must be 'view' or 'truth'")

    def params(self, t):
        """Resolved parameters of type t."""
        spec = self.levels.get(t, "none")
        if isinstance(spec, str):
            if spec not in LEVELS:
                raise ValueError(f"{t}: level must be one of {LEVELS}, not {spec!r}")
            return dict(PRESETS[t][spec])
        out = dict(PRESETS[t]["none"])
        unknown = set(spec) - set(out)
        if unknown:
            raise ValueError(f"{t}: unknown parameters {sorted(unknown)} (expected {sorted(out)})")
        out.update({k: float(v) for k, v in spec.items()})
        return out

    def param(self, t, name):
        return self.params(t)[name]

    def active(self, t):
        """Whether type t changes anything at its configured parameters."""
        p = self.params(t)
        if t == "E1":
            return p["sigma_job"] > 0 or p["sigma_op"] > 0
        if t == "E2":
            return p["sigma_s"] > 0
        if t == "E2d":
            return p["sigma_d"] > 0 or p["wear"] > 0
        if t == "E3":
            return p["sigma_a"] > 0 or (p["q"] > 0 and p["m"] > 0)
        if t in ("E4", "E4n"):
            return p["r"] > 0 or p["x"] > 0
        if t == "O1":
            return p["mean"] > 0 or (p["p_extra"] > 0 and p["mean_extra"] > 0)
        if t == "O1p":
            return p["period"] > 0
        if t == "O2":
            return p["rate"] > 0 and p["mean"] > 0
        return p["sigma_g"] > 0

    def is_zero(self):
        """True when no type is active and the plan is unbiased (the world equals the plan; the view equals the
        truth up to the I1 estimator, which then stays at 1)."""
        return not any(self.active(t) for t in TYPES) and self.plan_optimism == 1.0

    @classmethod
    def level(cls, level, types=MIX_TYPES, **kw):
        """Every type in `types` at one level (default the screen's MIX)."""
        return cls(levels={t: level for t in types}, **kw)

    @classmethod
    def zero(cls, **kw):
        """Every type present at severity "none" (exercises the code paths; must equal the noise-free twin)."""
        return cls(levels={t: "none" for t in TYPES if t != "E4"}, **kw)

    @classmethod
    def from_spec(cls, spec):
        """A config from a spec: an UncertaintyConfig, a dict (to_dict() form, or {type: level}), or a string:
        "none", "<level>" (MIX at that level), "mix:<level>", or "E2d=realistic,O1=severe[,info=I1,...]"."""
        if spec is None:
            return None
        if isinstance(spec, UncertaintyConfig):
            return spec
        if isinstance(spec, dict):
            if "levels" in spec or any(k in spec for k in ("info", "predispatch_on")):
                return cls.from_dict(spec)
            return cls(levels=dict(spec))
        s = str(spec).strip()
        if s in LEVELS:
            return cls.level(s) if s != "none" else cls.zero()
        if s.lower().startswith("mix:"):
            return cls.level(s.split(":", 1)[1])
        levels, kw = {}, {}
        for part in s.split(","):
            k, _, v = part.partition("=")
            k, v = k.strip(), v.strip()
            if k.lower() == "mix":
                levels.update({t: v for t in MIX_TYPES})
            elif k in TYPES:
                levels[k] = v
            elif k in cls.__dataclass_fields__ and k != "levels":
                d = cls.__dataclass_fields__[k].default
                kw[k] = (v.lower() in ("1", "true", "yes")) if isinstance(d, bool) else \
                    (float(v) if isinstance(d, float) else v)
            else:
                raise ValueError(f"cannot parse uncertainty spec part {part!r}")
        return cls(levels=levels, **kw)

    def to_dict(self):
        return {f.name: copy.deepcopy(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def mix(cls, rng, types=MIX_TYPES, weights=(0.2, 0.3, 0.3, 0.2), **kw):
        """Domain randomization (plan §5b): per type a level drawn with these weights over none / mild / realistic /
        severe. rng: a numpy Generator."""
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        return cls(levels={t: LEVELS[int(rng.choice(len(LEVELS), p=w))] for t in types}, **kw)

    def label(self):
        on = [f"{t}={self.levels[t] if isinstance(self.levels[t], str) else 'custom'}" for t in TYPES
              if t in self.levels and self.active(t)]
        return ",".join(on + ([f"info={self.info}"] if self.info != "I0" else [])) or "none"

    @property
    def adaptive(self):
        """Controllers use p_hat = p_plan x b_hat (I1 estimate, or I2 true bias)."""
        return self.info != "I0"


# ── Trace ───────────────────────────────────────────────────────────────────────────────────────────────────

def _ops_of(jd):
    """[{mid: planned duration}] of a des_jobs/1 job entry (float32 values, as the engine reads them)."""
    return [{int(m): f32(d) for m, d in op["eligible"]} for op in jd["ops"]]


def _twk(jd):
    return sum(min(float(f32(d)) for _, d in op["eligible"]) for op in jd["ops"])


def _f32r(x):
    """float32 value as des_jobs.json writes it (des_twin.scenario._f32)."""
    return float(str(np.float32(x)))


def arrival_tick(arrival, dt):
    """The engine tick at which a job with this arrival is admitted (Twin._events / Twin.replace_pending's formula:
    max(1, ceil(arrival / dt - 1e-9))). A job can be swapped into a running twin at tick k only if this is > k."""
    return max(1, math.ceil(arrival / dt - 1e-9))


def next_tick_arrival(k, dt):
    """@brief The arrival a redrawn job gets when its redrawn arrival is not after the current tick k: the time of the
    next tick, (k + 1) x dt, as a des_jobs float32 value (_f32r) whose arrival_tick is the earliest tick > k.

    float32 rounding can move (k + 1) x dt onto tick k (rounded down past the 1e-9 tolerance) or onto k + 2 (rounded
    up); the value is stepped along the float32 grid until its tick is > k and, where float32 resolution allows, == k + 1
    (beyond ~2^18 s, where the float32 spacing exceeds dt = 0.02 s, the earliest representable tick > k)."""
    v = np.float32((k + 1) * dt)
    a = _f32r(v)
    up, down = np.float32(np.inf), np.float32(-np.inf)
    while arrival_tick(a, dt) <= k:
        v = np.nextafter(v, up)
        a = _f32r(v)
    while arrival_tick(a, dt) > k + 1:
        w = np.nextafter(v, down)
        b = _f32r(w)
        if arrival_tick(b, dt) <= k:
            break
        v, a = w, b
    return a


def _op_due(arrival, due, jd):
    """Operation due dates from a job's PLANNED arrival (Twin.__init__'s formula, so an unmoved job is unchanged)."""
    work = [float(min(f32(d) for _, d in op["eligible"])) for op in jd["ops"]]
    total, cum, out = sum(work), 0.0, []
    for w in work:
        cum += w
        out.append(float(arrival) + (float(due) - float(arrival)) * (cum / total if total > 0 else 1.0))
    return out


class UncertaintyTrace:
    """@brief One episode's uncertainty: the world derived from the plan, and the observation-noise draws.

    Build with `UncertaintyTrace.build(plan_jobs, floor, cfg, noise_seed, ...)`; pass `world_jobs` to the Twin and the
    trace as TwinConfig.uncertainty."""

    def __init__(self):
        self.cfg = None

    # ── construction ────────────────────────────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, plan_jobs, floor, cfg, noise_seed, horizon=None, rush_pool=None, draw=0, dt=None):
        """@param plan_jobs  the plan: a des_jobs/1 dict (des_twin.scenario.resolve_jobs output)
        @param floor        des_twin.Floor (machine ids, dt)
        @param cfg          UncertaintyConfig (or a spec for UncertaintyConfig.from_spec)
        @param noise_seed   the noise seed (default convention: the instance seed)
        @param horizon      seconds the bias / dropout paths cover (default: last planned arrival + 2 days)
        @param rush_pool    E4 / E4n: candidate unscheduled jobs, a des_jobs/1 dict or job list from the same
                            generator with another seed (scenarios.uncertainty.rush_pool); required when E4 is on
        @param draw         index of the world drawn for this plan (several worlds per plan)"""
        cfg = UncertaintyConfig.from_spec(cfg)
        tr = cls()
        tr.cfg = cfg
        tr.seed, tr.draw = int(noise_seed), int(draw)
        tr.prefix = (tr.seed, tr.draw)
        tr.dt = float(dt or floor.dt)
        tr.machines = [m.id for m in floor.machines]
        tr.plan_jobs = plan_jobs
        last = max((float(j["arrival"]) for j in plan_jobs["jobs"]), default=0.0)
        tr.horizon = float(horizon) if horizon is not None else last + 2 * 86400.0
        tr.redrawn_at = None
        tr._frozen_g = {}
        tr._p = {t: cfg.params(t) for t in TYPES}
        tr._on = {t: cfg.active(t) for t in TYPES}
        tr._build_bias()
        tr._build_dropouts()
        tr._build_world(rush_pool)
        return tr

    # E2 / E2d ------------------------------------------------------------------------------------------------

    def _build_bias(self):
        """Per machine: static factor and a 60 s grid of dynamic factors (drift x wear), mean one each."""
        p2, pd = self._p["E2"], self._p["E2d"]
        n = int(math.ceil(self.horizon / BIAS_DT)) + 2
        self.bstat, self.bdyn, self.bdrift_x = {}, {}, {}
        s = p2["sigma_s"]
        for mid in self.machines:
            if self._on["E2"]:
                z = keyed_normal(*self.prefix, _E2, mid)
                self.bstat[mid] = math.exp(s * z - 0.5 * s * s)
            else:
                self.bstat[mid] = 1.0
        if not self._on["E2d"]:
            return
        sd, tau, w, T = pd["sigma_d"], pd["tau"], pd["wear"], pd["t_tool"]
        phi = math.exp(-BIAS_DT / tau) if tau > 0 else 0.0
        innov = sd * math.sqrt(max(0.0, 1.0 - phi * phi))
        t_grid = np.arange(n) * BIAS_DT
        for mid in self.machines:
            z = _stream(*self.prefix, _E2D, mid).standard_normal(n)
            x = np.empty(n)
            x[0] = sd * z[0]
            for i in range(1, n):
                x[i] = phi * x[i - 1] + innov * z[i]
            self.bdrift_x[mid] = x
            dyn = np.exp(x - 0.5 * sd * sd)
            if w > 0 and T > 0:
                phase = keyed_uniform(self.seed, self.draw, _WEAR, mid) * T      # wear phase: the true tool state
                frac = np.mod(t_grid + phase, T) / T
                dyn = dyn * (1.0 + w * frac) / (1.0 + 0.5 * w)
            self.bdyn[mid] = dyn

    def bias_at(self, mid, t):
        """Machine bias b_m(t) (float64; exactly 1.0 when E2 and E2d are off)."""
        b = self.bstat.get(mid, 1.0)
        dyn = self.bdyn.get(mid)
        if dyn is not None:
            i = int(t // BIAS_DT)
            b = b * float(dyn[min(max(i, 0), len(dyn) - 1)])
        return b

    def log_drift_at(self, mid, t):
        x = self.bdrift_x.get(mid)
        if x is None:
            return 0.0
        return float(x[min(max(int(t // BIAS_DT), 0), len(x) - 1)])

    # E1 ------------------------------------------------------------------------------------------------------

    def factor(self, jid, op, mid):
        """E1 per-op actual factor eps_jo x eta_jom x plan_optimism (exactly 1.0 when E1 is off and the plan is
        unbiased). Keyed by (job, op[, machine]), so CRN holds across policies."""
        out = self.cfg.plan_optimism
        if self._on["E1"]:
            p = self._p["E1"]
            sj, so = p["sigma_job"], p["sigma_op"]
            pre = self.prefix
            if sj > 0:
                out *= math.exp(sj * keyed_normal(*pre, _E1J, jid, op) - 0.5 * sj * sj)
            if so > 0:
                out *= math.exp(so * keyed_normal(*pre, _E1O, jid, op, mid) - 0.5 * so * so)
        return out

    def actual_duration(self, p_plan, jid, op, mid, t):
        """f32(p_plan x b_m(t) x f_jom): the operation's actual duration when it starts at t."""
        b = self.bias_at(mid, t)
        f = self.factor(jid, op, mid)
        if b == 1.0 and f == 1.0:
            return p_plan
        return f32(float(p_plan) * b * f)

    # O3 ------------------------------------------------------------------------------------------------------

    def est_factor(self, jid, op):
        """O3: persistent factor g of the remaining-time estimate of (job, op) (1.0 when off)."""
        g = self._frozen_g.get((jid, op))
        if g is not None:
            return g
        if not self._on["O3"]:
            return 1.0
        s = self._p["O3"]["sigma_g"]
        return math.exp(s * keyed_normal(*self.prefix, _O3, jid, op) - 0.5 * s * s)

    # O1 / O1p / O2 -------------------------------------------------------------------------------------------

    def report_delay(self, jid, op, trans, occ):
        """O1: booking delay of the occ-th (job, op, transition) event (0.0 when off)."""
        if not self._on["O1"]:
            return 0.0
        p = self._p["O1"]
        pre = self.prefix
        d = p["mean"] * keyed_exp(*pre, _O1, jid, op, trans, occ, 0) if p["mean"] > 0 else 0.0
        if p["p_extra"] > 0 and p["mean_extra"] > 0 and keyed_uniform(*pre, _O1, jid, op, trans, occ, 1) < p["p_extra"]:
            d += p["mean_extra"] * keyed_exp(*pre, _O1, jid, op, trans, occ, 2)
        return d

    def _snap(self, v):
        """O1p: the first snapshot at or after v."""
        per = self._p["O1p"]["period"]
        phase = keyed_uniform(self.seed, self.draw, _O1P) * per
        n = math.ceil((v - phase) / per - 1e-12)
        return phase + n * per

    def _build_dropouts(self):
        """O2: per machine an alternating renewal process of outages [(start, end)], standardized Exp(1) draws
        scaled by the level (CRN across levels: outage i uses draws 2i, 2i + 1 of the machine's stream)."""
        self.dropouts, self._dstarts = {}, {}
        if not self._on["O2"]:
            return
        self._redraw_dropouts({mid: ([], 0.0) for mid in self.machines}, self.prefix)

    def outage_end(self, mid, t):
        """End of the outage machine mid is in at time t, or None."""
        iv = self.dropouts.get(mid)
        if not iv:
            return None
        i = bisect_right(self._dstarts[mid], t) - 1
        if i >= 0 and iv[i][0] <= t < iv[i][1]:
            return iv[i][1]
        return None

    def visible_time(self, t, jid, op, trans, occ, mid):
        """When a booking of an event at time t becomes visible to the controllers: t + O1 delay, frozen by an O2
        outage of the machine it concerns (visible at the outage end), at the next O1p snapshot."""
        v = t + self.report_delay(jid, op, trans, occ)
        if self._on["O1p"]:
            v = self._snap(v)
        if mid is not None and mid >= 0 and self._on["O2"]:
            e = self.outage_end(mid, v)
            if e is not None:
                v = e
                if self._on["O1p"]:
                    v = self._snap(v)
        return v

    @property
    def observation_on(self):
        return self._on["O1"] or self._on["O1p"] or self._on["O2"] or self._on["O3"]

    # E3 / E4: the world's job stream ---------------------------------------------------------------------------

    def _lateness(self, jid):
        p = self._p["E3"]
        pre = self.prefix
        late = p["sigma_a"] * keyed_normal(*pre, _E3Z, jid) if p["sigma_a"] > 0 else 0.0
        if p["q"] > 0 and p["m"] > 0 and keyed_uniform(*pre, _E3U, jid) < p["q"]:
            late += p["m"] * keyed_exp(*pre, _E3E, jid)
        return late

    def _e4(self):
        return self._p["E4n"] if self._on["E4n"] else self._p["E4"]

    def _cancel(self, jid, planned_arrival):
        """(cancelled?, notice time) of a planned job."""
        if not (self._on["E4"] or self._on["E4n"]) or planned_arrival <= 0.0:
            return False, None
        x = self._e4()["x"]
        if x <= 0 or keyed_uniform(*self.prefix, _E4C, jid) >= x:
            return False, None
        if self.cfg.cancel_notice == "noshow":
            return True, planned_arrival
        return True, max(0.0, planned_arrival - CANCEL_NOTICE_MAX * keyed_uniform(*self.prefix, _E4N, jid))

    def _build_world(self, rush_pool, keep=None):
        """world_jobs (des_jobs/1) from the plan. keep: redraw support (ids already arrived in the world at the
        split time, the split time, optionally "k" = the twin's current tick): those keep their parent values; the
        rest is redrawn.

        Redrawn arrivals in the past (the 10-11 clip fix, results/dev-uncert-clipfix): a job not yet arrived whose
        redrawn arrival (E3 earliness) is not after the current tick k (the tick at the split time t = twin.now) arrives
        at the NEXT tick, (k + 1) x dt (next_tick_arrival), the earliest time a running twin can still admit it.
        Before the fix it was clipped to exactly t, i.e. tick k, which Twin.replace_pending rejects. Without `keep`
        (the world at episode start) nothing changes."""
        plan = self.plan_jobs["jobs"]
        self.actual_arrival, self.cancelled, self.rush = {}, {}, {}
        if keep is not None:
            k_now = keep.get("k")
            if k_now is None:                                  # t = k x dt (twin.now) for a caller without the tick
                k_now = math.floor(keep["t"] / self.dt + 1e-6)
            a_next = next_tick_arrival(k_now, self.dt)
        moved = False
        out = []
        for jd in plan:
            jid, a = int(jd["id"]), float(jd["arrival"])
            if keep is not None and jid in keep["arrived"]:
                self.actual_arrival[jid] = keep["parent"].actual_arrival.get(jid, a)
                out.append(keep["parent"].world_by_id[jid])
                continue
            if keep is not None and jid in keep["parent"].cancelled and keep["parent"].cancelled[jid] <= keep["t"]:
                self.cancelled[jid] = keep["parent"].cancelled[jid]          # a known cancellation stays
                continue
            c, notice = self._cancel(jid, a)
            if c:
                self.cancelled[jid] = notice
                moved = True
                continue
            act = a
            if self._on["E3"] and a > 0.0:
                act = max(0.0, a + self._lateness(jid))
            if act != a:
                act = _f32r(act)
            if keep is not None and arrival_tick(act, self.dt) <= k_now:
                act = a_next                                   # in the past: arrives at the next tick
            self.actual_arrival[jid] = act
            if act != a:
                moved = True
                d = dict(jd, arrival=act)
                if jd.get("due") is not None and jd.get("op_due") is None:
                    d["op_due"] = _op_due(a, float(jd["due"]), jd)      # due dates stay on the plan
                out.append(d)
            else:
                out.append(jd)
        if self._on["E4"] or self._on["E4n"]:
            if rush_pool is None:
                raise ValueError("E4 / E4n need a rush_pool (candidate unscheduled jobs from another instance)")
            pool = rush_pool["jobs"] if isinstance(rush_pool, dict) else list(rush_pool)
            cand = [jd for jd in pool if float(jd["arrival"]) > 0.0]
            share = self._e4()["r"] * len([j for j in plan if float(j["arrival"]) > 0.0]) / max(len(cand), 1)
            for jd in cand:
                rid = RUSH_ID_OFFSET + int(jd["id"])
                if keep is not None and float(jd["arrival"]) <= keep["t"]:
                    if rid in keep["parent"].rush:                       # an arrived rush job stays as it was
                        self.rush[rid] = keep["parent"].rush[rid]
                        out.append(keep["parent"].world_by_id[rid])
                    continue
                if keyed_uniform(*self.prefix, _E4R, int(jd["id"])) >= share:
                    continue
                c = 1.0 + 0.5 * keyed_uniform(*self.prefix, _E4RC, int(jd["id"]))
                a = float(jd["arrival"])
                d = {"id": rid, "arrival": a, "ops": jd["ops"], "due": _f32r(a + c * _twk(jd))}
                self.rush[rid] = c
                out.append(d)
                moved = True
        if not moved and keep is None:
            self.world_jobs = self.plan_jobs                                  # the world IS the plan
        else:
            w = dict(self.plan_jobs)
            w["jobs"] = sorted(out, key=lambda jd: (np.float32(jd["arrival"]), int(jd["id"])))
            self.world_jobs = w
        self.world_by_id = {int(jd["id"]): jd for jd in self.world_jobs["jobs"]}
        self.plan_by_id = {int(jd["id"]): jd for jd in plan}

    # ── the plan, as the planner and the look-ahead see it ───────────────────────────────────────────────

    def plan_view(self, t, arrived):
        """@brief The published schedule for planning at time t: the planned jobs not yet arrived (ids not in
        `arrived`), at their planned arrival (P1: the true one when it is due within p1_lead), minus the
        cancellations noticed by t. Rush orders are unknown until they arrive. A planned job that is overdue (planned
        arrival < t, not arrived) is believed to arrive at t. Returns des_jobs/1 entries (plan due dates)."""
        lead = self.cfg.p1_lead
        out = []
        for jd in self.plan_jobs["jobs"]:
            jid = int(jd["id"])
            if jid in arrived:
                continue
            notice = self.cancelled.get(jid)
            if notice is not None and notice <= t:
                continue
            a = float(jd["arrival"])
            if lead > 0 and notice is None:
                act = self.actual_arrival.get(jid, a)
                if act <= t + lead:
                    a = act
            if a < t:
                a = t
            out.append(jd if a == float(jd["arrival"]) else dict(jd, arrival=a))
        return out

    def plan_future_jobs(self, t, arrived):
        """plan_view as light job objects (arrival, ops, types, due) for des_twin.lookahead."""
        out = []
        for jd in self.plan_view(t, arrived):
            out.append(_PlanJob(float(jd["arrival"]), _ops_of(jd), [op["type"] for op in jd["ops"]], jd.get("due")))
        out.sort(key=lambda j: j.arrival)
        return out

    # ── conditional redraw (§1.5) ───────────────────────────────────────────────────────────────────────────

    def redraw_after(self, twin, seed_k, mode="oracle", rush_pool=None, bias_est=None):
        """@brief A copy that keeps everything that has happened by twin.now and redraws the rest with seed_k.

        Kept: realized durations of started ops (they live in the engine), arrived jobs, known cancellations, bookings
        already emitted (they live in the view), the bias path up to now, the O3 factor of started ops, the outage in
        progress. Redrawn: E1 factors of ops not started, future arrival lateness (a redrawn arrival in the past moves
        to the next tick), future rush orders and cancellations, future booking delays and dropouts, the bias path
        after now - continued from the true current log bias (mode "oracle") or sampled around the online estimate (mode "realizable": bias_est(mid) -> (b_hat,
        n_eff), the posterior's mean and effective observation count). Apply it to a twin (normally a forked child)
        with apply_trace(twin, trace)."""
        t = twin.now
        tr = copy.copy(self)
        tr.prefix = (int(seed_k), _REDRAW)
        tr.redrawn_at = t
        tr.parent = self
        # O3 of started ops is kept (the view reads it while they run)
        frozen = dict(self._frozen_g)
        for j in twin.jobs.values():
            for r in j.op_rows:
                if r.get("start") is not None and (j.id, r["op"]) not in frozen:
                    frozen[(j.id, r["op"])] = self.est_factor(j.id, r["op"])
        tr._frozen_g = frozen
        # bias: static stays (it is the machine's), the drift continues after the grid point of t
        if self._on["E2d"] or self._on["E2"]:
            i0 = int(t // BIAS_DT)
            carry = {}
            pd = self._p["E2d"]
            s2 = self._p["E2"]["sigma_s"] ** 2
            d2 = pd["sigma_d"] ** 2
            new_stat = dict(self.bstat)
            for mid in self.machines:
                x_t = self.log_drift_at(mid, t)
                if mode == "realizable" and bias_est is not None:
                    b_hat, n_eff = bias_est(mid)
                    # total log bias at t ~ N(log b_hat, v), split between the static and drift parts by variance
                    v = (s2 + d2) * bias_prior_share(n_eff, self.cfg.bias_prior)
                    z = keyed_normal(int(seed_k), _REDRAW, _E2D, mid)
                    total = math.log(max(b_hat, 1e-9)) + math.sqrt(max(v, 0.0)) * z
                    share = d2 / (s2 + d2) if s2 + d2 > 0 else 0.0
                    x_t = total * share
                    new_stat[mid] = math.exp(total * (1.0 - share)) if s2 > 0 else 1.0
                carry[mid] = (self.bdrift_x.get(mid, np.zeros(1)), i0, x_t)
            tr._redraw_bias(i0, carry, int(seed_k))
            tr.bstat = new_stat
        # dropouts: keep outages started before t (the current one ends as drawn), redraw after
        if self._on["O2"]:
            keep = {}
            for mid, iv in self.dropouts.items():
                kept = [(s, e) for s, e in iv if s <= t]
                restart = max([t] + [e for _, e in kept])
                keep[mid] = (kept, restart)
            tr._redraw_dropouts(keep, tr.prefix)
        # the job stream: arrived jobs and known cancellations stay
        arrived = {j.id for j in twin.order}
        tr._build_world(rush_pool, keep={"arrived": arrived, "t": t, "k": twin.k, "parent": self})
        return tr

    def _redraw_bias(self, i0, carry, seed_k):
        pd = self._p["E2d"]
        if not self._on["E2d"]:
            return
        n = int(math.ceil(self.horizon / BIAS_DT)) + 2
        sd, tau, w, T = pd["sigma_d"], pd["tau"], pd["wear"], pd["t_tool"]
        phi = math.exp(-BIAS_DT / tau) if tau > 0 else 0.0
        innov = sd * math.sqrt(max(0.0, 1.0 - phi * phi))
        t_grid = np.arange(n) * BIAS_DT
        self.bdrift_x, self.bdyn = dict(self.bdrift_x), dict(self.bdyn)
        for mid in self.machines:
            parent_x, _, x_split = carry[mid]
            z = _stream(seed_k, _REDRAW, _E2D, mid).standard_normal(n)
            x = np.empty(n)
            m = min(i0 + 1, len(parent_x), n)
            x[:m] = parent_x[:m]
            xi = x_split
            for i in range(m, n):
                xi = phi * xi + innov * z[i]
                x[i] = xi
            self.bdrift_x[mid] = x
            dyn = np.exp(x - 0.5 * sd * sd)
            if w > 0 and T > 0:
                phase = keyed_uniform(self.seed, self.draw, _WEAR, mid) * T
                dyn = dyn * (1.0 + w * np.mod(t_grid + phase, T) / T) / (1.0 + 0.5 * w)
            self.bdyn[mid] = dyn

    def _redraw_dropouts(self, keep, prefix):
        rate, mean = self._p["O2"]["rate"], self._p["O2"]["mean"]
        self.dropouts = {}
        for mid in self.machines:
            g = _stream(*prefix, _O2, mid)
            out, t = list(keep[mid][0]), keep[mid][1]
            while t < self.horizon:
                e = g.standard_exponential(2)
                s = t + e[0] / rate
                end = s + e[1] * mean
                out.append((s, end))
                t = end
            self.dropouts[mid] = out
        self._dstarts = {mid: [s for s, _ in iv] for mid, iv in self.dropouts.items()}

    # ── serialization (the Unity injection format for execution noise, plan §4) ─────────────────────────────

    def to_json(self, schedule=False):
        """{"schema": "uncertainty/1", config, seeds, horizon}; with schedule=True also the execution-noise injection
        data: per world job and op the E1 factor per eligible machine ("actualFactors", parallel to the eligible list)
        and the per-machine bias schedule on its 60 s grid ("machineBiasSchedule")."""
        out = {"schema": "uncertainty/1", "config": self.cfg.to_dict(), "noiseSeed": self.seed, "draw": self.draw,
               "horizon": self.horizon}
        if schedule:
            if self.redrawn_at is not None:
                raise ValueError("a redrawn trace has no injection schedule")
            out["actualFactors"] = {str(jd["id"]): [[self.factor(int(jd["id"]), o, int(m)) for m, _ in op["eligible"]]
                                                    for o, op in enumerate(jd["ops"])]
                                    for jd in self.world_jobs["jobs"]}
            out["machineBiasSchedule"] = {"dt": BIAS_DT, "machines": {
                str(mid): ([self.bstat[mid]] if self.bdyn.get(mid) is None else
                           [self.bstat[mid] * float(v) for v in self.bdyn[mid]]) for mid in self.machines}}
        return out

    @classmethod
    def from_json(cls, data, plan_jobs, floor, rush_pool=None):
        """Rebuilds the trace from to_json() output (the draws are a function of config + seeds)."""
        if data.get("schema") != "uncertainty/1":
            raise ValueError("not an uncertainty/1 block")
        return cls.build(plan_jobs, floor, UncertaintyConfig.from_dict(data["config"]), data["noiseSeed"],
                         horizon=data.get("horizon"), rush_pool=rush_pool, draw=data.get("draw", 0))


def bias_prior_share(n_eff, prior):
    """Posterior variance share of a normal-normal estimate with `prior` pseudo-observations and n_eff data points."""
    return prior / (prior + max(n_eff, 0.0)) if prior > 0 else 0.0


class _PlanJob:
    __slots__ = ("arrival", "ops", "types", "due")

    def __init__(self, arrival, ops, types, due):
        self.arrival, self.ops, self.types, self.due = arrival, ops, types, due


def apply_trace(twin, trace):
    """@brief Swaps a redrawn trace (UncertaintyTrace.redraw_after) into a running twin, normally in a forked child:
    the future job stream (arrivals not yet happened) is replaced by the trace's, future operations draw their factors
    and bias from it, future bookings and outages follow it. Bookings already emitted keep their visibility times."""
    twin.unc = trace
    twin.cfg.uncertainty = trace
    if getattr(twin, "view", None) is not None and twin.view is not twin:
        twin.view.core.tr = trace
    arrived = {j.id for j in twin.order}
    future = [jd for jd in trace.world_jobs["jobs"] if int(jd["id"]) not in arrived]
    twin.replace_pending(future)
