"""Machine failure models and failure traces for the twin (2026-10-09, dev-twin-failures; plan
docs/Plans/TWIN_FAILURES_AND_HETERO_HEADROOM_PLAN_2026-10-09.md, phase A).

Unity's model (PhysicalMachine.TickTTF / TickRepair, StochasticEventManager, FailureCoordinator):
- Time to failure (TTF) is a float32 countdown decremented by dt every tick the machine is Operational, idle or busy
  (calendar-time aging). When it reaches <= 0 the machine is Failed and the repair duration is drawn at once; the
  orchestrator harvests the flag the next tick (FactoryOrchestrator ticks before the machines).
- The repair is a float32 countdown that starts on the harvest tick (AcknowledgeFailure, then the machine's own
  FixedUpdate in the same tick) and is harvested the tick after it reaches <= 0. Repair is as-good-as-new: a fresh
  TTF is drawn at the repair harvest and counts down from that tick.
- Episode start: fullTtf = Weibull draw, countdown = UnityEngine.Random.Range(0, fullTtf), i.e. U x T with an
  ordinary T (not the equilibrium residual life; mean 0.45 lambda instead of 0.66 lambda at k = 1.5).
- Every machine shares one Weibull(k, lambda) / lognormal(mu, sigma) model and one System.Random stream, so the draws
  cannot be matched from Python. What CAN be matched is the semantics: because aging is on calendar time, the failure
  trace depends only on the seed, never on the policy.

A trace here is therefore exogenous data the engine consumes: per machine, the failures as (harvest tick F, repair
seconds), with F computed from the drawn durations by exactly the tick rules above (`harvest_tick`), so the same
durations injected into Unity (results/dev-twin-failures/CSHARP_INJECTION_SPEC.md) would fail on the same ticks.
`MachineTrace.lives` keeps the drawn countdowns for that export. Every policy and every forced choice run on one
trace sees the same failures (common random numbers).

Models (all calendar-time, all seeded per machine and purpose with numpy PCG64, so changing one machine's parameters
does not move any other machine's draws, and a model's repair draws do not depend on its TTF parameters):
- `WeibullModel`: per-machine Weibull TTF / lognormal repair. `uniform_from_stochastic` builds Unity's global model
  from a scenario's "stochastic" block (initial-age convention "unity" = U x T); `hetero` gives a subset of machines a
  shorter life (fixed set across episodes, or redrawn per episode); renewal="minimal" (as-bad-as-old: age keeps
  growing across repairs) gives a degrading machine.
- `RegimeModel`: each machine is a hidden two-state Markov chain (good / bad) with proportional Weibull hazards; the
  bad state shortens the life, and which machines are bad changes within an episode.
- Availability helpers hold the floor-mean availability fixed while heterogeneity changes.

Conditional redraw (`FailureTrace.redraw_after`): keep everything observable at tick k (failures already harvested,
the current repair and its end, each machine's age) and redraw the rest from the model, for scoring rollouts over
futures that use current-state knowledge only.

AGV failures are not modelled (TODO): Unity ages AGVs on operating time, so their trace depends on the policy and
would need native sampling in the engine plus the zone-holding of a broken AGV, which the free-flow twin lacks.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .motion import f32_countdown

DT = 0.02
# Purposes of the per-machine random streams (SeedSequence([seed, machine, purpose])).
_LIFE, _REPAIR, _INIT, _SUBSET, _REGIME = 1, 2, 3, 4, 5
# Unity's StochasticConfig defaults.
UNITY_DEFAULTS = {"weibullK": 1.5, "weibullLambda": 900.0, "repairLogMu": 4.0, "repairLogSigma": 0.5}


def _rng(seed, mid, purpose):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(mid), purpose])))


def harvest_tick(start_tick, duration, dt=DT):
    """Tick the orchestrator harvests a float32 countdown that starts at `duration` and is decremented by dt from tick
    `start_tick` on (decrement i at tick start_tick + i - 1, flag at the first value <= 0, harvested next tick)."""
    return start_tick + max(1, f32_countdown(np.float32(duration), dt, until=0.0)[0])


def remaining_countdown(duration, dt, steps):
    """Value of the float32 countdown after `steps` decrements (>= 0), as RemainingRepairTime."""
    return max(0.0, float(f32_countdown(np.float32(duration), dt, steps=max(0, steps))[1]))


# ── Reliability parameters and availability ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Rel:
    """Weibull(k, lam) time to failure, lognormal(mu, sigma) repair (seconds)."""
    k: float = 1.5
    lam: float = 7200.0
    mu: float = 6.3
    sigma: float = 0.5

    @property
    def mtbf(self):
        return mtbf(self.k, self.lam)

    @property
    def mttr(self):
        return mttr(self.mu, self.sigma)

    @property
    def availability(self):
        return self.mtbf / (self.mtbf + self.mttr)


def mtbf(k, lam):
    return lam * math.gamma(1.0 + 1.0 / k)


def mttr(mu, sigma):
    return math.exp(mu + sigma * sigma / 2.0)


def availability(k, lam, mu, sigma):
    a, r = mtbf(k, lam), mttr(mu, sigma)
    return a / (a + r)


def lam_for_availability(target, k=1.5, mu=6.3, sigma=0.5):
    """Weibull lambda that gives long-run availability `target` with this repair model."""
    if not 0.0 < target < 1.0:
        raise ValueError(f"availability must be in (0, 1), got {target}")
    return mttr(mu, sigma) * target / (1.0 - target) / math.gamma(1.0 + 1.0 / k)


def _bisect(f, lo, hi, iters=200):
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def hetero_lams_for_availability(target, ratio, fraction, k=1.5, mu=6.3, sigma=0.5):
    """(lam_reliable, lam_unreliable): a `fraction` of machines fail `ratio` times as often (lambda / ratio), and the
    floor-mean long-run availability is `target`."""
    def excess(lam):
        a = (1 - fraction) * availability(k, lam, mu, sigma) + fraction * availability(k, lam / ratio, mu, sigma)
        return a - target
    lam = _bisect(excess, 1.0, 1e9)
    return lam, lam / ratio


def regime_lams_for_availability(target, ratio, bad_fraction, k=1.5, mu=6.3, sigma=0.5):
    """(lam_good, lam_bad) for a regime model whose machines spend `bad_fraction` of the time in the bad state, where
    the life is `ratio` times shorter (lam_bad = lam_good / ratio). Approximation: failure rate per up-second
    pi_g / MTBF_g + pi_b / MTBF_b, availability 1 / (1 + rate x MTTR); exact for k = 1 and slow switching."""
    rate = (1.0 - target) / target / mttr(mu, sigma)
    lam_good = ((1.0 - bad_fraction) + bad_fraction * ratio) / rate / math.gamma(1.0 + 1.0 / k)
    return lam_good, lam_good / ratio


# ── Traces ──────────────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class MachineTrace:
    """One machine's failures. events: [(F, repair)] with F the tick the orchestrator harvests the failure and repair
    the drawn repair duration (seconds, float32 value). lives: the TTF countdowns as drawn (lives[0] the episode-start
    countdown, lives[i] the one drawn at the i-th repair harvest; the last may run past the horizon). age0: the age at
    t = 0 (Unity's _ageSinceLastRepair after InitializeStochastic). spliced: tick from which the events were redrawn
    (None for a trace drawn whole; such a trace cannot be exported as a Unity schedule)."""
    events: list = field(default_factory=list)
    lives: list = field(default_factory=list)
    age0: float = 0.0
    spliced: int = None
    regime: list = None          # RegimeModel: the hidden state path [(t, state)] (oracle information)


class FailureTrace:
    """Per-machine failure events for one episode (TwinConfig.failures). `model` is the generator (for static and
    oracle reliability information and conditional redraws)."""

    def __init__(self, machines, model, seed, horizon, dt=DT):
        self.machines = machines            # {machine id: MachineTrace}
        self.model, self.seed, self.horizon, self.dt = model, seed, horizon, dt

    def fail_times(self, mid):
        """[(failure time in seconds (harvest tick), repair duration)] for one machine."""
        return [(f * self.dt, r) for f, r in self.machines[mid].events]

    def intervals(self, mid):
        """[(F, R)] harvest ticks of each failure and of its repair."""
        return [(f, harvest_tick(f, r, self.dt)) for f, r in self.machines[mid].events]

    def downtime(self, mid, t_end):
        """Seconds machine `mid` is down (harvested failure to harvested repair) in [0, t_end]."""
        k_end = t_end / self.dt
        return sum(max(0.0, min(r, k_end) - f) for f, r in self.intervals(mid) if f < k_end) * self.dt

    def n_failures(self):
        return sum(len(m.events) for m in self.machines.values())

    def redraw_after(self, tick, seed, regime="keep"):
        """A copy whose events after tick `tick` are redrawn from the model with streams seeded by `seed`, keeping what
        is observable at that tick: failures harvested at or before it (and the end of a repair in progress) and each
        machine's age. regime="keep" continues each hidden regime from its true state at that time (oracle-informed);
        "stationary" draws it from the stationary distribution (no regime knowledge)."""
        out = {}
        for mid, mt in self.machines.items():
            out[mid] = self.model.redraw_machine(mid, mt, tick, seed, self, regime)
        tr = FailureTrace(out, self.model, seed, self.horizon, self.dt)
        tr.parent, tr.redrawn_at = self, tick
        return tr

    def to_schedule(self):
        """The Unity failureSchedule (CSHARP_INJECTION_SPEC.md): per machine the initial countdown, the repair
        durations and the TTF drawn at each repair harvest, in draw order."""
        sched = {}
        for mid, mt in self.machines.items():
            if mt.spliced is not None:
                raise ValueError("a redrawn trace has no Unity schedule (its current life was re-sampled mid-way)")
            sched[str(mid)] = {"initialCountdown": float(mt.lives[0]) if mt.lives else None,
                               "repairs": [float(r) for _, r in mt.events],
                               "ttfs": [float(x) for x in mt.lives[1:]]}
        return {"dt": self.dt, "machines": sched}

    @classmethod
    def from_schedule(cls, sched, horizon=None, dt=None, model=None):
        """Rebuilds the events from a schedule by the tick rules (the inverse of to_schedule)."""
        dt = dt or sched.get("dt", DT)
        horizon = horizon or math.inf
        machines = {}
        for key, s in sched["machines"].items():
            mt = MachineTrace(lives=[s["initialCountdown"]] + list(s["ttfs"]))
            start, ttf = 1, s["initialCountdown"]
            for i, rep in enumerate(s["repairs"]):
                f = harvest_tick(start, ttf, dt)
                mt.events.append((f, float(rep)))
                start = harvest_tick(f, rep, dt)
                if i < len(s["ttfs"]):
                    ttf = s["ttfs"][i]
            machines[int(key)] = mt
        return cls(machines, model, None, horizon, dt)


# ── Models ──────────────────────────────────────────────────────────────────────────────────────────────────

def _f32(x):
    return float(np.float32(x))


def _weibull_cond(rng, k, lam, age):
    """Residual life given survival to `age`: lam ((age/lam)^k - ln U)^(1/k) - age."""
    e = rng.standard_exponential()
    return lam * ((age / lam) ** k + e) ** (1.0 / k) - age


def _equilibrium_age(rng, k, lam):
    """Age (= residual life) of a stationary renewal process: U x length-biased T, (T/lam)^k ~ Gamma(1 + 1/k)
    (StochasticEventManager.SampleAGVResidualTTF)."""
    return rng.random() * lam * rng.gamma(1.0 + 1.0 / k) ** (1.0 / k)


class WeibullModel:
    """Per-machine Weibull TTF / lognormal repair in calendar time.

    @param params    {machine id: Rel}, or one Rel for every machine
    @param initial   "unity": countdown U x T with T a full Weibull draw (Unity's InitializeStochastic, age T - U x T);
                     "equilibrium": age and residual of a stationary renewal process (the plan's choice for new
                     conditions); "fresh": age 0, a full first life
    @param renewal   "good_as_new" (Unity: age 0 after a repair) or "minimal" (as-bad-as-old: the operating age keeps
                     growing, so the hazard rises over the episode when k > 1 - a degradation model)
    @param pm_interval  with renewal="minimal": a preventive-maintenance renewal (age back to 0, no downtime) every
                     this many operating seconds; None = never
    @param static_info  what the "static" information level (the datasheet) knows: "machine" = each machine's own
                     long-run parameters; "pooled" = only the floor-wide pooled values (mean failure rate, mean
                     repair), the same for every machine - for a plant whose unreliable machines are not known in
                     advance (condition b', the set redrawn per episode). current_rel (oracle) is per machine either
                     way."""

    def __init__(self, params, initial="unity", renewal="good_as_new", pm_interval=None, unreliable=(),
                 static_info="machine"):
        if initial not in ("unity", "equilibrium", "fresh"):
            raise ValueError(initial)
        if renewal not in ("good_as_new", "minimal"):
            raise ValueError(renewal)
        if static_info not in ("machine", "pooled"):
            raise ValueError(static_info)
        self.params, self.initial, self.renewal, self.pm_interval = params, initial, renewal, pm_interval
        self.unreliable = frozenset(unreliable)
        self.static_info = static_info
        self._pooled = None

    def rel(self, mid):
        return self.params if isinstance(self.params, Rel) else self.params[mid]

    def _machine_rel(self, mid):
        r = self.rel(mid)
        return r.k, r.lam, r.mtbf, r.mttr

    def pooled_rel(self):
        """Floor-wide (k, lam, mtbf, mttr): the mean failure rate (MTBF = 1 / mean of 1 / MTBF_i, the floor's
        failures per up-second) and the mean repair; k of the first machine (the models use one shape)."""
        if self._pooled is None:
            rels = [self.params] if isinstance(self.params, Rel) else [self.params[m] for m in sorted(self.params)]
            k = rels[0].k
            m = len(rels) / sum(1.0 / r.mtbf for r in rels)
            self._pooled = (k, m / math.gamma(1.0 + 1.0 / k), m, sum(r.mttr for r in rels) / len(rels))
        return self._pooled

    # information for reliability-aware rules
    def static_rel(self, mid):
        """Long-run (k, lam, mtbf, mttr) the plant's datasheet would give (per machine, or pooled: static_info)."""
        if self.static_info == "pooled":
            return self.pooled_rel()
        return self._machine_rel(mid)

    def current_rel(self, mid, t, trace=None):
        """True current (k, lam, mtbf, mttr): each machine's own long-run parameters for a stationary model."""
        return self._machine_rel(mid)

    def _age_after(self, age, ttf_ticks, dt):
        if self.renewal == "good_as_new":
            return 0.0
        a = age + ttf_ticks * dt
        if self.pm_interval:
            a = a % self.pm_interval
        return a

    def _residual(self, rng, mid, age, start_s, ctx):
        r = self.rel(mid)
        if self.renewal == "minimal" and self.pm_interval:
            # a PM renewal at operating age pm_interval resets the hazard: draw piecewise
            total, a = 0.0, age
            while True:
                x = _weibull_cond(rng, r.k, r.lam, a)
                if a + x <= self.pm_interval:
                    return total + x
                total += self.pm_interval - a
                a = 0.0
        return _weibull_cond(rng, r.k, r.lam, age)

    def _start(self, rng_init, rng_life, mid, ctx):
        """(first countdown, age at t = 0)."""
        r = self.rel(mid)
        if self.initial == "unity":
            full = _f32(r.lam * (-math.log(1.0 - rng_init.random())) ** (1.0 / r.k))
            c0 = rng_init.random() * full
            return c0, full - c0
        if self.initial == "equilibrium":
            a0 = _equilibrium_age(rng_init, r.k, r.lam)
            return self._residual(rng_life, mid, a0, 0.0, ctx), a0
        return self._residual(rng_life, mid, 0.0, 0.0, ctx), 0.0

    def _ctx(self, mid, seed, horizon):
        return None

    def trace(self, machine_ids, seed, horizon, dt=DT):
        """Draws a FailureTrace for these machines until `horizon` seconds."""
        out = {}
        for mid in machine_ids:
            ctx = self._ctx(mid, seed, horizon)
            life, rep, init = _rng(seed, mid, _LIFE), _rng(seed, mid, _REPAIR), _rng(seed, mid, _INIT)
            c0, a0 = self._start(init, life, mid, ctx)
            mt = MachineTrace(age0=a0)
            self._extend(mt, mid, 1, _f32(c0), a0, life, rep, ctx, horizon, dt)
            out[mid] = mt
        return FailureTrace(out, self, seed, horizon, dt)

    def _extend(self, mt, mid, start, ttf, age, life, rep, ctx, horizon, dt):
        """Appends events: the countdown `ttf` starts at tick `start` with the machine at operating age `age`."""
        r = self.rel(mid)
        mt.lives.append(ttf)
        while True:
            if ttf * 1.0 > horizon + 1.0 or start * dt > horizon:
                return
            f = harvest_tick(start, ttf, dt)
            if f * dt > horizon:
                return
            repair = _f32(math.exp(r.mu + r.sigma * rep.standard_normal()))
            mt.events.append((f, repair))
            age = self._age_after(age, f - start, dt)
            start = harvest_tick(f, repair, dt)
            ttf = _f32(self._residual(life, mid, age, start * dt, ctx))
            mt.lives.append(ttf)

    def _age_at(self, mt, mid, tick, dt):
        """(start tick of the life in progress at `tick`, operating age at `tick`, down until tick R or None)."""
        start, age = 1, mt.age0
        for f, rep in mt.events:
            if f > tick:
                break
            r_tick = harvest_tick(f, rep, dt)
            age_f = self._age_after(age, f - start, dt)
            if r_tick > tick:                    # down at `tick`: the repair in progress is kept
                return r_tick, age_f, r_tick
            start, age = r_tick, age_f
        age = age + (tick - start) * dt
        if self.pm_interval and self.renewal == "minimal":
            age = age % self.pm_interval
        return start, age, None

    def redraw_machine(self, mid, mt, tick, seed, trace, regime):
        dt = trace.dt
        kept = [(f, r) for f, r in mt.events if f <= tick]
        _, age, down_until = self._age_at(mt, mid, tick, dt)
        life, rep = _rng(seed, mid, _LIFE), _rng(seed, mid, _REPAIR)
        ctx = self._redraw_ctx(mid, seed, trace, tick, regime)
        new = MachineTrace(events=kept, lives=[], age0=mt.age0, spliced=tick)
        if down_until is not None:
            start, a = down_until, age
        else:
            start, a = tick, age
        ttf = _f32(self._residual(life, mid, a, start * dt, ctx))
        self._extend(new, mid, start, ttf, a, life, rep, ctx, trace.horizon, dt)
        return new

    def _redraw_ctx(self, mid, seed, trace, tick, regime):
        return None


class RegimeModel(WeibullModel):
    """Each machine is a hidden two-state Markov chain over calendar time: good (Weibull lam_good) and bad (lam_bad),
    switching good -> bad at rate 1 / mean_good and bad -> good at rate 1 / mean_bad (seconds). The hazard is
    proportional within a life: H over a stretch spent in state s with age a1 -> a2 is (a2/lam_s)^k - (a1/lam_s)^k.
    The regime path is drawn once per machine from its own stream; its stationary bad share is
    mean_bad / (mean_good + mean_bad)."""

    def __init__(self, k=1.5, lam_good=15000.0, lam_bad=1900.0, mean_good=16200.0, mean_bad=5400.0, mu=6.3,
                 sigma=0.5, initial="equilibrium"):
        super().__init__(Rel(k, lam_good, mu, sigma), initial=initial)
        self.k, self.lam = k, (lam_good, lam_bad)
        self.mean = (mean_good, mean_bad)
        self.mu, self.sigma = mu, sigma

    @property
    def bad_share(self):
        return self.mean[1] / (self.mean[0] + self.mean[1])

    def rel(self, mid):
        return Rel(self.k, self.lam[0], self.mu, self.sigma)

    def _path(self, rng, t0, state, horizon):
        """[(t, state)] switch points from t0 (state at t0) until past the horizon."""
        path, t = [(t0, state)], t0
        while t <= horizon:
            t += rng.exponential(self.mean[state])
            state = 1 - state
            path.append((t, state))
        return path

    def _ctx(self, mid, seed, horizon):
        rng = _rng(seed, mid, _REGIME)
        s0 = int(rng.random() < self.bad_share)
        return self._path(rng, 0.0, s0, horizon + 4 * max(self.mean))

    def state_at(self, path, t):
        s = path[0][1]
        for ti, si in path:
            if ti > t:
                break
            s = si
        return s

    def static_rel(self, mid):
        """Long-run mix: the stationary failure rate (per up-second) of the two states."""
        pb = self.bad_share
        rate = (1 - pb) / mtbf(self.k, self.lam[0]) + pb / mtbf(self.k, self.lam[1])
        m = 1.0 / rate
        lam = m / math.gamma(1.0 + 1.0 / self.k)
        return self.k, lam, m, mttr(self.mu, self.sigma)

    def current_rel(self, mid, t, trace=None):
        path = trace.machines[mid].regime if trace is not None else None
        s = self.state_at(path, t) if path else 0
        lam = self.lam[s]
        return self.k, lam, mtbf(self.k, lam), mttr(self.mu, self.sigma)

    def _start(self, rng_init, rng_life, mid, ctx):
        s0 = ctx[0][1]
        if self.initial == "unity":
            lam = self.lam[s0]
            full = _f32(lam * (-math.log(1.0 - rng_init.random())) ** (1.0 / self.k))
            c0 = rng_init.random() * full
            return c0, full - c0
        a0 = _equilibrium_age(rng_init, self.k, self.lam[s0]) if self.initial == "equilibrium" else 0.0
        return self._residual(rng_life, mid, a0, 0.0, ctx), a0

    def _residual(self, rng, mid, age, start_s, ctx):
        """Inverts the cumulative hazard along the regime path from time start_s (age `age`)."""
        e = rng.standard_exponential()
        k = self.k
        t, a = start_s, age
        i = 0
        while i + 1 < len(ctx) and ctx[i + 1][0] <= t:
            i += 1
        while True:
            s = ctx[i][1]
            lam = self.lam[s]
            t_next = ctx[i + 1][0] if i + 1 < len(ctx) else math.inf
            a_next = a + (t_next - t)
            cap = (a_next / lam) ** k - (a / lam) ** k if math.isfinite(t_next) else math.inf
            if e <= cap:
                a_star = lam * ((a / lam) ** k + e) ** (1.0 / k)
                return (t - start_s) + (a_star - a)
            e -= cap
            t, a = t_next, a_next
            i += 1

    def trace(self, machine_ids, seed, horizon, dt=DT):
        tr = super().trace(machine_ids, seed, horizon, dt)
        for mid in machine_ids:
            tr.machines[mid].regime = self._ctx(mid, seed, horizon)
        return tr

    def _redraw_ctx(self, mid, seed, trace, tick, regime):
        t = tick * trace.dt
        old = trace.machines[mid].regime
        rng = _rng(seed, mid, _REGIME)
        if regime == "keep" and old:
            s = self.state_at(old, t)
        elif regime == "stationary":
            s = int(rng.random() < self.bad_share)
        else:
            raise ValueError(regime)
        past = [(ti, si) for ti, si in (old or []) if ti <= t]
        path = past + self._path(rng, t, s, trace.horizon + 4 * max(self.mean))
        return path

    def redraw_machine(self, mid, mt, tick, seed, trace, regime):
        new = super().redraw_machine(mid, mt, tick, seed, trace, regime)
        new.regime = self._redraw_ctx(mid, seed, trace, tick, regime)
        return new


# ── Builders ────────────────────────────────────────────────────────────────────────────────────────────────

def uniform_from_stochastic(block, initial="unity"):
    """Unity's global model from a scenario "stochastic" block (StochasticConfig field names and defaults)."""
    g = {**UNITY_DEFAULTS, **{k: v for k, v in (block or {}).items() if k in UNITY_DEFAULTS}}
    return WeibullModel(Rel(float(g["weibullK"]), float(g["weibullLambda"]), float(g["repairLogMu"]),
                            float(g["repairLogSigma"])), initial=initial)


def pick_subset(machine_types, fraction, seed):
    """`fraction` of the machines of each type (rounded), drawn with a seeded stream; machine_types {id: type}.
    Spreading the subset over types keeps reliable alternatives for every operation."""
    rng = _rng(seed, 0, _SUBSET)
    by_type = {}
    for mid in sorted(machine_types):
        by_type.setdefault(machine_types[mid], []).append(mid)
    out = []
    for t in sorted(by_type):
        ids = by_type[t]
        n = int(round(fraction * len(ids)))
        out += [ids[i] for i in sorted(rng.choice(len(ids), size=n, replace=False))]
    return sorted(out)


def hetero(machine_types, ratio=4.0, fraction=0.4, availability_target=0.90, k=1.5, mu=6.3, sigma=0.5,
           subset_seed=0, initial="equilibrium", static_info="machine"):
    """Stationary heterogeneous reliability: `fraction` of each type fails `ratio` times as often, with the floor-mean
    availability held at `availability_target`. subset_seed fixed = the same machines in every episode; pass the
    episode seed to redraw the set per episode. The unreliable ids are on model.unreliable. static_info: see
    WeibullModel ("pooled" for a redrawn set the datasheet cannot know)."""
    lam_r, lam_u = hetero_lams_for_availability(availability_target, ratio, fraction, k, mu, sigma)
    bad = set(pick_subset(machine_types, fraction, subset_seed))
    params = {mid: Rel(k, lam_u if mid in bad else lam_r, mu, sigma) for mid in machine_types}
    return WeibullModel(params, initial=initial, unreliable=bad, static_info=static_info)


def regime(ratio=8.0, bad_fraction=0.25, mean_bad=5400.0, availability_target=0.90, k=1.5, mu=6.3, sigma=0.5,
           initial="equilibrium"):
    """Non-stationary reliability: a RegimeModel whose bad state is `ratio` times shorter-lived, with a stationary bad
    share `bad_fraction`, mean bad sojourn `mean_bad` seconds and (approximately) the given floor availability."""
    lam_g, lam_b = regime_lams_for_availability(availability_target, ratio, bad_fraction, k, mu, sigma)
    mean_good = mean_bad * (1.0 - bad_fraction) / bad_fraction
    return RegimeModel(k, lam_g, lam_b, mean_good, mean_bad, mu, sigma, initial=initial)


def degradation(k=2.5, lam=None, availability_target=0.90, mu=6.3, sigma=0.5, pm_interval=None, initial="fresh"):
    """Minimal repair (as-bad-as-old): the hazard keeps rising with operating age (k > 1), optionally renewed by a
    zero-duration preventive maintenance every pm_interval operating seconds. lam defaults to the value that gives
    `availability_target` for a renewal process (the realized availability is lower: the machines degrade)."""
    lam = lam or lam_for_availability(availability_target, k, mu, sigma)
    return WeibullModel(Rel(k, lam, mu, sigma), initial=initial, renewal="minimal", pm_interval=pm_interval)


# ── Online reliability estimate (R-online) ──────────────────────────────────────────────────────────────────

class OnlineReliability:
    """Per-machine failure rate and repair time estimated from the failures a twin has observed so far (warm-up
    included): the information a plant could track without a datasheet per machine.

    Failure rate: Gamma-Poisson posterior with exponential forgetting (half-life `half_life` seconds, so it tracks
    drifting rates). Per machine it keeps the decayed failure count n_i and decayed up-time T_i; the prior is the
    pooled floor-wide rate r_floor = (a0 + N) / (a0 x MTBF_prior + T) with shape a0 pseudo-failures, MTBF_prior the
    floor-mean MTBF of the model (the floor average is the only datasheet value it gets), and
    rate_i = (a0 + n_i) / (a0 / r_floor + T_i). Repair time: decayed mean of the machine's observed repairs shrunk
    towards the floor's observed mean (or the model's MTTR) with one pseudo-repair. The Weibull shape k is taken
    from the model (floor-wide). Estimates are cached per tick."""

    def __init__(self, twin, half_life=7200.0, a0=2.0):
        self.twin, self.h, self.a0 = twin, float(half_life), float(a0)
        self.k_shape = 1.0
        model = twin.fail.model
        ids = [m.id for m in twin.machines]
        if model is not None and ids:
            stats = [model.static_rel(mid) for mid in ids]
            self.k_shape = stats[0][0]
            # floor MTBF = 1 / mean failure rate (the pooled rate per up-second; the arithmetic mean of the MTBFs
            # overstated it by ~35% on the hetero conditions before 2026-10-10)
            self.mtbf0 = len(stats) / sum(1.0 / s[2] for s in stats)
            self.mttr0 = sum(s[3] for s in stats) / len(stats)
        else:
            self.mtbf0, self.mttr0 = math.inf, 0.0
        self.st = {mid: [0.0, 0.0, 0.0, 0.0, 0.0, True] for mid in ids}   # n, T, repair sum, repair count, t, up
        self._cache_tick, self._cache = None, {}

    def _decay(self, s, t):
        dt = t - s[4]
        if dt > 0:
            f = 0.5 ** (dt / self.h)
            s[0] *= f
            s[2] *= f
            s[3] *= f
            s[1] = s[1] * f + (self.h / math.log(2.0) * (1.0 - f) if s[5] else 0.0)
            s[4] = t
        return s

    def on_fail(self, m):
        s = self._decay(self.st[m.id], self.twin.now)
        s[0] += 1.0
        s[5] = False
        self._cache_tick = None

    def on_repair(self, m):
        s = self._decay(self.st[m.id], self.twin.now)
        s[2] += float(m.repair_dur)
        s[3] += 1.0
        s[5] = True
        self._cache_tick = None

    def estimate(self, m):
        """(k, lam, mtbf, mttr) of machine m now."""
        if self._cache_tick != self.twin.k:
            t = self.twin.now
            for s in self.st.values():
                self._decay(s, t)
            n_tot = sum(s[0] for s in self.st.values())
            t_tot = sum(s[1] for s in self.st.values())
            r_tot = sum(s[2] for s in self.st.values())
            c_tot = sum(s[3] for s in self.st.values())
            if not math.isfinite(self.mtbf0):
                self._cache = {mid: (1.0, math.inf, math.inf, 0.0) for mid in self.st}
            else:
                r_floor = (self.a0 + n_tot) / (self.a0 * self.mtbf0 + t_tot)
                mttr_floor = (self.mttr0 + r_tot) / (1.0 + c_tot)
                k = self.k_shape
                g = math.gamma(1.0 + 1.0 / k)
                self._cache = {}
                for mid, s in self.st.items():
                    rate = (self.a0 + s[0]) / (self.a0 / r_floor + s[1])
                    mtbf_ = 1.0 / rate
                    mttr_ = (mttr_floor + s[2]) / (1.0 + s[3])
                    self._cache[mid] = (k, mtbf_ / g, mtbf_, mttr_)
            self._cache_tick = self.twin.k
        return self._cache[m.id]


# ── Simulated availability calibration ──────────────────────────────────────────────────────────────────────

def scaled(model, s):
    """A copy of a WeibullModel / RegimeModel with every Weibull lambda multiplied by s (lives s times longer)."""
    import copy
    m = copy.copy(model)
    m._pooled = None
    if isinstance(model, RegimeModel):
        m.lam = (model.lam[0] * s, model.lam[1] * s)
    elif isinstance(model.params, Rel):
        r = model.params
        m.params = Rel(r.k, r.lam * s, r.mu, r.sigma)
    else:
        m.params = {mid: Rel(r.k, r.lam * s, r.mu, r.sigma) for mid, r in model.params.items()}
    return m


def simulated_availability(model, machine_ids, horizon=400000.0, seed=0, dt=DT):
    tr = model.trace(machine_ids, seed, horizon, dt)
    return float(np.mean([1.0 - tr.downtime(m, horizon) / horizon for m in machine_ids]))


def calibrate_availability(model, target, machine_ids=range(200), horizon=400000.0, seed=0, iters=30):
    """`model` with its lambdas scaled so the simulated floor-mean availability (over `horizon`, the given machines,
    one fixed seed) is `target`: for models whose closed-form helper is approximate (regime switching, degradation).
    Deterministic; returns (calibrated model, scale)."""
    ids = list(machine_ids)
    lo, hi = 0.05, 20.0
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if simulated_availability(scaled(model, mid), ids, horizon, seed) > target:
            hi = mid
        else:
            lo = mid
    s = math.sqrt(lo * hi)
    return scaled(model, s), s
