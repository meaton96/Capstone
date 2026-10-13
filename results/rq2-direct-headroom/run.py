"""
@file run.py
@brief rq2-direct-headroom (2026-10-07): does choosing jobs and machines directly (the GNN / end-to-end action space)
       have more headroom than choosing among the 15 H15 rule pairs? One-step rollout in the twin, both action spaces
       on the same base policy, instance and decisions.

Why (user, 10-07): the fallback plan is a GNN that picks job and machine directly (L2D / Song 2022 style). Those are
hard to train, so check first that the bigger action space buys anything here. Rule selection is capped by its rule
set (Link et al. 2026, arXiv:2604.24117); direct choice is not, but a better non-rule choice has to exist.

Method (Bertsekas rollout, one step): base policy MDD-TECT (best fixed pair on B2). For every agent decision in the
first --rollout-seconds of the window, in order: try each option, play the rest of the episode with the base, keep
the option with the lowest window tardiness, then move to the next decision with the kept choices replayed.
Scoring an option (--futures K):
  - K = 0 (hindsight): on the instance's own future. A clairvoyant bound: one decision's effect on one realized
          future predicts its sign on other futures only ~50% of the time (credit trace 10-07), so this mostly
          measures how far chaos can be exploited with foresight.
  - K > 0 (expected): mean over K redrawn futures: arrivals after the decision time are replaced by those of other B2
          instances (seeds 50000+, splice as in the 10-07 credit trace; fleet schedule and regime blocks kept). A
          policy that knows only the current state could do this. The kept choices are then scored on the real
          instance.
Options:
  - rule:   the distinct (job, machine) outcomes of the 15 H15 pairs at that decision (aliased pairs counted once)
  - direct: every legal (job, machine): a dispatch picks any queued job; a routing picks any routable job and any
            eligible machine for it. Over --cap options: all rule outcomes plus a seeded sample of the rest.
Both variants make the same number of decisions on the same instance, so direct - rule is the extra headroom of the
bigger action space (direct options include the rule outcomes, so direct can only lose by the cap or greediness).
Window tardiness as rq2-oracle-steady/sim.py (fixed pairs here equal rq2-twin-fleet's B2 numbers).

Forcing a choice: rank_jobs / select_machine in des_twin.engine are wrapped to return a preset job / machine for the
next decision only; the twin is deterministic, so a replay with the same forced choices reproduces the run exactly
(checked at startup: forcing the base's own choices gives the base result).

One call = one (setting, seed, variant, futures) -> <setting>/<variant>-<hindsight|expectedK>/s<seed>.json (skips
existing).
@par Usage
@code{.sh}
python results/rq2-direct-headroom/run.py --setting B2 --seed 0 --variant direct
@endcode
"""
import argparse
import copy
import importlib.util
import json
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sim", HERE.parent / "rq2-oracle-steady" / "sim.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

from des_twin import TwinConfig            # noqa: E402  (sim puts env/ on the path)
from des_twin import engine as E           # noqa: E402
from des_twin.rules import f32             # noqa: E402

BASE = "MDD-TECT"
_rank, _select = E.rank_jobs, E.select_machine
FORCE = {}


def _forced_rank(job_rule, ids, s, proc_time):
    jid = FORCE.pop("job", None)
    if jid is None:
        return _rank(job_rule, ids, s, proc_time)
    assert jid in ids, (jid, ids)
    return jid


def _forced_select(machine_rule, cands, *rest):
    mid = FORCE.pop("machine", None)
    if mid is None:
        return _select(machine_rule, cands, *rest)
    assert mid in cands, (mid, cands)
    return mid


E.rank_jobs, E.select_machine = _forced_rank, _forced_select


def make_twin(ep):
    return E.Twin(sim.sig.floor(), ep.jobs, TwinConfig(
        rule=ep.warm_rule, transport="kinematic", warmup_seconds=ep.warmup,
        episode_duration_seconds=ep.cfg["window"], max_sim_seconds=ep.warmup + ep.cfg["window"] + 200000.0,
        agv_schedule=ep.sched))


def window_tardiness(tw, t0):
    t1 = tw.now
    return sum(sim.fleet.due.overlap(j.due, j.exit_time if j.exit_time is not None else t1, t0, t1)
               for j in tw.jobs.values() if j.due is not None) / 1000.0


def play(ep, forces, stop_at=None):
    """Plays the base pair with forces {decision index: (job, machine)}. With stop_at, returns (tw, dec, t0) at that
    decision instead (state before the choice). Otherwise returns (window tardiness, decisions, t0)."""
    tw = make_twin(ep)
    gen = tw.agent_decisions()
    pair = tuple(BASE.split("-"))
    idx, t0 = 0, None
    try:
        dec = gen.send(None)
        while True:
            if t0 is None:
                t0 = tw.now
            if idx == stop_at:
                return tw, dec, t0
            f = forces.get(idx)
            if f is not None:
                FORCE["job"] = f[0]
                if dec.kind == "routing":
                    FORCE["machine"] = f[1]
            dec = gen.send(pair)
            FORCE.clear()
            idx += 1
    except StopIteration:
        pass
    if stop_at is not None:
        return None
    return window_tardiness(tw, ep.warmup if t0 is None else t0), idx, t0


def splice(ep, future_seed, t):
    """ep with the arrivals after t replaced by those of instance future_seed (ids offset by 100000)."""
    from des_twin.scenario import resolve_jobs
    other = sim.scenario(ep.setting, future_seed)
    sc = copy.deepcopy(ep.sc)
    sc["jobs"] = ([j for j in ep.sc["jobs"] if j["arrivalTime"] <= t]
                  + [dict(copy.deepcopy(j), id=int(j["id"]) + 100000) for j in other["jobs"] if j["arrivalTime"] > t + 1.0])
    f = copy.copy(ep)
    f.sc, f.jobs = sc, resolve_jobs(sc, sim.sig.floor())
    return f


def rule_outcome(tw, dec, pair):
    """The (job, machine) the pair would pick at this decision, computed as Twin._route / _dispatch_decision do."""
    jr, mr = pair.split("-")
    if dec.kind == "dispatch":
        return _rank(jr, list(dec.queue), tw, lambda i: tw.jobs[i].proc(dec.machine)), dec.machine
    ids = list(dec.job_candidates)
    jid = _rank(jr, ids, tw, lambda i: tw.jobs[i].min_proc())
    job = tw.jobs[jid]
    cands = tw.candidate_machines(job)
    times = [job.proc(m) for m in cands]
    loads = [tw.machine_load(m) for m in cands]
    utils = [tw.utilization(tw.mach[m]) for m in cands]
    if tw.instant:
        travel = [f32(0)] * len(cands)
    else:
        speed = tw.floor.agv["speed"]
        travel = [f32(tw.floor.estimate_path_length(job.location, m) / speed) for m in cands]
    return jid, _select(mr, cands, times, loads, utils, travel)


def all_options(tw, dec):
    if dec.kind == "dispatch":
        return [(j, dec.machine) for j in dec.queue]
    return [(j, m) for j in dec.job_candidates for m in tw.candidate_machines(tw.jobs[j])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="B2")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", choices=("rule", "direct"), required=True)
    ap.add_argument("--rollout-seconds", type=float, default=1800.0)
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--futures", type=int, default=8, help="0 = hindsight (score on the real future)")
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args(argv)
    mode = "hindsight" if a.futures == 0 else f"expected{a.futures}"
    out = Path(a.out) / a.setting / f"{a.variant}-{mode}" / f"s{a.seed}.json"
    if out.exists():
        print("exists", out)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    ep = sim.Episode(a.setting, a.seed)
    base, n_dec, _ = play(ep, {})
    ref = ep.play([BASE])[0]
    assert abs(base - ref) < 1e-9, (base, ref)
    rng = random.Random(a.seed)

    forces, chosen_tard, rows, idx = {}, base, [], 0
    while True:
        st = play(ep, forces, stop_at=idx)
        if st is None:
            break
        tw, dec, t0 = st
        if tw.now - t0 >= a.rollout_seconds:
            break
        base_choice = rule_outcome(tw, dec, BASE)
        rule_opts = list(dict.fromkeys(rule_outcome(tw, dec, p) for p in sim.PAIRS))
        if a.variant == "rule":
            opts = rule_opts
        else:
            full = all_options(tw, dec)
            rest = [o for o in full if o not in rule_opts]
            if len(rule_opts) + len(rest) > a.cap:
                rest = rng.sample(rest, max(0, a.cap - len(rule_opts)))
            opts = rule_opts + rest
        if idx == 0:                                    # replay check: forcing the base's choice reproduces it
            check = play(ep, {0: base_choice})[0]
            assert abs(check - base) < 1e-9, (check, base)
        best, best_t = base_choice, None
        if len(opts) > 1:
            if a.futures:
                futs = [splice(ep, 50000 + 100 * a.seed + k, tw.now) for k in range(a.futures)]
                st_f = play(futs[0], forces, stop_at=idx)
                assert st_f is not None and abs(st_f[0].now - tw.now) < 1e-9, "splice changed the past"
            else:
                futs = [ep]
            for o in opts:
                t = sum(play(f, {**forces, idx: o})[0] for f in futs) / len(futs)
                if best_t is None or t < best_t - 1e-12 or (abs(t - best_t) <= 1e-12 and o == base_choice):
                    best, best_t = o, t
            chosen_tard = best_t
        forces[idx] = best
        rows.append({"idx": idx, "t": tw.now - t0, "kind": dec.kind, "n_opts": len(opts),
                     "n_rule_opts": len(rule_opts),
                     "n_full": len(all_options(tw, dec)) if a.variant == "direct" else None,
                     "deviates": best != base_choice, "in_rule_set": best in rule_opts,
                     "tard": chosen_tard})
        idx += 1

    final = play(ep, forces)[0]
    res = {"setting": a.setting, "seed": a.seed, "variant": a.variant, "futures": a.futures, "base_pair": BASE, "base": base,
           "rollout": final, "gain_pct": 100.0 * (base - final) / base if base else 0.0,
           "rollout_seconds": a.rollout_seconds, "cap": a.cap, "decisions_in_window": n_dec,
           "decisions_rolled": len(rows), "deviations": sum(r["deviates"] for r in rows),
           "outside_rule_set": sum(not r["in_rule_set"] for r in rows),
           "capped": sum(1 for r in rows if r["n_full"] and r["n_full"] > a.cap),
           "seconds": time.time() - t_start, "decisions": rows}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.rename(out)
    print(f"{a.setting} s{a.seed} {a.variant}-{mode}: base {base:.3f} rollout {final:.3f} ({res['gain_pct']:+.2f}%) "
          f"{len(rows)} decisions, {res['deviations']} deviate, {res['outside_rule_set']} outside rules, "
          f"{res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
