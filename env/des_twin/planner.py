"""
@file planner.py
@brief Twin planner v2 (2026-10-11, dev-twin-uncertainty; plan §2.6): rolling-horizon rule-pair selection by forking
       the running twin (des_twin.branch) instead of replaying the episode from t = 0, with a belief transform so that
       the simulated futures use only the controllers' information set under the uncertainty layer.

@details
**v1** (results/rq2-oracle-perturbed/run.py, rq2-head-to-head): at each slot k (re-plan time t_k = warm-up + k x slot)
build a forecast instance (real jobs arrived by t_k + the published future), replay the whole episode on it with the
chosen prefix and each candidate pair as the tail, keep the best. Correct only because the twin is deterministic: the
replay reproduces the true current state.

**v2** pauses the world twin at t_k (Twin.pause_at: the state after every tick before t_k, no extra tick), forks one
child per candidate pair, and in each child
  1. applies the belief transform (`belief_transform`): operations not yet started run their planned durations (x the
     online bias estimate b_hat at I1); operations in progress end when the controllers' estimate says (planned
     countdown x O3 factor, at least one more tick); the future job stream is the published plan (plan_source:
     planned arrivals, cancellations known by their notice, P1 refreshes; rush orders unknown until they arrive);
     machines whose reported state differs from the truth are set to the reported state where the engine can
     represent it (an unreported finish of a job still waiting at the machine: processing with its estimated
     remaining time; an unreported start of a job reported queued there: idle with the job queued); the child's own
     execution and observation are then exact (it is the planner's model);
  2. plays the rest of the current slot with the current pair, then the candidate pair to the end of the window;
  3. returns the window tardiness of its simulated future.
The world twin then goes on with the best pair. Unrepresentable differences (a job in transit or already routed
whose booking is late, AGV states) keep the truth: a small leak, identical for every planner variant.

With the uncertainty layer off (TwinConfig.uncertainty None, or an all-zero configuration) and plan_source = the true
future, v2 equals v1's "exact" level (the greedy hindsight oracle); with v1's Forecast as plan_source it equals v1 on
that forecast level. Both are checked in results/dev-twin-uncertainty/check_planner.py (rq2-oracle-perturbed exact seed 0
= 93.78, rq2-head-to-head jit10-slot300 seeds 0-2).

Variants: belief=False plays the children on the true world (the hindsight oracle under noise, CRN); ensemble=K
averages each pair's tails over K conditional redraws of the noise around the belief (UncertaintyTrace.redraw_after,
mode "realizable"; the noise-aware planner, plan §2.6).
"""
import time

import numpy as np

from . import branch
from .engine import NEEDS_ROUTING, PROCESSING, QUEUED
from .motion import f32_countdown

f32 = np.float32


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def window_tardiness(tw, t0):
    """Tardiness accrued inside the agent window [t0, now] (rq2-oracle-steady/sim.py's measure), in 1000 job-s."""
    t1 = tw.now
    tard = sum(overlap(j.due, j.exit_time if j.exit_time is not None else t1, t0, t1)
               for j in tw.jobs.values() if j.due is not None)
    return tard / 1000.0


def plan_view_source(tw):
    """plan_source for a world with the uncertainty layer: the trace's published schedule at t."""
    def src(t, twin):
        return twin.unc.plan_view(t, {j.id for j in twin.order})
    return src if tw.unc is not None else None


# ── Belief transform ─────────────────────────────────────────────────────────────────────────────────────────

def _countdown_tick(start_tick, dur, dt, until):
    return start_tick + max(1, f32_countdown(dur, dt, until=until)[0])


def belief_transform(tw, future=None, i1=None, keep_trace=None):
    """@brief Turns a (forked) world twin into the planner's model of it; see the module docstring.

    @param future    the published future: des_jobs/1 entries replacing the jobs not yet arrived (None keeps them)
    @param i1        use the online bias estimates (default: the trace's information level)
    @param keep_trace  an UncertaintyTrace (a conditional redraw) that the child keeps running on instead of becoming
                     exact (the noise-aware planner); its future job stream replaces `future`
    Returns a dict of what was changed (counts)."""
    stats = {"remaining_reset": 0, "unreported_finish": 0, "unreported_start": 0}
    unc = tw.unc
    if unc is not None:
        core = tw._vcore
        view = tw.view
        core.sync()
        view.clear_local()
        use_i1 = unc.cfg.adaptive if i1 is None else bool(i1)
        bh = {m.id: (core.bhat(m.id) if use_i1 else 1.0) for m in tw.machines}
        rep_proc = dict(core.proc)
        rep = core.rep
        tw.unc = None                     # from here on the child is the planner's exact model
        tw.cfg.uncertainty = None
        k, dt, lead = tw.k, tw.dt, tw.lead
        for m in tw.machines:
            if m.down:
                continue
            rp = rep_proc.get(m.id)
            if m.idle and rp is not None:
                # unreported finish: representable while the job still waits at the machine for routing
                job = tw.jobs[rp[0]]
                if (job.state == NEEDS_ROUTING and job.location == m.id and job.cur_op == rp[1] + 1
                        and job.pre_agv < 0 and job.assigned_agv < 0 and job.op_rows):
                    row = job.op_rows[-1]
                    m.busy -= (row["end"] - row["start"]) if row.get("end") is not None else 0.0
                    row["end"] = None
                    job.cur_op -= 1
                    job.completed_ops -= 1
                    tw._to(job, PROCESSING)
                    plan = job.ops[rp[1]][m.id]
                    est = core.estimate(rp[0], rp[1], m.id, plan, use_i1)
                    m.idle, m.job, m.start_tick, m.dur, m.proc_start = False, job.id, rp[2], est, rp[3]
                    m.finish_tick = max(_countdown_tick(rp[2], est, dt, 0.0), k + 1)
                    m.almost_tick = None
                    tw._dispatchable.discard(tw._mpos[m.id])
                    tw._wake(m.finish_tick)
                    stats["unreported_finish"] += 1
                continue
            if not m.idle and rp is None and m.job >= 0:
                # unreported start: representable when the job is reported queued at the machine
                job = tw.jobs[m.job]
                r = rep.get(job.id)
                if r is not None and r.state == QUEUED and r.loc == m.id and job.state == PROCESSING \
                        and job.pre_agv < 0:
                    job.op_rows[-1]["start"] = None
                    m.idle, m.job, m.proc_start, m.dur = True, -1, None, f32(0)
                    m.finish_tick = m.almost_tick = None
                    tw._place_on_belt(m, job.id)
                    tw._to(job, QUEUED)
                    tw._wake(k + 1)
                    stats["unreported_start"] += 1
                continue
            if not m.idle and m.job >= 0:
                # in progress: ends when the controllers' estimate says (from the reported start)
                job = tw.jobs[m.job]
                op = job.cur_op
                plan = job.ops[op][m.id]
                est = core.estimate(job.id, op, m.id, plan, use_i1)
                start = rp[2] if rp is not None and rp[0] == job.id else m.start_tick
                if est != m.dur or start != m.start_tick:
                    m.dur, m.start_tick = est, start
                    m.finish_tick = max(_countdown_tick(start, est, dt, 0.0), k + 1)
                    tw._wake(m.finish_tick)
                    if m.almost_tick is not None:
                        m.almost_tick = max(_countdown_tick(start, est, dt, lead), k + 1)
                        tw._wake(m.almost_tick)
                    stats["remaining_reset"] += 1
        if keep_trace is not None:
            from .uncertainty import apply_trace
            tw.unc = keep_trace
            tw.cfg.uncertainty = keep_trace
            apply_trace(tw, keep_trace)
            return stats
    if future is not None:
        tw.replace_pending(future)
    if unc is not None and use_i1:
        # future operations run at p_hat = p_plan x b_hat (the model's truth); due dates stay
        for job in list(tw._live_d.values()) + tw.pending[tw.pending_i:]:
            first = job.cur_op + 1 if job.state == PROCESSING else job.cur_op
            for o in range(first, len(job.ops)):
                job.ops[o] = {mid: (p if bh.get(mid, 1.0) == 1.0 else f32(float(p) * bh[mid]))
                              for mid, p in job.ops[o].items()}
    return stats


# ── Planner loop ─────────────────────────────────────────────────────────────────────────────────────────────

def _play_tail(tw, gen, prefix, pair, t0, slot, warmup):
    """In a child paused at a re-plan: the rest of the current slot with the prefix, then `pair` to the end."""
    y = gen.send(None)
    try:
        while True:
            if y.kind == "pause":
                y = gen.send(None)
                continue
            if t0 is None:
                t0 = tw.now
            kk = int((tw.now - t0) // slot)
            p = prefix[kk] if kk < len(prefix) else pair
            y = gen.send(tuple(p.split("-")))
    except StopIteration:
        pass
    return window_tardiness(tw, warmup if t0 is None else t0)


def plan_episode(tw, pairs, slot, warmup, window, plan_source="auto", max_parallel=1, belief=True, ensemble=0,
                 ensemble_seed=0, rush_pool=None, timeout=None, i1=None):
    """@brief Runs the world twin `tw` (fresh, not started) to the end with planner v2 choosing the pair per slot.

    @param pairs        candidate rule pairs ("JOB-MACHINE")
    @param slot         re-plan and switch interval (s); re-plans at warmup + k x slot, switches at t0 + k x slot (t0 =
                        the first agent decision), as v1
    @param plan_source  "auto": the trace's plan_view when the world has an uncertainty trace, else the true future;
                        None: the true future; or a callable (t, tw) -> des_jobs/1 entries of the future
    @param belief       False: children play on the true world (hindsight oracle under the same noise, CRN)
    @param ensemble     K > 0: each pair's score is the mean over K conditional redraws (noise-aware planner);
                        redraw j at re-plan k uses seed 50000 + 100 x ensemble_seed + 1000 x k + j
    @return dict: schedule, planned (score of the chosen tail per re-plan), real (window tardiness of the world),
            plan_seconds (wall time per re-plan), t0, decisions"""
    if plan_source == "auto":
        plan_source = plan_view_source(tw)
    n = int(round(window / slot))
    gen = tw.agent_decisions()
    schedule, planned, plan_s, scores_all = [], [], [], []
    t0 = None
    k_next = 0
    tw.pause_at(warmup)
    try:
        y = gen.send(None)
        while True:
            if y.kind == "pause":
                t_plan = warmup + k_next * slot
                fut = plan_source(t_plan, tw) if (plan_source is not None and belief) else None
                prefix = list(schedule)

                def child(opt, prefix=prefix, fut=fut, k=k_next):
                    pair, j = opt
                    if belief:
                        keep = None
                        if ensemble:
                            keep = tw.unc.redraw_after(tw, 50000 + 100 * ensemble_seed + 1000 * k + j,
                                                       mode="realizable", rush_pool=rush_pool,
                                                       bias_est=tw._vcore.bias_estimate) if tw.unc is not None else None
                        belief_transform(tw, fut, i1=i1, keep_trace=keep)
                    return _play_tail(tw, gen, prefix, pair, t0, slot, warmup)
                opts = [(p, j) for p in pairs for j in range(max(1, ensemble))]
                t_start = time.time()
                res = branch.fork_map(child, opts, max_parallel=max_parallel, timeout=timeout)
                plan_s.append(time.time() - t_start)
                K = max(1, ensemble)
                tails = [float(np.mean(res[i * K:(i + 1) * K])) for i in range(len(pairs))]
                i = int(np.argmin(tails))
                schedule.append(pairs[i])
                planned.append(tails[i])
                scores_all.append(tails)
                k_next += 1
                if k_next < n:
                    tw.pause_at(warmup + k_next * slot)
                y = gen.send(None)
            else:
                if t0 is None:
                    t0 = tw.now
                kk = int((tw.now - t0) // slot)
                p = schedule[min(kk, len(schedule) - 1)]
                y = gen.send(tuple(p.split("-")))
    except StopIteration:
        pass
    return {"schedule": schedule, "planned": planned, "tails": scores_all,
            "real": window_tardiness(tw, warmup if t0 is None else t0), "plan_seconds": plan_s, "t0": t0,
            "decisions": tw.decisions, "replans": len(schedule)}


def fixed_rule(tw, pair, warmup):
    """Plays `pair` at every agent decision of a fresh twin; returns its window tardiness."""
    gen = tw.agent_decisions()
    t0 = None
    halves = tuple(pair.split("-"))
    try:
        y = gen.send(None)
        while True:
            if t0 is None:
                t0 = tw.now
            y = gen.send(halves)
    except StopIteration:
        pass
    return window_tardiness(tw, warmup if t0 is None else t0)


__all__ = ["belief_transform", "fixed_rule", "plan_episode", "plan_view_source", "window_tardiness"]
