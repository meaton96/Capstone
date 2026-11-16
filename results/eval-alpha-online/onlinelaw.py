"""
@file onlinelaw.py
@brief eval-alpha-online: the laws of ../rq3-alpha-online/law_online.json as alpha(on-duty AGVs, full fleet), with no
       imports, so the policy harness (task.py, v7 code on sys.path) and its checks use one definition. It mirrors
       ../rq3-alpha-online/run.py law_alpha_fn (the simulator's law runs, frozen v5 code); task.py's check that the
       harness's MDD-PTECT@x1 equals the simulator's online-law run tests that the two agree.
"""


def hinge(x, x0, x1):
    return min(1.0, max(0.0, (x1 - x) / (x1 - x0)))


def spread_factor(sigma):
    return max(0.0, 3.0 - 6.0 * sigma)


def law_alpha_fn(law, cell_stats, sigma):
    """alpha(k, full) of one law for a floor: min(4, max(0, 3 - 6 sigma) x h(x_k)), x_k = tau / (k / M), h the law's
    hinge for the fleet state ('full' / 'short' knots; a static law reads the full fleet in every block; a map law
    plays short_alpha in short blocks). full None = no fleet schedule (the whole fleet). Rounded to 3 decimals."""
    spread = spread_factor(sigma)
    tau, m = cell_stats["tau"], cell_stats["machines"]

    def f(k, full):
        if full is None:                    # no fleet schedule: the whole fleet all episode (the full-fleet x)
            return round(min(4.0, spread * hinge(cell_stats["x_full"], *law["knots"]["full"])), 3)
        regime = "full" if k >= full else "short"
        knots = law["knots"][regime] if law["form"] != "static" else law["knots"]["full"]
        kk = full if law["form"] == "static" else k
        if law.get("short_alpha") is not None and regime == "short":
            return float(law["short_alpha"])
        return round(min(4.0, spread * hinge(tau / (kk / m), *knots)), 3)
    return f


def full_alpha(law, cell_stats, sigma):
    """The law's alpha with the whole fleet on duty (episode labels)."""
    return round(min(4.0, spread_factor(sigma) * hinge(cell_stats["x_full"], *law["knots"]["full"])), 3)


def on_duty(tw):
    """AGVs that may take a new task now: the engine's _on_duty rule (the last agv_schedule entry at or before now)."""
    n = len(tw.agvs)
    for t, k in tw.cfg.agv_schedule:
        if t > tw.now:
            break
        n = k
    return n


def full_fleet(tw):
    sched = tw.cfg.agv_schedule
    return max(int(k) for _, k in sched) if sched else None
