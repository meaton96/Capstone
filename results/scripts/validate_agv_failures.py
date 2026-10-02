#!/usr/bin/env python3
"""
@file validate_agv_failures.py
@brief Checks AGV breakdowns in batch-run output against the configured failure model.

@details Reads agv_failures.csv, agv_performance.csv and results.csv under the given result directories
(searched recursively; one run per directory, as run_experiment_queue.py writes them).

Model (AGVController, "Breakdowns"): time to failure is Weibull(k, lambda) in OPERATING seconds; the first life
of each AGV in an episode is drawn from the equilibrium residual-life distribution, every later life is a
full Weibull life; repairs are LogNormal(mu, sigma).

Checks:
  1. Repairs. Each sampled repair is logged at the breakdown, so the sample is uncensored: KS test against
     LogNormal(mu, sigma).
  2. Lives. Observed failures are biased toward short lives (long lives run past the episode end), so a KS test
     on them alone would be wrong. Instead each life, failed or still running at episode end (censored),
     contributes its cumulative hazard H(age). Under the model the sum of H over all lives is the expected
     number of failures E, and the observed count O is approximately Poisson(E): z = (O - E) / sqrt(E)
     (a one-sample log-rank test). Done separately for residual first lives and full lives.
        full life:      H(t) = (t / lambda)^k
        residual life:  S_e(t) = Q(1/k, (t / lambda)^k)  (regularized upper incomplete gamma), H_e = -ln S_e
  3. Totals: failures per AGV-episode, mean repair, time queued behind broken AGVs.

Only batch runs (run_experiment_queue.py / HeadlessBatchRunner) write these CSVs; evaluate.py runs report
per-episode totals instead (episodes.csv: agv_failures, agv_repair_time, agv_blocked_by_failure_time).

@par Usage
@code{.sh}
.venv/bin/python results/scripts/validate_agv_failures.py linux_server_dev/Results/AGVF_check
.venv/bin/python results/scripts/validate_agv_failures.py --self-test
@endcode
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy import special, stats


def cumulative_hazard(age: np.ndarray, k: float, lam: float, residual: bool) -> np.ndarray:
    """@brief H(age) for a full Weibull life, or for the equilibrium residual life when @p residual."""
    z = (np.asarray(age, dtype=float) / lam) ** k
    if not residual:
        return z
    survival = special.gammaincc(1.0 / k, z)
    return -np.log(np.clip(survival, 1e-300, 1.0))


def observed_vs_expected(event_ages, censored_ages, k, lam, residual):
    """@brief (observed failures, expected failures, z) for one group of lives."""
    events = np.asarray(event_ages, dtype=float)
    censored = np.asarray(censored_ages, dtype=float)
    observed = len(events)
    expected = float(cumulative_hazard(events, k, lam, residual).sum()
                     + cumulative_hazard(censored, k, lam, residual).sum())
    z = (observed - expected) / math.sqrt(expected) if expected > 0 else float("nan")
    return observed, expected, z


def read_rows(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_run(run_dir: Path):
    """@brief Failures and censored lives of one run directory, with its model parameters (None if AGV
    breakdowns were off or the files are from a player built before them)."""
    results = run_dir / "results.csv"
    if not results.exists():
        return None
    rows = read_rows(results)
    if not rows or "agv_failures_enabled" not in rows[0]:
        return None
    on = [r for r in rows if r["agv_failures_enabled"] == "1"]
    if not on:
        return None
    params = {(float(r["agv_weibull_k"]), float(r["agv_weibull_lambda"]),
               float(r["agv_repair_log_mu"]), float(r["agv_repair_log_sigma"])) for r in on}
    if len(params) != 1:
        print(f"skip {run_dir}: {len(params)} different AGV failure settings in one results.csv")
        return None

    failures = read_rows(run_dir / "agv_failures.csv") if (run_dir / "agv_failures.csv").exists() else []
    perf = read_rows(run_dir / "agv_performance.csv")
    return {
        "params": params.pop(),
        "episodes": len(on),
        "agv_episodes": len(perf),
        "fail_age": [(float(r["operating_age"]), r["residual_life"] == "1") for r in failures],
        "repairs": [float(r["repair_duration"]) for r in failures],
        "censored": [(float(r["censored_operating_age"]), r["censored_life_is_residual"] == "1") for r in perf],
        "time_broken": sum(float(r["time_broken"]) for r in perf),
        "time_blocked": sum(float(r["time_blocked_by_failure"]) for r in perf),
    }


def report(runs) -> bool:
    """@brief Prints the checks for each parameter set. @return True when no check is rejected at 1%."""
    ok = True
    by_params = {}
    for run in runs:
        by_params.setdefault(run["params"], []).append(run)

    for (k, lam, mu, sigma), group in by_params.items():
        fail_age = [fa for run in group for fa in run["fail_age"]]
        censored = [c for run in group for c in run["censored"]]
        repairs = [x for run in group for x in run["repairs"]]
        agv_episodes = sum(run["agv_episodes"] for run in group)
        print(f"\nAGV failures k={k:g} lambda={lam:g} repair LogNormal({mu:g}, {sigma:g}): "
              f"{len(group)} runs, {sum(run['episodes'] for run in group)} episodes, {agv_episodes} AGV-episodes")
        print(f"  failures: {len(fail_age)} ({len(fail_age) / max(agv_episodes, 1):.2f} per AGV-episode); "
              f"time broken {sum(r['time_broken'] for r in group):.0f} s; "
              f"time queued behind broken AGVs {sum(r['time_blocked'] for r in group):.0f} s")

        if len(repairs) >= 20:
            ks = stats.kstest(repairs, stats.lognorm(s=sigma, scale=math.exp(mu)).cdf)
            verdict = "ok" if ks.pvalue >= 0.01 else "REJECTED"
            ok &= ks.pvalue >= 0.01
            print(f"  repairs vs LogNormal: n={len(repairs)} mean={np.mean(repairs):.1f} "
                  f"(theory {math.exp(mu + sigma * sigma / 2):.1f})  KS D={ks.statistic:.3f} p={ks.pvalue:.3f}  {verdict}")
        else:
            print(f"  repairs: only {len(repairs)}, too few for a KS test")

        for residual, label in ((True, "first lives (equilibrium residual)"), (False, "later lives (full Weibull)")):
            events = [a for a, r in fail_age if r == residual]
            cens = [a for a, r in censored if r == residual]
            observed, expected, z = observed_vs_expected(events, cens, k, lam, residual)
            if expected < 5:
                print(f"  {label}: O={observed} E={expected:.1f}, too few expected failures to test")
                continue
            verdict = "ok" if abs(z) < 2.576 else "REJECTED"
            ok &= abs(z) < 2.576
            print(f"  {label}: O={observed} E={expected:.1f} z={z:+.2f}  {verdict}")
    return ok


def self_test(seed: int = 1, agv_episodes: int = 4000, k: float = 1.5, lam: float = 8400.0,
              horizon_mean: float = 3200.0) -> bool:
    """@brief Simulates AGV lives from the model with random censoring and checks the test does not reject it,
    and does reject a 20% wrong scale."""
    rng = np.random.default_rng(seed)
    gamma = rng.gamma(1.0 + 1.0 / k, 1.0, size=agv_episodes)
    first = rng.random(agv_episodes) * lam * gamma ** (1.0 / k)        # equilibrium residual life
    horizon = rng.uniform(0.5, 1.5, size=agv_episodes) * horizon_mean  # operating seconds per AGV-episode
    ev_r, ce_r, ev_f, ce_f = [], [], [], []
    for life0, end in zip(first, horizon):
        if life0 >= end:
            ce_r.append(end)
            continue
        ev_r.append(life0)
        t = life0
        while True:
            life = lam * rng.weibull(k)
            if t + life >= end:
                ce_f.append(end - t)
                break
            ev_f.append(life)
            t += life
    good = True
    for scale, should_pass in ((lam, True), (lam * 1.2, False)):
        for residual, ev, ce in ((True, ev_r, ce_r), (False, ev_f, ce_f)):
            o, e, z = observed_vs_expected(ev, ce, k, scale, residual)
            passed = abs(z) < 2.576
            print(f"self-test lambda={scale:g} {'residual' if residual else 'full'}: O={o} E={e:.1f} z={z:+.2f} "
                  f"{'pass' if passed else 'reject'} (expected {'pass' if should_pass else 'reject'})")
            if residual and passed != should_pass:
                good = False
    return good


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("@par")[0], formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dirs", nargs="*", help="Result directories to search for runs")
    parser.add_argument("--self-test", action="store_true", help="Check the test itself on simulated lives")
    args = parser.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1
    if not args.dirs:
        parser.error("give result directories, or --self-test")

    runs = []
    for d in args.dirs:
        for results in sorted(Path(d).rglob("results.csv")):
            run = load_run(results.parent)
            if run is not None:
                runs.append(run)
    if not runs:
        print("No runs with AGV failures on (results.csv needs the agv_failures_enabled column).")
        return 1
    return 0 if report(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
