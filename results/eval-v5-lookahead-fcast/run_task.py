"""
@file run_task.py
@brief eval-v5-lookahead-fcast (2026-10-15, user): one train-v5-lookahead checkpoint in the twin with its look-ahead
       features computed from a PERTURBED published schedule instead of the exact one. B2 fitting seeds 2000-2159
       (--split fit) or test seeds 0-39 (--split test), one forecast error level per task.
@details Everything else is eval-v5-lookahead's run_task.py (frozen code rq2-rebase-v5/code/env, floor
         dev-popart-smoke/floor7, B2 params, random warm-up, 21,600 s window, 900 s slots, 5,400 s horizon,
         deterministic). Only the 15 look-ahead features change; the world the policy acts in is the real instance.

         Forecast model = rq2-perturbed-v5's `Forecast` (imported, not copied), with the same RNG keys, so at a given
         (seed, level) the policy sees exactly the published schedule the twin planner planned on:
           - jit1 / jit10 / jit30: arrival + N(0, sigma), sigma 1 / 10 / 30 min, due date moves with it, drawn once per
             job; a published arrival already in the past is shown at now + 1 s;
           - churn10 / churn25: 10 / 25% of the real jobs missing from the plan and as many phantom jobs in it (from
             B2 instance 70000 + seed), plus jit10;
           - redrawn: the whole future replaced by instance 70000 + seed (no information; control);
           - exact: the real schedule through the same adapter (check: must reproduce eval-v5-lookahead).
         A real job leaves the published schedule when it arrives in the twin (the ids of twin.order); a phantom when
         its published arrival has passed. That is Forecast.episode's view with the twin's own arrival bookkeeping.

         The adapter hands des_twin.lookahead.lookahead_features a stand-in twin whose `unc.plan_future_jobs` returns
         the published future (the path the frozen code already uses for the uncertainty layer's plan). The frozen
         code is not edited. A finished task (DONE) is skipped.
@par Usage
@code{.sh}
python results/eval-v5-lookahead-fcast/run_task.py --run train-v5-lookahead_s3 --level jit10 --split test
python results/eval-v5-lookahead-fcast/run_task.py --run train-v5-lookahead_s3 --level exact --seeds 0-1 --tag smoke
@endcode
"""
import argparse
import contextlib
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CODE = REPO / "results" / "rq2-rebase-v5" / "code" / "env"
B2 = {"failure_probability": 0.0, "due_date_allowance_range": [1.75, 2.5], "regime_block_seconds": 5400,
      "horizon_seconds": 30600,
      "load_mix": [{"name": "normal", "utilization": [0.6, 1.2], "weight": 2},
                   {"name": "surge", "utilization": [1.0, 1.6]},
                   {"name": "short", "utilization": [0.6, 1.2], "agv_count": 2}]}
LEVELS = ("exact", "jit1", "jit10", "jit30", "churn10", "churn25", "redrawn")
MAX_HORIZON = 7200.0           # des_twin.lookahead.HORIZONS[-1]; later jobs never enter a feature


class PublishedFuture:
    """The published schedule of one twin episode at forecast `level` (rq2-perturbed-v5 Forecast draws), as the light
    job objects des_twin.lookahead reads (arrival, ops, types, due)."""

    def __init__(self, twin, seed, level, pert, make_job, resolve_jobs):
        sc = pert.sim.scenario("B2", seed)
        fc = pert.Forecast(SimpleNamespace(sc=sc, setting="B2"), level, seed)
        self.redrawn = bool(fc.cfg.get("redrawn"))
        # real jobs: the twin's own resolved jobs, matched by id (assert the instance is the planner's)
        self.ids, self.arr, self.shift, self.keep, self.real = [], [], [], [], []
        for i, j in enumerate(fc.source):
            jid = int(j["id"])
            job = twin.jobs.get(jid)
            if job is None or abs(float(job.arrival) - float(np.float32(j["arrivalTime"]))) > 1e-3:
                raise SystemExit(f"seed {seed}: twin instance differs from sim.scenario('B2', {seed}) at job {jid}")
            self.ids.append(jid)
            self.real.append(job)
            self.arr.append(float(job.arrival))
            self.shift.append(float(fc.shift[i]))
            self.keep.append(not bool(fc.missing[i]))
        if len(fc.source) != len(twin.jobs):
            raise SystemExit(f"seed {seed}: {len(fc.source)} scenario jobs vs {len(twin.jobs)} twin jobs")
        self.pub = np.asarray(self.arr) + np.asarray(self.shift)      # unclipped published arrival
        # phantom / redrawn jobs, resolved on the twin's floor once
        self.other = []
        if fc.other:
            jd = resolve_jobs(dict(sc, jobs=fc.other), twin.floor)["jobs"]
            self.other = sorted((make_job(d) for d in jd), key=lambda o: o.arrival)

    def plan_future_jobs(self, t, arrived):
        end = t + MAX_HORIZON
        out = []
        if not self.redrawn:
            for k in np.nonzero(self.pub < end)[0]:
                if not self.keep[k] or self.ids[k] in arrived:
                    continue
                job = self.real[k]
                if self.shift[k]:
                    a = max(self.arr[k] + self.shift[k], t + 1.0)
                    due = None if job.due is None else job.due + (a - self.arr[k])
                    out.append(SimpleNamespace(arrival=a, ops=job.ops, types=job.types, due=due))
                else:
                    out.append(job)
        for o in self.other:
            if o.arrival >= end:
                break
            if o.arrival > t + 1.0:
                out.append(o)
        out.sort(key=lambda j: j.arrival)
        return out


def install(level, pert):
    """Patch the frozen twin env so every new episode carries its PublishedFuture and the look-ahead reads it."""
    import des_twin.lookahead as la
    import env_wrappers.twin_env as te
    from des_twin.engine import Twin
    from des_twin.scenario import resolve_jobs
    new_twin, features = te.TwinSchedulingEnv._new_twin, la.lookahead_features

    def _new_twin(self):
        tw = new_twin(self)
        tw._published = PublishedFuture(tw, int(self._seed), level, pert, Twin._make_job, resolve_jobs)
        return tw

    def _features(twin):
        pub = getattr(twin, "_published", None)
        if pub is None or getattr(twin, "unc", None) is not None:
            raise RuntimeError("eval-v5-lookahead-fcast: look-ahead called on a twin without a published schedule")
        stand_in = SimpleNamespace(now=twin.now, floor=twin.floor, unc=pub, order=twin.order, cfg=twin.cfg,
                                   agvs=twin.agvs)
        return features(stand_in)

    te.TwinSchedulingEnv._new_twin = _new_twin
    la.lookahead_features = _features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="checkpoint")
    ap.add_argument("--level", required=True, choices=LEVELS)
    ap.add_argument("--seeds", default="2000-2159")
    ap.add_argument("--split", default="fit", choices=["fit", "test"], help="test: B2 test seeds 0-39")
    ap.add_argument("--tag", default="", help="output folder suffix (smoke tests)")
    a = ap.parse_args()
    if a.split == "test":
        a.seeds = "0-39"
    out = HERE / a.level / (f"{a.run}__{a.ckpt}" + ("__test" if a.split == "test" else "") + a.tag)
    if (out / "DONE").exists():
        print(f"{a.level}/{out.name}: done, skipped")
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ck = f"results/{a.run}/{a.ckpt}.pt"
    ev = ["--twin", "results/dev-popart-smoke/floor7/des_floor.json",
          "--reward-spec", str(CODE / "config" / "rewards" / "tardiness.json"), "--scenario-generator", "randomized",
          "--random-warmup", "--episode-duration-seconds", "21600", "--params", json.dumps(B2), "--seeds", a.seeds,
          "--pdr", "none", "--slot-seconds", "900", "--discount-horizon-s", "5400", "--device", "cpu",
          "--checkpoint", ck, "--out", str(out)]
    # rq2-perturbed-v5/run.py puts the same frozen code first on sys.path and asserts nothing came from the live env/
    spec = importlib.util.spec_from_file_location("pert", REPO / "results" / "rq2-perturbed-v5" / "run.py")
    pert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pert)
    sys.path.insert(0, str(CODE))
    os.chdir(REPO)
    import evaluate
    if not str(Path(evaluate.__file__).resolve()).startswith(str(CODE)):
        raise SystemExit(f"evaluate imported from {evaluate.__file__}, not the frozen code")
    install(a.level, pert)
    t0 = time.time()
    with open(out / "eval.out", "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        evaluate.main(ev)
    rows = list(csv.DictReader(open(out / "episodes.csv")))
    want = len(evaluate.parse_seeds(a.seeds))
    if len(rows) != want:
        raise SystemExit(f"{out.name}: {len(rows)} episodes, expected {want}")
    (out / "DONE").write_text(f"{time.time() - t0:.0f}\n")
    print(f"{a.level}/{out.name}: {len(rows)} episodes in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
