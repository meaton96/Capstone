"""
@file common.py
@brief rq2-oracle-bc-future (2026-10-15): shared setup. Loads the frozen v5 code exactly as rq2-oracle-bc-v5 does
       (../rq2-rebase-v5/run.py: code snapshot 6d13412a first on sys.path, its frozen sim.py, asserts nothing comes
       from the live env/), then overlays this experiment's two new files (code/, STAMP) as des_twin.future_table and
       models.future_net. The frozen snapshot itself is not edited.

Also: the published-schedule adapter for forecast errors (levels of ../rq2-perturbed-v5, the PublishedFuture class of
../eval-v5-lookahead-fcast/run_task.py, both imported, not copied), and small helpers.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
REPO = RESULTS.parent
CODE = HERE / "code"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rebase = _load("rebase", RESULTS / "rq2-rebase-v5" / "run.py")   # frozen code first on sys.path; asserts no live env/
sim = rebase.sim
# sim.py put the repo's env/ first on sys.path after the frozen code was imported; put the frozen code first again
# before importing what rebase did not (models, as rq2-oracle-bc-v5/bc_train.py does)
FROZEN = RESULTS / "rq2-rebase-v5" / "code" / "env"
sys.path.insert(0, str(FROZEN))
import des_twin                                                    # noqa: E402  (the frozen package)
import models.network                                              # noqa: E402,F401
import train                                                       # noqa: E402,F401  (save_checkpoint)
for _m in (des_twin, models.network, train):
    assert str(FROZEN) in _m.__file__, _m.__file__
future_table = _load("des_twin.future_table", CODE / "des_twin" / "future_table.py")
future_net = _load("models.future_net", CODE / "models" / "future_net.py")

import numpy as np                                                 # noqa: E402,F401

V5 = list(rebase.V5)
CAPS = rebase.CAPS                                                 # (15, 256), the labels' row caps
STORE_ROWS = 128                                                   # rows stored per slot; an arm uses the first R
HORIZON = 7200.0
ORACLE_DIR = RESULTS / "rq2-rebase-v5" / "out" / "oracle"          # labels of the evaluation seeds
FIT_SEEDS = list(range(2000, 2160))
TEST_SEEDS = list(range(40))
EVAL_SEEDS = TEST_SEEDS + FIT_SEEDS
TRAIN_SEEDS = list(range(3000, 4000))                              # new oracle labels (collect.py label)
LEVELS = ("exact", "jit1", "jit10", "jit30", "churn10", "churn25", "redrawn")


def check_frozen():
    live = str(REPO / "env") + "/"
    bad = [n for n, m in list(sys.modules.items()) if getattr(m, "__file__", None) and str(m.__file__).startswith(live)]
    assert not bad, f"modules imported from the live env/: {bad}"


def stamp():
    out = rebase.stamp()
    h = hashlib.md5()
    for f in sorted(CODE.rglob("*.py")):
        h.update(f.read_bytes())
    out["future_code_md5"] = h.hexdigest()
    out["future_head"] = [ln.split()[1] for ln in (CODE / "STAMP").read_text().splitlines()
                          if ln.startswith("git_head")][0]
    return out


_pert = _fcast = None


def forecast_modules():
    """rq2-perturbed-v5/run.py (Forecast, the planner's draws) and eval-v5-lookahead-fcast/run_task.py
    (PublishedFuture), loaded once."""
    global _pert, _fcast
    if _pert is None:
        _pert = _load("pert", RESULTS / "rq2-perturbed-v5" / "run.py")
        _fcast = _load("fcast", RESULTS / "eval-v5-lookahead-fcast" / "run_task.py")
        assert _fcast.MAX_HORIZON >= HORIZON
    return _pert, _fcast


class Published:
    """Future tables of one episode at forecast level @p level. exact reads the twin's pending arrivals; any other
    level reads the published schedule of rq2-perturbed-v5's Forecast (the planner's draws at the same seed and
    level) through eval-v5-lookahead-fcast's PublishedFuture, as twin.unc.plan_future_jobs (the uncertainty layer's
    path). @p via_adapter also sends exact through the adapter (a check: it must equal the direct path)."""

    def __init__(self, seed, level, via_adapter=False):
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}")
        self.seed, self.level = int(seed), level
        self.direct = level == "exact" and not via_adapter
        self._pub, self._twin_id = None, None

    def __call__(self, tw, rows):
        if self.direct:
            return future_table.future_job_table(tw, rows, HORIZON)
        if self._twin_id != id(tw):
            pert, fcast = forecast_modules()
            from des_twin.engine import Twin
            from des_twin.scenario import resolve_jobs
            self._pub = fcast.PublishedFuture(tw, self.seed, self.level, pert, Twin._make_job, resolve_jobs)
            self._twin_id = id(tw)
        stand_in = SimpleNamespace(now=tw.now, floor=tw.floor, unc=self._pub, order=tw.order)
        return future_table.future_job_table(stand_in, rows, HORIZON)


def write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def mdd_ptect(seed):
    """MDD-PTECT window tardiness (1000 job-s) of a seed, from rq2-rebase-v5's policies (the same harness)."""
    return json.loads((RESULTS / "rq2-rebase-v5" / "out" / "policies" / f"s{seed}.json").read_text())["tard"][
        "fixed:MDD-PTECT"]
