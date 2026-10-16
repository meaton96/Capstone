"""Future job table (des_twin.future_table) and the future-encoder network (models.future_net), rq2-oracle-bc-future
(2026-10-15)."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from des_twin.future_table import FUTURE_JOB_FEATURES, future_job_table, future_jobs
from models.future_net import FutureSchedulingNetwork, network_for, reads_future_table
from models.network import SchedulingNetwork
from tests.test_architecture import make_dummy_obs


def _job(arrival, ops, types, due=None):
    return SimpleNamespace(arrival=arrival, ops=ops, types=types, due=due)


def _twin(now, pending, unc=None, machines=15):
    return SimpleNamespace(now=now, pending=pending, pending_i=0, unc=unc, order=[],
                           floor=SimpleNamespace(machines=[None] * machines))


def test_rows_follow_the_pending_arrivals():
    jobs = [_job(50.0, [{0: 100.0}], ["Mill"]),                       # already arrived: skipped
            _job(160.0, [{0: 100.0, 1: 300.0}, {2: 60.0}], ["Mill", "Weld"], due=2000.0),
            _job(400.0, [{3: 50.0}], ["Lathe"]),
            _job(9000.0, [{3: 50.0}], ["Lathe"])]                    # beyond the horizon
    tw = _twin(100.0, jobs)
    assert [j.arrival for j in future_jobs(tw, 7200.0)] == [160.0, 400.0]
    t = future_job_table(tw, rows=4, horizon=7200.0)
    assert t.shape == (4, FUTURE_JOB_FEATURES) and t.dtype == np.float32
    assert t[:, 0].tolist() == [1.0, 1.0, 0.0, 0.0] and not t[2:].any()
    r = t[0]
    assert r[1] == pytest.approx(60.0 / 7200.0) and r[3] == 0.0 and t[1, 3] == pytest.approx(1.0 / 33.0)
    work, min_work = 200.0 + 60.0, 100.0 + 60.0
    assert r[6] == pytest.approx(work / (work + 1200.0)) and r[7] == pytest.approx(min_work / (min_work + 1200.0))
    assert r[10] == pytest.approx(200.0 / 800.0) and r[12] == pytest.approx(60.0 / 660.0)   # Mill, Weld
    assert r[11] == 0.0 and r[13] == 0.0 and r[14] == 0.0
    assert r[15] == 1.0 and t[1, 15] == 0.0 and t[1, 16] == 0.0
    slack = 2000.0 - 160.0 - min_work
    assert r[16] == pytest.approx(slack / (slack + 1200.0)) and r[17] == pytest.approx(1900.0 / 3100.0)
    assert t[1, 4] > r[4] > 0.0                                       # cumulative work grows


def test_the_published_plan_replaces_the_pending_arrivals():
    real = [_job(500.0, [{0: 100.0}], ["Mill"])]
    plan = [_job(300.0, [{1: 80.0}], ["Lathe"]), _job(600.0, [{0: 100.0}], ["Mill"])]
    calls = []

    class Plan:
        def plan_future_jobs(self, t, arrived):
            calls.append((t, arrived))
            return plan

    t = future_job_table(_twin(100.0, real, unc=Plan()), rows=8)
    assert calls == [(100.0, set())] and t[:, 0].sum() == 2.0
    assert t[0, 1] == pytest.approx(200.0 / 7200.0)                   # the plan's first job, not the real one
    assert np.array_equal(t, future_job_table(_twin(100.0, plan), rows=8))


def test_rows_cap_keeps_the_earliest_jobs():
    jobs = [_job(100.0 + 10 * i, [{0: 50.0}], ["Mill"]) for i in range(20)]
    t = future_job_table(_twin(100.0, jobs), rows=5)
    assert t[:, 0].sum() == 5.0 and t[4, 1] == pytest.approx(40.0 / 7200.0)
    assert np.array_equal(t, future_job_table(_twin(100.0, jobs), rows=12)[:5])   # rows do not change the features


def test_future_network_reads_the_table_and_round_trips():
    torch.manual_seed(0)
    net = FutureSchedulingNetwork()
    batch = make_dummy_obs(4)
    table = torch.zeros(4, 16, FUTURE_JOB_FEATURES)
    table[:, :5, 0] = 1.0
    table[:, :5, 1:] = torch.rand(4, 5, FUTURE_JOB_FEATURES - 1)
    batch["future_job_table"] = table
    batch2 = dict(batch, future_job_table=torch.cat([table[:, :, :1], torch.rand(4, 16, FUTURE_JOB_FEATURES - 1)], -1))
    d1 = [d.probs for d in net.distributions(batch)]
    d2 = [d.probs for d in net.distributions(batch2)]
    assert not all(torch.allclose(a, b) for a, b in zip(d1, d2))      # the actor reads the table
    actions, logp, _ = net.act(batch)
    logp2, _, _ = net.evaluate(batch, actions)
    assert torch.allclose(logp, logp2, atol=1e-6)
    with pytest.raises(ValueError):
        net.act(make_dummy_obs(2))
    sd = net.state_dict()
    assert reads_future_table(sd) and not reads_future_table(SchedulingNetwork().state_dict())
    again = network_for(sd)
    again.load_state_dict(sd)
    assert isinstance(again, FutureSchedulingNetwork)
    assert all(torch.equal(a, b.probs) for a, b in zip(d1, again.distributions(batch)))
    assert type(network_for(SchedulingNetwork().state_dict())) is SchedulingNetwork


def test_twin_env_adds_the_future_table(tmp_path):
    from env_wrappers.twin_env import TwinSchedulingEnv
    from tests.test_learning_stack_fixes import _fixture_floor
    from tests.test_twin_env import FIX
    floor = _fixture_floor(tmp_path)
    scenario = dict(json.load(open(os.path.join(FIX, "convoy_waves.json"))), agvCount=3)
    env = TwinSchedulingEnv(floor, scenario=scenario, obs_caps=(15, 64), future_table=(32, 3600.0))
    obs = env.reset()
    t = obs["future_job_table"]
    assert t.shape == (32, FUTURE_JOB_FEATURES) and np.isfinite(t).all()
    assert np.array_equal(t, future_job_table(env._twin, 32, 3600.0))
    assert "future_job_table" not in TwinSchedulingEnv(floor, scenario=scenario, obs_caps=(15, 64)).reset()
