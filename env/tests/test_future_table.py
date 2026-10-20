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


# ── Vectorized table (10-20) and PPO plumbing: train.py --future-table / --init-from / --critic-warmup-steps ──────────

def _reference_table(twin, rows, horizon):
    """The per-row loop the rq2-oracle-bc-future clones were trained with (results/rq2-oracle-bc-future/code)."""
    from des_twin.future_table import RANK_SCALE, CUM_WORK_SCALE, TYPE_WORK_SCALE, SLACK_RATIO_SCALE
    from des_twin.observation import AGE_SCALE, COUNT_SCALE, OPS_SCALE, TIME_SCALE, TYPE_INDEX, signed_squash, squash
    t = np.zeros((rows, FUTURE_JOB_FEATURES), dtype=np.float32)
    now = float(twin.now)
    n_machines = max(len(twin.floor.machines), 1)
    cum = 0.0
    for i, job in enumerate(future_jobs(twin, horizon)[:rows]):
        r = t[i]
        offset = float(job.arrival) - now
        means = [float(np.mean([float(v) for v in op.values()])) for op in job.ops]
        mins = [float(min(op.values())) for op in job.ops]
        work, min_work = sum(means), sum(mins)
        cum += work
        r[0] = 1.0
        r[1] = min(max(offset / horizon, 0.0), 1.0)
        r[2] = squash(offset, TIME_SCALE)
        r[3] = squash(i, RANK_SCALE)
        r[4] = squash(cum / n_machines, CUM_WORK_SCALE)
        r[5] = squash(len(job.ops), OPS_SCALE)
        r[6] = squash(work, AGE_SCALE)
        r[7] = squash(min_work, AGE_SCALE)
        if job.ops:
            r[8] = squash(mins[0], TIME_SCALE)
            r[9] = squash(len(job.ops[0]), COUNT_SCALE)
            r[19] = squash(float(np.mean([len(op) for op in job.ops])), COUNT_SCALE)
        for w, typ in zip(means, job.types):
            k = TYPE_INDEX.get(typ)
            if k is not None:
                r[10 + k] += w
        for k in range(len(TYPE_INDEX)):
            r[10 + k] = squash(r[10 + k], TYPE_WORK_SCALE)
        if job.due is not None:
            due = float(job.due)
            r[15] = 1.0
            r[16] = signed_squash(due - float(job.arrival) - min_work, AGE_SCALE)
            r[17] = signed_squash(due - now, AGE_SCALE)
            if min_work > 0:
                r[18] = signed_squash((due - float(job.arrival) - min_work) / min_work, SLACK_RATIO_SCALE)
    return t


def test_vectorized_table_equals_the_row_loop_and_caches_per_twin():
    rng = np.random.default_rng(0)
    types = ["Mill", "Lathe", "Weld", "Inspect", "Assemble"]
    jobs = []
    for i in range(150):
        ops = [{int(m): float(rng.uniform(20, 900)) for m in rng.choice(15, size=rng.integers(1, 4), replace=False)}
               for _ in range(rng.integers(1, 5))]
        arrival = float(np.round(rng.uniform(0, 9000), 3))
        due = None if i % 7 == 0 else arrival + float(rng.uniform(-200, 4000))
        jobs.append(_job(arrival, ops, [types[k] for k in rng.integers(0, 5, size=len(ops))], due))
    jobs.sort(key=lambda j: j.arrival)
    tw = _twin(0.0, jobs)
    for now in (0.0, 123.4, 2000.0, 5000.5, 8999.0):
        tw.now = now
        tw.pending_i = sum(j.arrival < now for j in jobs)
        for rows in (8, 64, 128):
            for _ in range(2):                                        # second call reads the per-twin cache
                assert np.array_equal(future_job_table(tw, rows, 7200.0), _reference_table(tw, rows, 7200.0))
    assert len(tw._future_rows) > 0


def _train(tmp_path, floor, run_id, *extra):
    import subprocess
    import sys
    env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, os.path.join(env_dir, "train.py"), "--twin", floor, "--scenario-generator", "randomized",
           "--random-warmup", "--episode-duration-seconds", "1200",
           "--params", json.dumps({"agv_count": 3, "failure_probability": 0.0, "due_date_allowance": 2.0}),
           "--reward-spec", os.path.join(env_dir, "config", "rewards", "tardiness.json"), "--train-seed", "0",
           "--num-envs", "2", "--rollout-length", "4", "--batch-size", "8", "--slot-seconds", "300",
           "--discount-horizon-s", "5400", "--device", "cpu", "--torch-threads", "2", "--stall-minutes", "0",
           "--results-dir", str(tmp_path), "--run-id", run_id, *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout, torch.load(tmp_path / run_id / "checkpoint.pt", map_location="cpu", weights_only=True)


def test_train_future_table_warm_start_and_critic_warmup(tmp_path):
    from tests.test_learning_stack_fixes import _fixture_floor
    floor = _fixture_floor(tmp_path)
    out, a = _train(tmp_path, floor, "a", "--future-table", "16", "--total-timesteps", "16")
    assert "Future job table: 16 rows" in out
    assert reads_future_table(a["model_state_dict"]) and a["run_settings"]["future_table"] == 16
    # warm start from a, value head only for the whole (one-update) run
    out, b = _train(tmp_path, floor, "b", "--future-table", "16", "--total-timesteps", "8",
                    "--init-from", str(tmp_path / "a" / "checkpoint.pt"), "--critic-warmup-steps", "1000")
    assert "Warm start" in out and "Critic warm-up" in out
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    changed = [k for k in sa if not torch.equal(sa[k], sb[k])]
    assert changed and all(k.startswith("actor_critic.critic.") for k in changed), changed
    assert b["global_step"] == 8
    # a mismatched table is refused
    import subprocess
    with pytest.raises(AssertionError):
        _train(tmp_path, floor, "c", "--total-timesteps", "8", "--init-from", str(tmp_path / "a" / "checkpoint.pt"))


def test_evaluate_runs_a_future_table_checkpoint(tmp_path):
    from evaluate import CheckpointPolicy
    from config import FusionConfig  # noqa: F401
    import train
    net = FutureSchedulingNetwork()
    path = tmp_path / "ck.pt"
    train.save_checkpoint(path, net, torch.optim.Adam(net.parameters()), 0, train.PPOConfig(), "tardiness", (15, 64),
                          settings={"future_table": 32, "future_horizon_s": 3600.0})
    pol = CheckpointPolicy(path)
    assert pol.future_table == (32, 3600.0)
    obs = {k: v[0].numpy() for k, v in make_dummy_obs(1).items()}
    obs["future_job_table"] = np.zeros((32, FUTURE_JOB_FEATURES), dtype=np.float32)
    assert len(pol(obs)) == 2
    bc = tmp_path / "bc.pt"                                            # a rq2-oracle-bc-future clone's settings
    train.save_checkpoint(bc, net, torch.optim.Adam(net.parameters()), 0, train.PPOConfig(), "tardiness", (15, 64),
                          settings={"future_rows": 128, "future_horizon": 7200.0})
    assert CheckpointPolicy(bc).future_table == (128, 7200.0)
