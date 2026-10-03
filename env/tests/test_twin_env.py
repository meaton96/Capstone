"""
@file test_twin_env.py
@brief Tests for training on the event-based twin: scenario resolution (des_twin.scenario), the engine's agent mode,
       warm-up and time cap, the observation port (des_twin.observation) and the gym wrapper (env_wrappers.twin_env).

The convoy fixtures (test_des_twin.py) predate the observation frame in des_floor.json, so the observation tests add
a synthetic one (machines at their pickup/dropoff midpoint, grid over the zones' extent). Exactness against the player
is the job of des_twin/parity.py, which needs a player built after 2026-10-01.

@par Usage
@code{.sh}
cd env && python -m pytest tests/test_twin_env.py -v
@endcode
"""

import copy
import json
import math
import os

import numpy as np
import pytest

from config import ACTION_MASK_LEN, JOB_HEAD_RULES, MACHINE_HEAD_RULES, obs_shapes
from des_twin import Floor, TwinConfig, run_twin
from des_twin.engine import Decision, Twin
from des_twin.observation import ObservationBuilder, action_mask
from des_twin.scenario import episode_settings, resolve_jobs
from rewards import load_reward

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "des_twin")
REWARDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "rewards")


def _load(name):
    with open(os.path.join(FIX, name)) as f:
        return json.load(f)


def _scenario():
    return _load("convoy_waves.json")


def with_frame(data):
    """A floor export plus a synthetic observation frame (the fixtures predate it)."""
    data = copy.deepcopy(data)
    xs = [z["centre"][0] for z in data["zones"]]
    zs = [z["centre"][1] for z in data["zones"]]
    data["obs"] = {"grid_centre": [(min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2],
                   "grid_half": [(max(xs) - min(xs)) / 2 + 2, (max(zs) - min(zs)) / 2 + 2],
                   "floor_size": [max(xs) - min(xs) + 4, max(zs) - min(zs) + 4]}
    floor = Floor(data)
    builder = ObservationBuilder.__new__(ObservationBuilder)
    builder.cx, builder.cz = (np.float32(v) for v in data["obs"]["grid_centre"])
    builder.hw, builder.hd = (np.float32(v) for v in data["obs"]["grid_half"])
    for m in data["machines"]:
        x = (m["pickup_pos"][0] + m["dropoff_pos"][0]) / 2
        z = (m["pickup_pos"][1] + m["dropoff_pos"][1]) / 2
        m["pos3"], m["cell"] = [x, 0.0, z], list(builder.cell(x, z))
    t = data["tiles"][0]
    t["incoming_pos3"] = [t["incoming_pos"][0], 0.5, t["incoming_pos"][1]]
    t["incoming_cell"] = list(builder.cell(*t["incoming_pos"]))
    return data


@pytest.fixture(params=["convoy_agv1", "convoy_agv3"])
def run_dir(request):
    return os.path.join(FIX, request.param)


def _floor(run_dir, frame=False):
    with open(os.path.join(run_dir, "des_floor.json")) as f:
        data = json.load(f)
    return Floor(with_frame(data) if frame else data)


# ── Scenario resolution ────────────────────────────────────────────────────────────────────────────

def test_resolve_jobs_matches_unity_export(run_dir):
    floor = _floor(run_dir)
    with open(os.path.join(run_dir, "des_jobs.json")) as f:
        ref = json.load(f)

    def norm(d):
        return [(j["id"], np.float32(j["arrival"]),
                 [(o["type"], [(int(m), np.float32(x)) for m, x in o["eligible"]]) for o in j["ops"]])
                for j in d["jobs"]]

    assert norm(resolve_jobs(_scenario(), floor)) == norm(ref)


def test_resolve_jobs_index_forms():
    floor = _floor(os.path.join(FIX, "convoy_agv1"))
    mills = sorted(m.id for m in floor.machines if m.type == "Mill")
    s = {"machineTypeLayout": [m.type for m in floor.machines], "jobs": [
        {"id": 7, "arrivalTime": 1.1, "operations": [
            {"machineType": "mill", "machineIndex": "any", "duration": 10},
            {"machineType": "Mill", "machineIndex": 2, "duration": 20},
            {"machineType": "Mill", "machineIndex": [2, 0], "duration": [30, 40]}]}]}
    job = resolve_jobs(s, floor)["jobs"][0]
    assert job["arrival"] == float(str(np.float32(1.1)))
    assert job["ops"][0]["eligible"] == [[m, 10.0] for m in mills]
    assert job["ops"][1]["eligible"] == [[mills[2], 20.0]]
    assert job["ops"][2]["eligible"] == [[mills[2], 30.0], [mills[0], 40.0]]   # array order kept
    s["jobs"][0]["operations"][1]["machineIndex"] = 3
    with pytest.raises(ValueError):
        resolve_jobs(s, floor)
    with pytest.raises(NotImplementedError):
        resolve_jobs(dict(s, machineFlexibilityProbability=0.3), floor)


def test_episode_settings_refuses_failures():
    assert episode_settings({"stochastic": {"warmupSeconds": 60, "episodeDurationSeconds": 600},
                             "dispatchingRule": "SPT_SMPT"}) == (60.0, 600.0, "SPT_SMPT")
    with pytest.raises(NotImplementedError):
        episode_settings({"stochastic": {"machineFailuresEnabled": True}})


# ── Agent mode ─────────────────────────────────────────────────────────────────────────────────────

def _jobs(run_dir):
    with open(os.path.join(run_dir, "des_jobs.json")) as f:
        return json.load(f)


@pytest.mark.parametrize("rule", ["SPT_ECT", "PTWINQ_SRWT", "FIFO_TECT"])
def test_agent_mode_with_a_fixed_answer_equals_the_rule_run(run_dir, rule):
    floor = _floor(run_dir)
    ref = run_twin(floor, _jobs(run_dir), TwinConfig(rule))
    twin = Twin(floor, _jobs(run_dir), TwinConfig("SRT_SRWT"))   # the config rule must not be used
    gen = twin.agent_decisions()
    halves = tuple(rule.split("_"))
    n = 0
    try:
        dec = next(gen)
        while True:
            assert isinstance(dec, Decision)
            n += 1
            dec = gen.send(halves)
    except StopIteration:
        pass
    assert n == ref.decisions == twin.decisions
    assert twin.decision_rows == ref.decision_rows
    assert twin.summary()["mean_flow_time"] == ref.summary()["mean_flow_time"]
    assert twin.trace == ref.trace


def test_decision_requests_follow_unity(run_dir):
    floor = _floor(run_dir)
    twin = Twin(floor, _jobs(run_dir), TwinConfig("SPT_ECT"))
    gen = twin.agent_decisions()
    seen = {"routing": 0, "dispatch": 0}
    try:
        dec = next(gen)
        while True:
            seen[dec.kind] += 1
            if dec.kind == "routing":
                routable = [j.id for j in twin.order if j.state == 0 and j.id in dec.job_candidates]
                assert dec.job == dec.job_candidates[0] == routable[0]     # oldest routable job is the focus
                assert dec.selected_by_rule == (len(dec.job_candidates) == 1)
                assert list(dec.machines) == twin.candidate_machines(twin.jobs[dec.job])
            else:
                m = twin.mach[dec.machine]
                assert m.idle and dec.queue
            dec = gen.send(("SPT", "ECT"))
    except StopIteration:
        pass
    assert seen["routing"] and seen["dispatch"]


def test_warmup_uses_the_rule_and_the_cap_truncates(run_dir):
    floor = _floor(run_dir)
    full = run_twin(floor, _jobs(run_dir), TwinConfig("SPT_ECT"))
    warm, cap = 200.0, 300.0
    twin = Twin(floor, _jobs(run_dir), TwinConfig("SPT_ECT", warmup_seconds=warm, episode_duration_seconds=cap))
    times = []
    gen = twin.agent_decisions()
    try:
        next(gen)
        while True:
            times.append(twin.now)
            gen.send(("SPT", "ECT"))
    except StopIteration:
        pass
    assert times and min(times) >= warm
    assert twin.truncated and not twin.timed_out
    assert twin.makespan > warm + cap >= twin.makespan - twin.dt
    assert twin.metrics()["truncated"] == 1.0 and twin.metrics()["episode_active"] == 0.0
    assert full.makespan > twin.makespan       # the convoy run is longer than the cap
    # Before the cap the agent (same rule) and the warm-up rule make the same decisions as the full run.
    assert twin.decision_rows == [r for r in full.decision_rows if r[0] <= twin.makespan][:len(twin.decision_rows)]


def test_metrics_track_flow_time(run_dir):
    twin = run_twin(_floor(run_dir), _jobs(run_dir), TwinConfig("SPT_ECT"))
    m, s = twin.metrics(), twin.summary()
    assert m["jobs_exited"] == m["jobs_total"] == s["jobs"]
    assert m["all_jobs_exited"] == 1.0 and m["wip"] == 0
    assert m["flow_time_exited_sum"] == pytest.approx(s["mean_flow_time"] * s["jobs"], rel=1e-6)
    assert m["time_in_system_sum"] == pytest.approx(m["flow_time_exited_sum"], rel=1e-6)


# ── Observation ────────────────────────────────────────────────────────────────────────────────────

def test_action_mask_is_heads_that_matter():
    def mask(job, mach):
        return ([1.0] + [float(job)] * (len(JOB_HEAD_RULES) - 1)
                + [1.0] + [float(mach)] * (len(MACHINE_HEAD_RULES) - 1))
    assert action_mask(Decision("dispatch", machine=0, queue=(1,))).tolist() == mask(False, False)
    assert action_mask(Decision("dispatch", machine=0, queue=(1, 2))).tolist() == mask(True, False)
    one = Decision("routing", job=1, job_candidates=(1,), selected_by_rule=True, machines=(0,))
    assert action_mask(one).tolist() == mask(False, False)
    assert action_mask(Decision("routing", job=1, job_candidates=(1,), machines=(0, 5))).tolist() == mask(False, True)
    pool = Decision("routing", job=1, job_candidates=(1, 2), selected_by_rule=False, machines=(0,))
    assert action_mask(pool).tolist() == mask(True, True)       # the picked job's candidates are unknown
    assert len(action_mask(pool)) == ACTION_MASK_LEN


def test_builder_needs_the_observation_frame():
    with pytest.raises(ValueError, match="observation frame"):
        ObservationBuilder(_floor(os.path.join(FIX, "convoy_agv1")), 15, 64)


def test_observation_contents(run_dir):
    floor = _floor(run_dir, frame=True)
    builder = ObservationBuilder(floor, 15, 64)
    shapes = obs_shapes(15, 64)
    twin = Twin(floor, _jobs(run_dir), TwinConfig("SPT_ECT"))
    gen = twin.agent_decisions()
    checked = 0
    try:
        dec = next(gen)
        while True:
            obs = builder.build(twin, dec)
            assert {k: v.shape for k, v in obs.items()} == shapes
            assert all(v.dtype == np.float32 for v in obs.values())
            mt, jt = obs["machine_table"], obs["job_table"]
            live = [j for j in twin.order if j.state != 5]
            assert mt[:, 0].sum() == len(twin.machines) and jt[:, 0].sum() == min(len(live), 64)
            assert obs["event_flags"][1 if dec.kind == "routing" else 0] == 1.0
            if dec.kind == "routing":
                assert sorted(np.flatnonzero(mt[:, 13])) == sorted(twin.machines.index(twin.mach[m])
                                                                    for m in dec.machines)
                assert jt[0, 14] == 1.0 and jt[1:, 14].sum() == 0          # focus row first
                assert jt[:, 15].sum() == len(set(dec.job_candidates))
                assert (mt[:, 14] > 0).sum() == len(dec.machines)
            else:
                assert mt[:, 13].sum() == 1 and jt[:, 15].sum() == len(dec.queue)
                assert (jt[:, 16] > 0).sum() == len(dec.queue)
            grid = obs["factory_grid"]
            assert (grid[0] > 0).sum() <= len(twin.machines)
            assert (grid[2] > 0).sum() >= 1 and (grid[2] > 0).sum() <= len(twin.agvs)
            assert np.all((grid >= 0) & (grid <= 1))
            checked += 1
            dec = gen.send(("SPT", "ECT"))
    except StopIteration:
        pass
    assert checked == twin.decisions


def test_instant_observation_reports_a_parked_fleet():
    floor = _floor(os.path.join(FIX, "convoy_agv3"), frame=True)
    twin = Twin(floor, _jobs(os.path.join(FIX, "convoy_agv3")), TwinConfig("SPT_ECT", transport="instant"))
    dec = next(twin.agent_decisions())
    obs = ObservationBuilder(floor, 15, 64, instant_fleet=3).build(twin, dec)
    assert obs["event_flags"][2] == 1.0 and obs["global_scalars"][9] == 0.0
    assert obs["global_scalars"][13] == pytest.approx(3 / 15)
    assert (obs["factory_grid"][2] == 0.25).sum() >= 1


# ── Gym wrapper ────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def floor_path(tmp_path):
    with open(os.path.join(FIX, "convoy_agv3", "des_floor.json")) as f:
        data = with_frame(json.load(f))
    path = tmp_path / "des_floor.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_seed_split_matches_the_unity_wrapper():
    from env_wrappers import twin_env, unity_env
    from channels.channels import EpisodeSeedChannel
    assert twin_env.TRAIN_SEED_LOW == unity_env.TRAIN_SEED_LOW
    assert twin_env.MAX_SEED == EpisodeSeedChannel.MAX_SEED
    for stream in (0, 571_656):
        a = twin_env.instance_seed_rng(7, 2, stream).integers(0, 1 << 24, 5)
        b = unity_env.instance_seed_rng(7, 2, stream).integers(0, 1 << 24, 5)
        assert (a == b).all()


def test_seed_stream_zero_keeps_the_original_key_and_a_resume_draws_new_seeds():
    import numpy as np
    from env_wrappers.twin_env import instance_seed_rng
    fresh = np.random.default_rng([7, 2]).integers(0, 1 << 24, 20)
    assert (instance_seed_rng(7, 2, 0).integers(0, 1 << 24, 20) == fresh).all()
    resumed = instance_seed_rng(7, 2, 571_656).integers(0, 1 << 24, 20)
    assert len(set(fresh) & set(resumed)) == 0


def test_vectorized_twin_passes_the_seed_stream(floor_path):
    from env_wrappers.twin_env import VectorizedTwinEnv
    a = VectorizedTwinEnv(2, floor_path, train_seed=3, scenario=_scenario(), obs_caps=(15, 64))
    b = VectorizedTwinEnv(2, floor_path, train_seed=3, scenario=_scenario(), obs_caps=(15, 64), seed_stream=99)
    draw = lambda v: [int(e.seed_rng.integers(0, 1 << 24)) for e in v.envs]
    assert draw(a) != draw(b)


def test_env_episode_reward_is_total_flow_time(floor_path):
    from env_wrappers.twin_env import TwinSchedulingEnv
    reward = load_reward(os.path.join(REWARDS, "flow_time.json"))
    scenario = dict(_scenario(), agvCount=3)
    env = TwinSchedulingEnv(floor_path, reward_fn=reward.build(), scenario=scenario, obs_caps=(15, 64))
    obs = env.reset()
    assert {k: v.shape for k, v in obs.items()} == obs_shapes(15, 64)
    total, steps = 0.0, 0
    while True:
        last = obs
        obs, r, done, info = env.step((0, 0))   # SPT, ECT
        total += r
        steps += 1
        if done:
            break
    ep = info["episode"]
    assert not ep["truncated"] and ep["jobs_exited"] == ep["jobs_total"]
    assert ep["length"] == steps
    assert total == pytest.approx(-ep["total_flow_time"] / 1000.0, rel=1e-4)
    # terminal_obs is the ended episode's last decision, not a zero pad (the player's own terminal obs).
    assert info["terminal_obs"] is last and info["terminal_obs"]["job_table"].any()
    ref = run_twin(env.floor, resolve_jobs(scenario, env.floor), TwinConfig("SPT_ECT"))
    assert ep["mean_flow_time"] == pytest.approx(ref.summary()["mean_flow_time"], rel=1e-5)
    assert obs["action_mask"].shape == (ACTION_MASK_LEN,)      # already the next episode's first decision


def test_env_reports_each_steps_simulated_duration(floor_path):
    """info["dt"] (the trainer's time discount) tiles the episode: the steps' durations sum to the time from the
    first decision to the end."""
    from env_wrappers.twin_env import TwinSchedulingEnv
    reward = load_reward(os.path.join(REWARDS, "flow_time.json"))
    env = TwinSchedulingEnv(floor_path, reward_fn=reward.build(), scenario=dict(_scenario(), agvCount=3),
                            obs_caps=(15, 64))
    env.reset()
    start = env._prev_metrics.sim_time
    dts = []
    while True:
        _, _, done, info = env.step((0, 0))
        dts.append(info["dt"])
        if done:
            break
    assert all(dt >= 0 for dt in dts) and max(dts) > 0
    assert sum(dts) == pytest.approx(info["episode"]["makespan"] - start, rel=1e-6)


def test_env_refuses_a_mismatched_floor(floor_path):
    from env_wrappers.twin_env import TwinSchedulingEnv
    env = TwinSchedulingEnv(floor_path, scenario=dict(_scenario(), agvCount=7), obs_caps=(15, 64))
    with pytest.raises(ValueError, match="agvCount"):
        env.reset()


def test_vectorized_env_with_the_randomized_generator(floor_path):
    from env_wrappers.twin_env import VectorizedTwinEnv
    from scenarios.randomized import RandomizedParams, randomized_generator
    params = RandomizedParams(agv_count=3, failure_probability=0.0)
    gen = randomized_generator(600.0, random_warmup=True, params=params)
    vec = VectorizedTwinEnv(2, floor_path, transport="kinematic", reward_spec=os.path.join(REWARDS, "flow_time.json"),
                            train_seed=0, scenario_generator=gen, obs_caps=(15, 256))
    obs, _ = vec.reset()
    assert obs["job_table"].shape == (2, 256, 17)
    finished = []
    for _ in range(20000):
        actions = np.stack([np.random.randint(len(JOB_HEAD_RULES), size=2),
                            np.random.randint(len(MACHINE_HEAD_RULES), size=2)], axis=1)
        obs, rewards, dones, truncs, infos = vec.step(actions)
        finished += [i["episode"] for d, i in zip(dones, infos) if d]
        if len(finished) >= 2:
            break
    assert len(finished) >= 2
    for ep in finished:
        assert ep["truncated"] and ep["seed"] >= 10_000
        assert ep["makespan"] > 600.0


def test_train_build_env_passes_the_resumed_step_as_seed_stream(floor_path, tmp_path):
    import argparse
    import train
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps(_scenario()))
    args = argparse.Namespace(twin=floor_path, scenario=str(scenario), obs_max_machines=0, obs_max_jobs=0,
                              twin_transport="kinematic", train_seed=3, twin_instant_fleet=None)
    cfg = argparse.Namespace(num_envs=2)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    draw = lambda v: [int(e.seed_rng.integers(0, 1 << 24)) for e in v.envs]
    fresh, _ = train.build_env(args, cfg, run_dir, None)
    resumed, _ = train.build_env(args, cfg, run_dir, None, seed_stream=1234)
    assert draw(fresh) != draw(resumed)
    from env_wrappers.twin_env import instance_seed_rng
    resumed2, _ = train.build_env(args, cfg, run_dir, None, seed_stream=1234)
    assert draw(resumed2) == [int(instance_seed_rng(3, i, 1234).integers(0, 1 << 24)) for i in range(2)]


@pytest.mark.parametrize("extra", [{"agvMoveSpeed": 5.0}, {"agvHandshakeDuration": 0.5}, {"travelPrice": 1.0},
                                   {"ioDocks": "siding"}, {"routingTrigger": "onCompletion"},
                                   {"reservationProtocol": "holdPrevious"}])
def test_env_refuses_scenario_keys_the_twin_does_not_model(floor_path, extra):
    from env_wrappers.twin_env import TwinSchedulingEnv
    env = TwinSchedulingEnv(floor_path, scenario=dict(_scenario(), agvCount=3, **extra), obs_caps=(15, 64))
    with pytest.raises(ValueError, match=next(iter(extra))):
        env.reset()


def test_env_accepts_scenario_keys_that_match_the_export(floor_path):
    from env_wrappers.twin_env import TwinSchedulingEnv
    same = {"agvMoveSpeed": 3.5, "agvHandshakeDuration": 1.5, "travelPrice": 0.0, "ioDocks": "corner",
            "routingTrigger": "onTransport", "reservationProtocol": "releasePrevious"}
    env = TwinSchedulingEnv(floor_path, scenario=dict(_scenario(), agvCount=3, **same), obs_caps=(15, 64))
    env.reset()
