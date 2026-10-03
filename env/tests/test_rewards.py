"""
@file test_rewards.py
@brief Tests for the Python reward pipeline (env/rewards).

@par Usage
@code{.sh}
cd env && python -m pytest tests/test_rewards.py -v
@endcode
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from rewards import (
    METRIC_NAMES, SCHEMA_VERSION, MetricsSnapshot, RewardContext, load_reward, resolve,
)

ENV_ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = ENV_ROOT / "config" / "rewards"
CSHARP_METRICS = ENV_ROOT.parent / "Capstone" / "Assets" / "Scripts" / "Simulation" / "RewardMetrics.cs"


def snap(**values):
    """@brief Snapshot with the given metrics set and everything else 0."""
    return MetricsSnapshot.from_dict(values)


@pytest.mark.skipif(not CSHARP_METRICS.exists(), reason="C# sources not present (e.g. on the cluster)")
class TestMetricsContract:
    """@brief Python METRIC_NAMES must stay in lockstep with RewardMetrics.cs."""

    def test_names_match_csharp(self):
        src = CSHARP_METRICS.read_text()
        block = re.search(r"Names\s*=\s*\{(.*?)\};", src, re.S).group(1)
        block = re.sub(r"//[^\n]*", "", block)
        assert tuple(re.findall(r'"([^"]+)"', block)) == METRIC_NAMES

    def test_schema_version_matches_csharp(self):
        match = re.search(r"SchemaVersion\s*=\s*(\d+)", CSHARP_METRICS.read_text())
        assert int(match.group(1)) == SCHEMA_VERSION

    def test_wrong_length_rejected(self):
        values = np.zeros(len(METRIC_NAMES) - 1)
        values[0] = SCHEMA_VERSION
        with pytest.raises(ValueError, match="out of sync"):
            MetricsSnapshot(values)

    def test_older_schema_reads_with_newer_metrics_missing(self):
        """@brief A v4 player (no due-date metrics) is still read; using a v5 metric on it fails loudly."""
        from rewards.metrics import SCHEMA_LENGTHS
        values = np.zeros(SCHEMA_LENGTHS[4])
        values[0], values[METRIC_NAMES.index("time_in_system_sum")] = 4, 123.0
        snap = MetricsSnapshot(values)
        assert snap.version == 4 and snap.time_in_system_sum == 123.0 and not snap.has("tardiness_sum")
        assert list(snap)[-1] == "tick_error" and len(snap) == SCHEMA_LENGTHS[4]
        with pytest.raises(KeyError, match="needs schema v5"):
            snap["tardiness_sum"]
        with pytest.raises(AttributeError, match="needs schema v5"):
            snap.tardiness_sum
        assert np.isnan(snap.to_array()[METRIC_NAMES.index("tardiness_sum")])

    def test_all_zero_snapshot_rejected(self):
        with pytest.raises(ValueError, match="all-zero"):
            MetricsSnapshot(np.zeros(len(METRIC_NAMES)))

    def test_wrong_schema_rejected(self):
        values = np.zeros(len(METRIC_NAMES))
        values[0] = SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="schema"):
            MetricsSnapshot(values)

    def test_named_access(self):
        s = snap(sim_time=12.5, jobs_exited=3)
        assert s["sim_time"] == 12.5
        assert s.jobs_exited == 3
        assert s.wip == 0
        assert s.delta(snap(sim_time=2.5), "sim_time") == 10.0
        with pytest.raises(AttributeError):
            s.not_a_metric


class TestSchedules:
    """@brief Parameter schedules (@ref resolve)."""

    def test_constant(self):
        assert resolve(2, 10_000) == 2.0

    def test_linear(self):
        schedule = {"start": 0.0, "end": -1.0, "steps": 100, "begin": 50}
        assert resolve(schedule, 0) == 0.0
        assert resolve(schedule, 100) == pytest.approx(-0.5)
        assert resolve(schedule, 1000) == -1.0

    def test_piecewise(self):
        schedule = {"piecewise": [[0, 1.0], [100, 0.0], [200, 0.5]]}
        assert resolve(schedule, -5) == 1.0
        assert resolve(schedule, 50) == pytest.approx(0.5)
        assert resolve(schedule, 150) == pytest.approx(0.25)
        assert resolve(schedule, 500) == 0.5


@pytest.mark.parametrize("spec_file", sorted(SPEC_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_shipped_specs_load_and_compute(spec_file):
    """@brief Every spec in config/rewards must load and produce a finite reward."""
    reward = load_reward(spec_file).build()
    prev = snap(sim_time=100, time_in_system_sum=1000, jobs_exited=1)
    curr = snap(sim_time=110, time_in_system_sum=1100, jobs_exited=2)
    total, terms = reward(prev, curr, RewardContext())
    assert terms
    assert np.isfinite(total)
    assert total == pytest.approx(sum(terms.values()))


class TestShippedRewards:
    """@brief Behaviour of the reference reward functions."""

    def test_time_penalty_sums_to_makespan(self):
        reward = load_reward(SPEC_DIR / "time_penalty.json").build()
        scale = reward.params["time_scale"]
        times = [0.0, 4.0, 4.0, 11.5, 30.0]
        total = sum(reward(snap(sim_time=a), snap(sim_time=b), RewardContext())[0]
                    for a, b in zip(times, times[1:]))
        assert total == pytest.approx(-30.0 / scale)

    def test_flow_time_penalty_and_failure(self):
        reward = load_reward(SPEC_DIR / "flow_time.json").build()
        params = reward.params
        prev = snap(time_in_system_sum=500.0)
        curr = snap(time_in_system_sum=800.0, deadlock=1)

        _, mid = reward(prev, curr, RewardContext(done=False))
        assert mid == {"flow_time": pytest.approx(-300.0 / params["time_scale"])}

        _, end = reward(prev, curr, RewardContext(done=True))
        assert end["failure"] == -params["failure_penalty"]

    def test_tardiness_penalty_flow_weight_and_guard(self):
        reward = load_reward(SPEC_DIR / "tardiness.json").build()
        scale = reward.params["time_scale"]
        prev = snap(tardiness_sum=100.0, time_in_system_sum=1000.0, jobs_total=5, jobs_with_due_date=5)
        curr = snap(tardiness_sum=400.0, time_in_system_sum=1500.0, jobs_total=5, jobs_with_due_date=5, timed_out=1)
        _, mid = reward(prev, curr, RewardContext(done=False))
        assert mid == {"tardiness": pytest.approx(-300.0 / scale)}
        _, end = reward(prev, curr, RewardContext(done=True))
        assert end["failure"] == -reward.params["failure_penalty"]

        mixed = load_reward({"entry": "rewards/functions/tardiness.py:TardinessReward",
                             "params": {"flow_weight": 0.5}}).build()
        _, terms = mixed(prev, curr, RewardContext())
        assert terms["flow_time"] == pytest.approx(-0.5 * 500.0 / 1000.0)

        no_due = snap(tardiness_sum=0.0, jobs_total=5, jobs_with_due_date=0, episode_seed=7)
        with pytest.raises(ValueError, match="needs due dates"):
            reward(no_due, no_due, RewardContext())
        startup = snap(tardiness_sum=0.0, jobs_total=15, jobs_with_due_date=0, episode_seed=-1)   # player default
        assert reward(startup, startup, RewardContext())[1] == {"tardiness": 0.0}

    def test_window_outcomes(self):
        from rewards import window_outcomes
        first = snap(time_in_system_sum=1000.0, tardiness_sum=50.0)
        final = snap(time_in_system_sum=4000.0, tardiness_sum=650.0, tardiness_exited_sum=300.0,
                     jobs_exited_late=4, jobs_with_due_date=20)
        out = window_outcomes(first, final)
        assert out == {"window_time_in_system": 3000.0, "window_tardiness": 600.0, "tardiness_exited_sum": 300.0,
                       "jobs_exited_late": 4, "jobs_with_due_date": 20}
        assert set(window_outcomes(None, final).values()) == {None}

    def test_weighted_modes_and_schedule(self):
        spec = {
            "entry": "rewards/functions/weighted.py:WeightedReward",
            "params": {"terms": {
                "d": {"metric": "sim_time", "mode": "delta", "weight": -1.0},
                "v": {"metric": "wip", "mode": "value", "weight": {"start": 0.0, "end": 2.0, "steps": 10}},
                "t": {"metric": "jobs_exited", "mode": "terminal", "weight": 3.0},
            }},
        }
        reward = load_reward(spec).build()
        prev, curr = snap(sim_time=1.0), snap(sim_time=4.0, wip=5, jobs_exited=2)

        _, terms = reward(prev, curr, RewardContext(global_step=5))
        assert terms == {"d": -3.0, "v": pytest.approx(5.0)}

        _, terms = reward(prev, curr, RewardContext(global_step=10, done=True))
        assert terms == {"d": -3.0, "v": pytest.approx(10.0), "t": 6.0}

    def test_weighted_rejects_unknown_metric(self):
        with pytest.raises(ValueError, match="unknown metric"):
            load_reward({"entry": "rewards/functions/weighted.py:WeightedReward",
                         "params": {"terms": {"x": {"metric": "nope", "weight": 1}}}})


class TestLoader:
    """@brief Loading user reward files without touching the package."""

    def test_plain_function_from_file_and_archive(self, tmp_path):
        (tmp_path / "my_reward.py").write_text(
            "def shaped(prev, curr, ctx, scale=1.0):\n"
            "    return {'ops': scale * curr.delta(prev, 'ops_completed')}\n"
        )
        spec_file = tmp_path / "spec.json"
        spec_file.write_text(json.dumps({"entry": "my_reward.py:shaped", "params": {"scale": 0.5}}))

        loaded = load_reward(spec_file)
        total, terms = loaded.build()(snap(ops_completed=1), snap(ops_completed=5), RewardContext())
        assert total == 2.0
        assert terms == {"ops": 2.0}

        archived = loaded.archive(tmp_path / "run")
        assert (archived / "spec.json").exists()
        assert (archived / "my_reward.py").exists()

    def test_scalar_function_and_subclass_check(self, tmp_path):
        source = tmp_path / "r.py"
        source.write_text("def f(prev, curr, ctx):\n    return 1.5\n\nclass NotAReward:\n    pass\n")
        reward = load_reward({"entry": f"{source}:f"}).build()
        assert reward(snap(), snap(), RewardContext()) == (1.5, {"reward": 1.5})
        with pytest.raises(TypeError, match="RewardFunction"):
            load_reward({"entry": f"{source}:NotAReward"})

    def test_bad_entries(self):
        with pytest.raises(FileNotFoundError):
            load_reward({"entry": "does/not/exist.py:X"})
        with pytest.raises(ValueError):
            load_reward({"entry": "no_colon"})
