"""
@file test_evaluate.py
@brief Tests for evaluate.py: seed parsing, episode-to-policy attribution, and summaries.

@par Usage
@code{.sh}
cd env && python -m pytest tests/test_evaluate.py -v
@endcode
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PDR_ACTIONS
from evaluate import ConstantPolicy, parse_seeds, resolve_pdr_names, run_evaluation, summarize
from rewards import MetricsSnapshot


class FakeEnv:
    """@brief Mimics UnitySchedulingEnv episode rollover and the seed queue.

    @details The episode running at reset is unseeded (index -1); each later episode
    consumes the next queued seed. Episodes last @p length steps and report
    makespan = seed * 10 + last action, so attribution is checkable.
    """

    def __init__(self, length=2):
        self.length = length
        self.queue, self.consumed = [], 0

    def queue_seeds(self, seeds, clear=False):
        if clear:
            self.queue, self.consumed = [], 0
        self.queue += list(seeds)

    def reset(self):
        self._start(-1, -1)
        return {}

    def _start(self, seed, index):
        self.seed, self.index, self.t = seed, index, 0

    @property
    def current_metrics(self):
        return MetricsSnapshot.from_dict({"episode_seed": self.seed, "episode_seed_index": self.index})

    def step(self, action):
        self.t += 1
        if self.t < self.length:
            return {}, 0.0, False, {}
        episode = {
            "seed": self.seed, "seed_index": self.index, "makespan": self.seed * 10 + action,
            "mean_flow_time": 1.0, "total_flow_time": 2.0, "return": -1.0, "length": self.length,
            "jobs_exited": 15, "deadlock": False, "timed_out": False,
        }
        if self.queue:
            self._start(self.queue.pop(0), self.consumed)
            self.consumed += 1
        else:
            self._start(-1, -1)
        return {}, -1.0, True, {"episode": episode}


def test_parse_seeds():
    assert parse_seeds("0-3") == [0, 1, 2, 3]
    assert parse_seeds("5, 1,9") == [5, 1, 9]
    assert parse_seeds("0-2,10") == [0, 1, 2, 10]
    with pytest.raises(ValueError):
        parse_seeds("1,1")
    with pytest.raises(ValueError):
        parse_seeds("5-3")


def test_resolve_pdr_names():
    assert resolve_pdr_names("all") == list(PDR_ACTIONS)
    assert resolve_pdr_names("none") == []
    assert resolve_pdr_names("spt_smpt, FIFO-SRWT") == ["SPT-SMPT", "FIFO-SRWT"]
    with pytest.raises(ValueError):
        resolve_pdr_names("NOT-A-RULE")


def test_run_evaluation_attributes_episodes_to_policies():
    """@brief Each queued (policy, seed) runs under that policy; the unseeded startup episode is dropped."""
    policies = [ConstantPolicy(3, "A"), ConstantPolicy(5, "B")]
    schedule = [(p, seed) for seed in (7, 8) for p in range(2)]

    rows = run_evaluation(FakeEnv(), policies, schedule, log=lambda *_: None)

    assert [(r["policy"], r["seed"], r["makespan"]) for r in rows] == [
        ("A", 7, 73), ("B", 7, 75), ("A", 8, 83), ("B", 8, 85),
    ]
    assert [r["seed_index"] for r in rows] == [0, 1, 2, 3]


def test_summarize_gaps_against_best_pdr_per_seed():
    def row(policy, kind, seed, makespan, flow):
        return {"policy": policy, "kind": kind, "seed": seed, "makespan": makespan,
                "total_flow_time": flow, "mean_flow_time": flow / 15, "deadlock": False, "timed_out": False}

    rows = [
        row("A", "pdr", 1, 100, 1000), row("B", "pdr", 1, 110, 900), row("C", "checkpoint", 1, 95, 950),
        row("A", "pdr", 2, 200, 2000), row("B", "pdr", 2, 180, 2100), row("C", "checkpoint", 2, 190, 1900),
    ]
    summary = {e["policy"]: e for e in summarize(rows)}

    assert [e["policy"] for e in summarize(rows)] == ["C", "B", "A"]   # sorted by mean makespan
    assert summary["C"]["makespan_gap_pct"] == pytest.approx(100 * (-5 / 100 + 10 / 180) / 2)
    # Best PDR flow is 900 on seed 1 and 2000 on seed 2 (C's own 1900 is not a PDR result).
    assert summary["C"]["flow_gap_pct"] == pytest.approx(100 * (50 / 900 - 100 / 2000) / 2)
    assert summary["A"]["makespan_gap_pct"] == pytest.approx(100 * (0 + 20 / 180) / 2)
