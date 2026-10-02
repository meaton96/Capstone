"""
@file test_evaluate.py
@brief Tests for evaluate.py: seed parsing, episode-to-policy attribution, and summaries.

@par Usage
@code{.sh}
cd env && python -m pytest tests/test_evaluate.py -v
@endcode
"""

import csv
import io

import numpy as np
import pytest

from config import ACTION_BRANCHES, ACTION_MASK_LEN, PDR_ACTIONS
from env_wrappers.unity_env import SEED_BUFFER
from evaluate import (
    DECISION_FIELDS, ConstantPolicy, decision_row, parse_seeds, resolve_pdr_names, run_evaluation, summarize,
)
from rewards import MetricsSnapshot


class FakeEnv:
    """@brief Mimics UnitySchedulingEnv episode rollover and the seed/scenario queues.

    @details The episode running at reset is unseeded (index -1); each later episode
    consumes the next queued seed (and scenario, if any were queued). Like the real side
    channels, queue messages only reach "Unity" with the next reset() or step(). Episodes
    last @p length steps and report makespan = seed * 10 + the last action's flat index
    (job * 3 + machine), so attribution is checkable. The episode index counts items
    consumed since the last clear.
    """

    def __init__(self, length=2):
        self.length = length
        self.queue, self.scenario_queue, self.consumed = [], [], 0
        self.pending = []            # messages not yet delivered to Unity
        self.queued_seeds = []       # every queue_seeds(seeds, clear) call, for assertions
        self.queued_scenarios = []   # every queue_scenarios(items, clear) call, for assertions
        self.max_ahead = 0           # most items ever waiting in Unity's seed queue
        self.scenario = None

    def queue_seeds(self, seeds, clear=False):
        self.queued_seeds.append((list(seeds), clear))
        self.pending.append(("seeds", list(seeds), clear))

    def queue_scenarios(self, items, clear=False):
        self.queued_scenarios.append((list(items), clear))
        self.pending.append(("scenarios", list(items), clear))

    def _deliver(self):
        for kind, items, clear in self.pending:
            if kind == "seeds":
                if clear:
                    self.queue, self.consumed = [], 0
                self.queue += items
            else:
                if clear:
                    self.scenario_queue = []
                self.scenario_queue += items
        self.pending = []
        self.max_ahead = max(self.max_ahead, len(self.queue))

    def reset(self):
        self._deliver()
        self._start(-1, -1)
        return {}

    def _start(self, seed, index):
        self.seed, self.index, self.t = seed, index, 0

    @property
    def current_metrics(self):
        return MetricsSnapshot.from_dict({"episode_seed": self.seed, "episode_seed_index": self.index})

    def step(self, action):
        self._deliver()
        self.t += 1
        if self.t < self.length:
            return {}, 0.0, False, {}
        episode = {
            "seed": self.seed, "seed_index": self.index, "makespan": self.seed * 10 + action[0] * ACTION_BRANCHES[1] + action[1],
            "mean_flow_time": 1.0, "total_flow_time": 2.0, "return": -1.0, "length": self.length,
            "jobs_exited": 15, "deadlock": False, "timed_out": False, "truncated": False,
            "scenario": self.scenario,
        }
        # Unity auto-resets immediately, with whatever is queued right now.
        if self.queue:
            self._start(self.queue.pop(0), self.consumed)
            self.scenario = self.scenario_queue.pop(0) if self.scenario_queue else None
            self.consumed += 1
        else:
            self._start(-1, -1)
            self.scenario = None
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
    assert resolve_pdr_names("spt_ect, FIFO-SRWT") == ["SPT-ECT", "FIFO-SRWT"]
    with pytest.raises(ValueError):
        resolve_pdr_names("NOT-A-RULE")
    with pytest.raises(ValueError):
        resolve_pdr_names("SPT-SMPT")   # a catalog rule outside the RL heads


def test_run_evaluation_attributes_episodes_to_policies():
    """@brief Each queued (policy, seed) runs under that policy; the unseeded startup episode is dropped."""
    policies = [ConstantPolicy((1, 0), "A"), ConstantPolicy((1, 2), "B")]
    schedule = [(p, seed) for seed in (7, 8) for p in range(2)]

    rows = run_evaluation(FakeEnv(), policies, schedule, log=lambda *_: None)

    assert [(r["policy"], r["seed"], r["makespan"]) for r in rows] == [
        ("A", 7, 73), ("B", 7, 75), ("A", 8, 83), ("B", 8, 85),
    ]
    assert [r["seed_index"] for r in rows] == [0, 1, 2, 3]


def test_run_evaluation_queues_matching_scenarios_when_generator_given():
    """@brief With a scenario_generator, each queued seed's scenario variant must be queued too,
    in the same order and in the same batches, cleared together with the seed queue."""
    policies = [ConstantPolicy((1, 0), "A")]
    schedule = [(0, seed) for seed in (11, 22, 33)]
    generator = lambda seed: {"name": f"variant-{seed}", "seed": seed}  # noqa: E731
    env = FakeEnv()
    run_evaluation(env, policies, schedule, log=lambda *_: None, scenario_generator=generator)
    assert [items for items, _ in env.queued_scenarios] == [
        [generator(seed) for seed in seeds] for seeds, _ in env.queued_seeds]
    assert [clear for _, clear in env.queued_scenarios] == [clear for _, clear in env.queued_seeds]
    assert sum((items for items, _ in env.queued_scenarios), []) == [generator(seed) for _, seed in schedule]
    assert env.queued_scenarios[0][1] is True


def test_run_evaluation_queues_incrementally_and_keeps_attribution():
    """@brief A long schedule is never queued at once (inline scenarios overflowed gRPC's 4 MB
    cap): the queues are cleared once, then topped up in schedule order a few items ahead of
    the running episode, and every episode still runs under its scheduled policy, seed and
    scenario variant."""
    # One policy per (job head, machine head) pair; its flat index is the makespan offset.
    policies = [ConstantPolicy(divmod(flat, ACTION_BRANCHES[1]), f"P{flat}")
                for flat in range(ACTION_BRANCHES[0] * ACTION_BRANCHES[1])]
    schedule = [(p, seed) for seed in range(20) for p in range(len(policies))]
    generator = lambda seed: {"name": f"variant-{seed}"}  # noqa: E731
    env = FakeEnv(length=3)
    rows = run_evaluation(env, policies, schedule, log=lambda *_: None, scenario_generator=generator)

    assert [(r["policy"], r["seed"], r["makespan"]) for r in rows] == [
        (f"P{p}", seed, seed * 10 + p) for p, seed in schedule]
    assert [r["seed_index"] for r in rows] == list(range(len(schedule)))
    # Only the first call clears; the concatenation of all calls is the schedule, in order.
    assert [clear for _, clear in env.queued_seeds] == [True] + [False] * (len(env.queued_seeds) - 1)
    assert sum((seeds for seeds, _ in env.queued_seeds), []) == [seed for _, seed in schedule]
    assert max(len(items) for items, _ in env.queued_scenarios) <= SEED_BUFFER
    assert env.max_ahead <= SEED_BUFFER
    # Unity never ran dry: only the startup episode was unseeded.
    assert len(env.queued_seeds) > 1


def test_run_evaluation_scenario_follows_its_seed():
    """@brief The scenario each episode actually ran with is the one generated for its seed."""
    schedule = [(0, seed) for seed in range(10)]
    env = FakeEnv(length=1)   # shortest episodes: the buffer must still stay ahead
    seen = []
    original_step = env.step

    def step(action):
        obs, reward, done, info = original_step(action)
        if done and info["episode"]["seed_index"] >= 0:
            seen.append((info["episode"]["seed"], info["episode"]["scenario"]))
        return obs, reward, done, info

    env.step = step
    run_evaluation(env, [ConstantPolicy((0, 0), "A")], schedule, log=lambda *_: None,
                   scenario_generator=lambda seed: {"seed": seed})
    assert seen == [(seed, {"seed": seed}) for _, seed in schedule]


def test_run_evaluation_queues_no_scenarios_without_generator():
    """@brief Without a scenario_generator (the default: evaluate on the generated config or a
    fixed --scenario), queue_scenarios must not be called at all."""
    env = FakeEnv()
    run_evaluation(env, [ConstantPolicy((1, 0), "A")], [(0, 1)], log=lambda *_: None)
    assert env.queued_scenarios == []


def test_decision_log_records_every_decision_of_scheduled_episodes():
    """@brief Each scored decision gets one row (step numbering restarts per episode); the
    discarded startup episode is not logged."""
    policies = [ConstantPolicy((1, 0), "A"), ConstantPolicy((1, 2), "B")]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=DECISION_FIELDS)
    writer.writeheader()

    run_evaluation(FakeEnv(length=3), policies, [(0, 7), (1, 7)], log=lambda *_: None,
                   decision_writer=writer)

    rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))
    assert [(r["policy"], r["step"], r["job_head"], r["machine_head"], r["rule"]) for r in rows] == [
        ("A", str(step), "1", "0", PDR_ACTIONS[3]) for step in range(3)
    ] + [("B", str(step), "1", "2", PDR_ACTIONS[5]) for step in range(3)]
    assert all(r["entropy"] == "" for r in rows)   # constant rules have no action distribution


def test_decision_row_includes_checkpoint_probabilities():
    class ProbabilisticPolicy:
        kind, name = "checkpoint", "P"
        last_probs = [np.array([0.7, 0.3, 0.0, 0.0]), np.array([0.5, 0.25, 0.25])]

    metrics = MetricsSnapshot.from_dict({"sim_time": 12.5, "wip": 3})
    row = decision_row(ProbabilisticPolicy(), seed=1, seed_index=0, step=4, metrics=metrics, action=(0, 1))

    assert row["chosen_prob"] == pytest.approx(0.7 * 0.25)
    assert row["p_job_SRT"] == 0.3 and row["p_machine_TECT"] == 0.25
    assert row["wip"] == 3 and row["rule"] == "SPT-TECT"
    assert row["job_head_used"] == 1 and row["machine_head_used"] == 1
    job_entropy = -(0.7 * np.log(0.7) + 0.3 * np.log(0.3))
    machine_entropy = -(0.5 * np.log(0.5) + 2 * 0.25 * np.log(0.25))
    assert row["entropy"] == pytest.approx(job_entropy + machine_entropy, abs=1e-4)


def test_decision_row_skips_masked_head_in_chosen_prob():
    """@brief A head Unity masked (could not change the decision) does not scale the chosen action's probability."""
    class ProbabilisticPolicy:
        kind, name = "checkpoint", "P"
        last_probs = [np.array([0.7, 0.3, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]

    mask = np.ones(ACTION_MASK_LEN)
    mask[ACTION_BRANCHES[0] + 1:] = 0.0
    metrics = MetricsSnapshot.from_dict({"sim_time": 1.0})
    row = decision_row(ProbabilisticPolicy(), 1, 0, 0, metrics, action=(1, 0), action_mask=mask)
    assert row["machine_head_used"] == 0
    assert row["chosen_prob"] == pytest.approx(0.3)


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
