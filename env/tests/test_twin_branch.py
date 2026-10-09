"""des_twin.branch (fork-based rollouts) and Twin.force (2026-10-09, dev-twin-speed). The full check against
rq2-direct-headroom's replay on B2 and the m50 floor is results/dev-twin-speed/check_branch.py."""
import json
import os
import time

import pytest

from des_twin import Floor, TwinConfig
from des_twin import branch
from des_twin.engine import Twin

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "des_twin", "convoy_agv3")
BASE = ("SPT", "ECT")


def _twin():
    with open(os.path.join(RUN, "des_floor.json")) as f:
        floor = Floor(json.load(f))
    with open(os.path.join(RUN, "des_jobs.json")) as f:
        jobs = json.load(f)
    return Twin(floor, jobs, TwinConfig("SPT_ECT", warmup_seconds=50.0))


def _end(tw):
    return tw.summary(), tw.metrics(), list(tw.decision_rows)


def _now(tw):
    return tw.now, tw.metrics(), list(tw.decision_rows), list(tw.trace)


def _replay(stop_at, opt):
    tw = _twin()
    gen = tw.agent_decisions()
    dec = gen.send(None)
    for _ in range(stop_at):
        dec = gen.send(BASE)
    if isinstance(opt[0], str):
        branch.play_to_end(tw, gen, opt, lambda t, d: BASE)
    else:
        tw.force(*opt)
        branch.play_to_end(tw, gen, BASE, lambda t, d: BASE)
    return _end(tw)


def _options(tw, dec):
    if dec.kind == "dispatch":
        return [(j, dec.machine) for j in dec.queue]
    return [(j, m) for j in dec.job_candidates for m in tw.candidate_machines(tw.jobs[j])]


def test_fork_map_order_errors_and_timeout():
    assert branch.fork_map(lambda x: x * x, range(7), max_parallel=3) == [x * x for x in range(7)]
    big = branch.fork_map(lambda n: "x" * n, [3 << 20], max_parallel=1)       # larger than a pipe buffer
    assert len(big[0]) == 3 << 20

    def boom(x):
        if x == 2:
            raise KeyError("no")
        return x
    with pytest.raises(branch.BranchError, match="KeyError"):
        branch.fork_map(boom, range(4))
    res = branch.fork_map(boom, range(4), return_exceptions=True)
    assert res[:2] == [0, 1] and isinstance(res[2], branch.BranchError) and res[3] == 3
    t = time.monotonic()
    res = branch.fork_map(lambda s: time.sleep(s) or s, [0.0, 30.0], timeout=1.0, return_exceptions=True)
    assert res[0] == 0.0 and isinstance(res[1], branch.BranchError) and time.monotonic() - t < 10
    assert branch.fork_map(lambda x: x, []) == []


def test_child_changes_stay_in_the_child():
    box = [0]

    def bump(x):
        box[0] += x
        return box[0]
    assert branch.fork_map(bump, [5, 7]) == [5, 7]
    assert box == [0]


def test_rollout_equals_replay_and_parent_stays_put():
    tw = _twin()
    gen = tw.agent_decisions()
    dec = gen.send(None)
    checked = 0
    idx = 0
    while checked < 3:
        opts = _options(tw, dec)
        if len(opts) > 1:
            opts = opts + [("LPT", "SMPT")]
            before = _now(tw)
            got = branch.rollout(tw, gen, dec, opts, lambda t, d: BASE, _end, max_parallel=4)
            assert _now(tw) == before
            assert got == [_replay(idx, o) for o in opts]
            checked += 1
        dec = gen.send(BASE)
        idx += 1


def test_force_rejects_an_illegal_choice():
    tw = _twin()
    gen = tw.agent_decisions()
    dec = gen.send(None)
    tw.force(10 ** 9)
    with pytest.raises(ValueError):
        gen.send(BASE)


def test_force_none_leaves_the_rule():
    a, b = _twin(), _twin()
    ga, gb = a.agent_decisions(), b.agent_decisions()
    ga.send(None)
    gb.send(None)
    b.force(None, None)
    branch.play_to_end(a, ga, BASE, lambda t, d: BASE)
    branch.play_to_end(b, gb, BASE, lambda t, d: BASE)
    assert _end(a) == _end(b)
