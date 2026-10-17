"""
@file screen_diff.py
@brief rq4-uncert-v5-policy (2026-10-17): why MDD-PTECT here (path A: evaluate.py --twin --uncertainty ->
       TwinSchedulingEnv) differs from MDD-PTECT@3 in ../rq4-uncert-screen (path B: uc.UEpisode.play) in the E3, mix and
       info=I1 cells. Runs path B on the same frozen code as path A (../rq4-uncert-screen/code_realizable = 6d13412a =
       ../rq2-rebase-v5/code/env) and changes one thing at a time. Writes nothing under the frozen snapshots.
@details Two causes (README, "Reproduction check against the screen"):
  1. Measure (E3, mix). Path A's window tardiness is the Twin.metrics() tardiness_sum delta (last minus first agent
     decision), which sums ARRIVED jobs only (Twin.order); path B sums every job of the world over the window
     (des_twin.planner.window_tardiness). They differ only for jobs whose actual arrival is after their due date (E3
     lateness): path A books such a job's lateness before t0 when it arrives inside the window, and never counts one
     that arrives after t1. The difference depends only on arrivals, due dates, t0 and t1, not on the policy.
  2. Estimator sync (info=I1). ViewCore.bhat / ObservedState._ops read the I1 bias estimator without ViewCore.sync(),
     so a read sees only the completions applied by the last syncing read. Path A's observation build syncs at every
     agent decision; path B does not. "full" syncs before every estimator read (the documented semantics),
     "fullwin" does so from the first agent decision on.
@par Usage (repo root)
@code{.sh}
.venv/bin/python results/rq4-uncert-v5-policy/screen_diff.py repro "E3=severe" [--seed 0] [--rule MDD-PTECT]
.venv/bin/python results/rq4-uncert-v5-policy/screen_diff.py offset [--seeds 0-39]   # -> screen_offset.csv
@endcode
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["UC_CODE"] = "code_realizable"                 # 6d13412a, the code of path A (../rq2-rebase-v5/code/env)
sys.path.insert(0, str(HERE.parent / "rq4-uncert-screen"))
import screen                                              # noqa: E402
import uc                                                  # noqa: E402
from des_twin import observed as OB                       # noqa: E402
from des_twin.planner import overlap, window_tardiness    # noqa: E402

OFFSET_CELLS = ["E3=realistic", "E3=severe", "mix=realistic", "mix=severe", "mix=realistic,info=I1", "mix=severe,info=I1"]


def play(cell, seed, rule="MDD-PTECT", alpha=3.0, sync="none"):
    """Path B's slot-free play of one rule, with both measures. sync: "none" (path B), "decision" (ViewCore.sync()
    before each send, as path A's observation build), "full" (every I1 estimator read syncs first, the documented
    semantics), "fullwin" ("full" from the first agent decision on; the warm-up as paths A and B play it)."""
    _FULL["on"] = sync == "full"
    e = uc.UEpisode(seed, screen.ucfg(cell))
    tw = e.twin(ptime_alpha=alpha)
    gen = tw.agent_decisions()
    t0 = m0 = None
    try:
        gen.send(None)
        while True:
            if t0 is None:
                t0, m0 = tw.now, tw.metrics()["tardiness_sum"]
                _FULL["on"] = sync in ("full", "fullwin")
            if sync == "decision" and tw.unc is not None:
                tw._vcore.sync()
            gen.send(tuple(rule.split("-")))
    except StopIteration:
        pass
    _FULL["on"] = False
    return {"B": window_tardiness(tw, t0), "A": (tw.metrics()["tardiness_sum"] - m0) / 1000.0, "t0": t0, "t1": tw.now}


_FULL = {"on": False}


def _full_sync_patch():
    """While _FULL["on"]: every I1 estimator read syncs first (monkeypatch, this process only)."""
    b0, o0 = OB.ViewCore.bhat, OB.ObservedState._ops

    def bhat(self, mid):
        if _FULL["on"]:
            self.sync()
        return b0(self, mid)

    def _ops(self, job):
        if _FULL["on"]:
            self.core.sync()
        return o0(self, job)
    OB.ViewCore.bhat, OB.ObservedState._ops = bhat, _ops


_full_sync_patch()


def offset(cell, seed):
    """B - A (1000 job-s) from the warm-up alone: t0 = the first agent decision, t1 = the cap tick; jobs not in
    Twin.order at t0 that arrive by t1 with due < t0 add (t0 - due) to A; jobs not arrived by t1 add
    (t1 - max(due, t0))+ to B. Policy-independent (arrivals, due dates, t0 and t1 are)."""
    e = uc.UEpisode(seed, screen.ucfg(cell))
    tw = e.twin(ptime_alpha=3.0)
    gen = tw.agent_decisions()
    gen.send(None)
    t0, dt, k1 = tw.now, tw.dt, tw.cap_tick
    t1 = k1 * dt
    s0 = {j.id for j in tw.order}
    a_extra = b_extra = 0.0
    n_a = n_b = 0
    for j in tw.jobs.values():
        if j.due is None or j.id in s0:
            continue
        if j.arrival <= (k1 - 1) * dt + 1e-9:              # arrives inside the window (processed before the cap tick)
            if j.due < t0:
                a_extra += t0 - j.due
                n_a += 1
        elif j.due < t1:
            b_extra += overlap(j.due, t1, t0, t1)
            n_b += 1
    return {"cell": cell, "seed": seed, "t0": t0, "t1": t1, "offset_B_minus_A": (b_extra - a_extra) / 1000.0,
            "B_only_after_t1": b_extra / 1000.0, "A_only_pre_t0": a_extra / 1000.0, "n_after_t1": n_b, "n_pre_t0": n_a}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("repro")
    r.add_argument("cell")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--rule", default="MDD-PTECT")
    r.add_argument("--modes", default="none,decision,full,fullwin")
    o = sub.add_parser("offset")
    o.add_argument("--seeds", default="0-39")
    o.add_argument("--cells", default=";".join(OFFSET_CELLS), help="cells separated by \";\"")
    a = ap.parse_args(argv)
    if a.cmd == "repro":
        out = {"cell": a.cell, "seed": a.seed, "rule": a.rule}
        if a.rule == "MDD-PTECT":
            f = HERE.parent / "rq4-uncert-screen/out/rules" / a.cell.replace(",", "_").replace("=", "-") / f"s{a.seed}.json"
            out["screen"] = json.loads(f.read_text())["tardiness"]["MDD-PTECT@3"]
        for m in a.modes.split(","):
            out[m] = play(a.cell, a.seed, a.rule, sync=m)
        out["offset"] = offset(a.cell, a.seed)["offset_B_minus_A"]
        print(json.dumps(out), flush=True)
        return
    rows = [offset(c, s) for c in a.cells.split(";") for s in uc.sim.parse_seeds(a.seeds)]
    with open(HERE / "screen_offset.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for c in a.cells.split(";"):
        v = [x["offset_B_minus_A"] for x in rows if x["cell"] == c]
        print(f"{c:15s} B - A over {len(v)} seeds: mean {sum(v) / len(v):8.3f}  min {min(v):8.3f}  max {max(v):8.3f}")


if __name__ == "__main__":
    main()
