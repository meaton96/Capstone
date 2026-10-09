"""
@file branch.py
@brief Fork-based branching for twin rollouts (2026-10-09, dev-twin-speed): evaluate several continuations of a
       twin paused at a decision without replaying the episode from t = 0.

@details A rollout scores each option at a decision by playing the rest of the episode with a base policy. Replaying
from t = 0 for every option (rq2-direct-headroom) costs the whole warm-up and the elapsed window again each time.
Here the process forks at the decision: each child owns a copy-on-write snapshot of the whole interpreter (the twin,
the paused agent_decisions() generator, any module-level state such as a FORCE wrapper), applies its option, plays to
the end, sends its score back over a pipe and exits with os._exit. The parent never advances, so after the call it is
still at the same decision and can apply the chosen option and go on.

Determinism: the twin has no random numbers or clock reads (engine, motion, rules, floor), and a child starts from
the parent's exact memory, so a child's result equals a replay from t = 0 with the same choices
(results/dev-twin-speed/check_branch.py verifies this with ==). A caller whose policy or scorer draws random numbers
gets the parent's RNG state in every child: seed per option inside the callback if that matters.

Linux / POSIX only (os.fork). Fork the twin process before starting threads where possible: a child of a
multi-threaded parent (e.g. after torch has started its pools) gets only the forking thread, so the callbacks must
not need the others (plain twin code and numpy are fine). The children do not run atexit handlers or flush the
parent's buffers (stdout and stderr are flushed before each fork).

@par Usage
@code{.py}
from des_twin import branch
gen = tw.agent_decisions(); dec = gen.send(None)
...                                              # at some decision
opts = [(j, m) for j in dec.job_candidates for m in tw.candidate_machines(tw.jobs[j])]   # forced choices
scores = branch.rollout(tw, gen, dec, opts, policy=lambda tw, dec: ("MDD", "TECT"),
                        score=lambda tw: my_window_tardiness(tw), max_parallel=8, timeout=600)
best = opts[min(range(len(opts)), key=scores.__getitem__)]
tw.force(*best); dec = gen.send(("MDD", "TECT"))   # the parent goes on with the kept choice
@endcode
"""
import os
import pickle
import selectors
import signal
import sys
import time
import traceback


class BranchError(RuntimeError):
    """A child raised (the message carries its traceback), died, or ran past the timeout."""


def fork_map(fn, items, max_parallel=None, timeout=None, return_exceptions=False):
    """@brief fn(item) for every item, each in a forked child of the current process; results in item order.

    @param max_parallel  children alive at once (default: os.cpu_count()); 1 runs them one after another
    @param timeout       seconds per child from its fork; a child past it is killed (SIGKILL) and reported
    @param return_exceptions  put a BranchError in the result list instead of raising the first one
    fn's return value must be picklable. Nothing fn does in a child is visible in the parent."""
    items = list(items)
    if not hasattr(os, "fork"):
        raise OSError("des_twin.branch needs os.fork (Linux / POSIX)")
    max_parallel = max(1, max_parallel or os.cpu_count() or 1)
    results = [None] * len(items)
    pending = list(range(len(items)))[::-1]
    active = {}                                    # read fd -> [index, pid, chunks, deadline]
    sel = selectors.DefaultSelector()
    try:
        while pending or active:
            while pending and len(active) < max_parallel:
                i = pending.pop()
                r, w = os.pipe()
                sys.stdout.flush()
                sys.stderr.flush()
                pid = os.fork()
                if pid == 0:                                       # child
                    os.close(r)
                    code = 0
                    try:
                        try:
                            payload = pickle.dumps((True, fn(items[i])), protocol=pickle.HIGHEST_PROTOCOL)
                        except BaseException:                       # noqa: BLE001 (report anything to the parent)
                            payload = pickle.dumps((False, traceback.format_exc()))
                        view = memoryview(payload)
                        while view:
                            view = view[os.write(w, view):]
                    except BaseException:                           # noqa: BLE001
                        code = 1
                    finally:
                        os._exit(code)
                os.close(w)
                active[r] = [i, pid, [], None if timeout is None else time.monotonic() + timeout]
                sel.register(r, selectors.EVENT_READ)
            wait = None
            if timeout is not None:
                wait = max(0.0, min(v[3] for v in active.values()) - time.monotonic())
            for key, _ in sel.select(wait):
                r = key.fd
                chunk = os.read(r, 1 << 20)
                if chunk:
                    active[r][2].append(chunk)
                    continue
                i, pid, chunks, _ = active.pop(r)                  # EOF: the child is done
                sel.unregister(r)
                os.close(r)
                _, status = os.waitpid(pid, 0)
                results[i] = _decode(chunks, status, items[i])
            if timeout is not None:
                now = time.monotonic()
                for r in [r for r, v in active.items() if v[3] <= now]:
                    i, pid, _, _ = active.pop(r)
                    sel.unregister(r)
                    os.close(r)
                    _kill(pid)
                    results[i] = BranchError(f"branch {items[i]!r}: no result within {timeout} s (killed)")
    finally:
        for r, (_, pid, _, _) in active.items():                   # an exception in the parent: no orphans
            sel.unregister(r)
            os.close(r)
            _kill(pid)
        sel.close()
    if not return_exceptions:
        for res in results:
            if isinstance(res, BranchError):
                raise res
    return results


def _kill(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    os.waitpid(pid, 0)


def _decode(chunks, status, item):
    data = b"".join(chunks)
    if not data:
        return BranchError(f"branch {item!r}: child exited without a result (wait status {status})")
    ok, value = pickle.loads(data)
    return value if ok else BranchError(f"branch {item!r} raised in the child:\n{value}")


def play_to_end(tw, gen, first, policy):
    """@brief Sends `first` (a rule-half pair) to the paused generator, then policy(tw, dec) -> pair at every later
    decision until the episode ends. Returns the number of decisions answered."""
    n = 0
    pair = first
    try:
        while True:
            dec = gen.send(tuple(pair))
            n += 1
            pair = policy(tw, dec)
    except StopIteration:
        pass
    return n


def rollout(tw, gen, dec, options, policy, score, max_parallel=None, timeout=None, return_exceptions=False):
    """@brief Scores each option at the decision the generator is paused at, by playing on with `policy` in a forked
    child; returns score(tw) per option (in order). The parent's twin and generator do not move.

    @param tw, gen   the Twin and its agent_decisions() generator, paused at a decision (after a yield, before the
                     send that answers it)
    @param dec       the Decision it is paused at
    @param options   each either a rule-half pair, e.g. ("ATC", "ECT") (that pair answers this decision), or a forced
                     choice (job, machine): Twin.force(job, machine) and then policy's pair for this decision (machine
                     is ignored for a dispatch; None leaves a half to the rule)
    @param policy    (tw, dec) -> pair; answers this decision for a forced choice, and every later decision
    @param score     tw -> picklable value, called in the child once the episode has ended"""
    def one(opt):
        if isinstance(opt[0], str):
            first = opt
        else:
            tw.force(opt[0], opt[1])
            first = policy(tw, dec)
        play_to_end(tw, gen, first, policy)
        return score(tw)
    return fork_map(one, options, max_parallel=max_parallel, timeout=timeout, return_exceptions=return_exceptions)
