"""
@file mfail.py
@brief rq4-mfail-ccal-resume (2026-10-28, user): rq4-mfail-ccal rerun with preempt-resume machine failures (the job in
       process waits out the repair and finishes the rest; before 10-28 its operation restarted). Code: the frozen v5
       snapshot of rq4-mfail-ccal (rq2-rebase-v5/code/env) plus only the failure-resume patch (code/env here,
       code/failure_resume.patch). Everything else is rq4-mfail-ccal's, below.

rq4-mfail-ccal (2026-10-14, user): machine failures on the 50-machine floor with due dates calibrated to the
       failures. With B2's c ~ U[1.75, 2.5] the failures triple tardiness (dev-hetero-tune / dev-pect-rule m50-unif:
       158 -> ~520 ks for MDD-TECT; MDD-PTECT 57.7 -> 172.8 on seed 0), so almost every job is late and due-date rules
       lose their leverage. Here the c range is scaled by S so that the share of late jobs under failures matches the
       failure-free floor, and the v5 rules are compared at no failures, failures at B2's c, and failures at the
       calibrated c.

@details Floor, instances and failures are dev-pect-rule's m50 family (harness.py / hetero.py there: 50 machines, 26
AGVs, load 0.9, random warm-up, 6 h window, 1.5 h blocks, uniform Weibull failures at availability 0.90), run on the
frozen v5 code (results/rq2-rebase-v5/code/env, 6d13412a; here code/env, the same plus preempt-resume). Only hetero.params' due-date range changes: (1.75 S, 2.5 S).
Late share = jobs whose due date falls in the window [t0, t1] that exit after it or are still open at t1, over all
jobs due in the window. Modes:
  calib S          seeds 1000-1029, MDD-PTECT (quad, alpha 3): m50-none at S = 1 and m50-unif at scale S -> out/calib/
  pick             S* by linear interpolation where the failure late share equals the failure-free one -> cscale.json
  compare SET K    test seeds 10K..10K+9, the 6 v5 pairs + MDD-TECT; SET = none | unif | unif-cal (S*) -> out/compare/
  list-calib, list-compare   task lines for the cluster arrays
  check-restart    calib seeds 1000-1001 at S = 1 with failure_restart: must equal rq4-mfail-ccal's out/calib/s1.json
                   exactly (the patch changes nothing else), and the same seeds under resume for the size of the change
@par Usage
@code{.sh}
python results/rq4-mfail-ccal-resume/mfail.py calib 1.5
@endcode
"""
import dataclasses
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
os.environ.setdefault("PECT_CODE", str(HERE / "code" / "env"))
os.environ.setdefault("PECT_MAIN", str(REPO))
PECT = REPO / "results" / "dev-pect-rule"
_spec = importlib.util.spec_from_file_location("pect_harness", PECT / "harness.py")
hz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hz)
H = hz.H
assert os.environ["PECT_CODE"] in hz.des_twin.__file__, hz.des_twin.__file__

SCALES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
CALIB_SEEDS = list(range(1000, 1030))
RULES = ["SRT_PECT:quad:3", "SRT_PTECT:quad:3", "MDD_PECT:quad:3", "MDD_PTECT:quad:3", "ATC_PECT:quad:3",
         "ATC_PTECT:quad:3", "MDD_TECT"]
REF = "MDD_PTECT:quad:3"
_orig_params = H.params


def set_scale(s):
    def params(*a, **k):
        p = _orig_params(*a, **k)
        return dataclasses.replace(p, due_date_allowance_range=(1.75 * s, 2.5 * s))
    H.params = params
    H._scen.clear()
    hz._inst.clear()


def run(cond, seed, variant):
    d = hz.instance(cond, seed)
    halves, extra = hz.variant_config(variant, seed, d)
    tw, d = hz.make_twin(cond, seed, extra)
    gen = tw.agent_decisions()
    t0 = None
    try:
        gen.send(None)
        while True:
            if t0 is None:
                t0 = tw.now
            gen.send(halves)
    except StopIteration:
        pass
    t0 = d["warmup"] if t0 is None else t0
    t1 = tw.now
    due_in = [j for j in tw.jobs.values() if j.due is not None and t0 <= j.due < t1]
    late = sum(1 for j in due_in if j.exit_time is None or j.exit_time > j.due)
    return {"tard": hz.window_tardiness(tw, t0, t1), "late": late, "due_in": len(due_in)}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def calib(s):
    out = HERE / "out" / "calib" / f"s{s:g}.json"
    if out.exists():
        return f"exists {out}"
    t = time.time()
    set_scale(s)
    res = {"scale": s, "unif": {seed: run("m50-unif", seed, REF) for seed in CALIB_SEEDS}}
    if s == 1.0:
        res["none"] = {seed: run("m50-none", seed, REF) for seed in CALIB_SEEDS}
    write(out, res)
    sh = lambda r: sum(v["late"] for v in r.values()) / sum(v["due_in"] for v in r.values())   # noqa: E731
    msg = f"calib S={s:g}: late share with failures {sh(res['unif']):.3f}"
    if "none" in res:
        msg += f", without {sh(res['none']):.3f}"
    return msg + f" ({time.time() - t:.0f} s)"


def pick():
    rows = {float(json.loads(f.read_text())["scale"]): json.loads(f.read_text())
            for f in (HERE / "out" / "calib").glob("s*.json")}
    sh = lambda r: sum(v["late"] for v in r.values()) / sum(v["due_in"] for v in r.values())   # noqa: E731
    tard = lambda r: sum(v["tard"] for v in r.values()) / len(r)   # noqa: E731
    target = sh(rows[1.0]["none"])
    table = {s: {"late_share": sh(r["unif"]), "mean_tard": tard(r["unif"])} for s, r in sorted(rows.items())}
    xs = sorted(table)
    star = None
    for a, b in zip(xs, xs[1:]):
        la, lb = table[a]["late_share"], table[b]["late_share"]
        if (la - target) * (lb - target) <= 0 and la != lb:
            star = a + (target - la) * (b - a) / (lb - la)
            break
    res = {"target_late_share": target, "none_mean_tard": tard(rows[1.0]["none"]), "table": table, "s_star": star}
    write(HERE / "cscale.json", res)
    print(json.dumps(res, indent=1))
    if star is None:
        raise SystemExit("the failure late share does not cross the target over the scales run; extend SCALES")


def compare(setting, k):
    seeds = list(range(10 * k, 10 * k + 10))
    out = HERE / "out" / "compare" / f"{setting}_c{k}.json"
    if out.exists():
        return f"exists {out}"
    t = time.time()
    s = 1.0
    if setting == "unif-cal":
        s = json.loads((HERE / "cscale.json").read_text())["s_star"]
    set_scale(s)
    cond = "m50-none" if setting == "none" else "m50-unif"
    res = {"setting": setting, "scale": s, "runs": {str(seed): {v: run(cond, seed, v) for v in RULES} for seed in seeds}}
    write(out, res)
    return f"compare {setting} chunk {k} (S={s:.3f}): {len(seeds) * len(RULES)} runs, {time.time() - t:.0f} s"


def check_restart():
    """failure_restart reproduces rq4-mfail-ccal exactly on two calibration seeds; resume differs."""
    old = json.loads((REPO / "results" / "rq4-mfail-ccal" / "out" / "calib" / "s1.json").read_text())
    set_scale(1.0)
    orig = hz.make_twin
    out, ok = [], True
    for restart in (True, False):
        hz.make_twin = (lambda c, s, e, r=restart: orig(c, s, {**e, "failure_restart": True} if r else e))
        for seed in (1000, 1001):
            got = run("m50-unif", seed, REF)
            ref = old["unif"][str(seed)]
            same = got == ref
            ok &= same or not restart
            out.append(f"{'restart' if restart else 'resume '} seed {seed}: tard {got['tard']:.3f} ks late {got['late']}"
                       f"/{got['due_in']} | rq4-mfail-ccal {ref['tard']:.3f} late {ref['late']} -> "
                       f"{'IDENTICAL' if same else 'differs'}")
    hz.make_twin = orig
    return "\n".join(out + [f"check-restart {'PASS' if ok else 'FAIL'}"])


def main(argv):
    if argv[0] == "list-calib":
        for s in SCALES:
            print("calib", s)
    elif argv[0] == "list-compare":
        for st in ("none", "unif", "unif-cal"):
            for k in range(4):
                print("compare", st, k)
    elif argv[0] == "calib":
        print(calib(float(argv[1])), flush=True)
    elif argv[0] == "pick":
        pick()
    elif argv[0] == "compare":
        print(compare(argv[1], int(argv[2])), flush=True)
    elif argv[0] == "check":
        # failure-free: must equal dev-pect-rule; with failures the restart value 172.84 is not expected any more
        set_scale(1.0)
        for cond, want in (("m50-none", 57.74296801405038), ("m50-unif", 172.8389618594804)):
            t = time.time()
            r = run(cond, 0, REF)
            verdict = ("OK" if abs(r["tard"] - want) < 1e-3 else "MISMATCH") if cond == "m50-none" else "resume"
            print(f"check {cond} seed 0 MDD_PTECT:quad:3: {r['tard']:.6f} (dev-pect-rule {want:.6f}, restart) "
                  f"{verdict}; late share {r['late']}/{r['due_in']} ({time.time() - t:.0f} s)")
            if verdict == "MISMATCH":
                sys.exit(1)
    elif argv[0] == "check-restart":
        print(check_restart(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
