"""
@file mfail.py
@brief rq4-mfail-ccal (2026-10-14, user): machine failures on the 50-machine floor with due dates calibrated to the
       failures. With B2's c ~ U[1.75, 2.5] the failures triple tardiness (dev-hetero-tune / dev-pect-rule m50-unif:
       158 -> ~520 ks for MDD-TECT; MDD-PTECT 57.7 -> 172.8 on seed 0), so almost every job is late and due-date rules
       lose their leverage. Here the c range is scaled by S so that the share of late jobs under failures matches the
       failure-free floor, and the v5 rules are compared at no failures, failures at B2's c, and failures at the
       calibrated c.

@details Floor, instances and failures are dev-pect-rule's m50 family (harness.py / hetero.py there: 50 machines, 26
AGVs, load 0.9, random warm-up, 6 h window, 1.5 h blocks, uniform Weibull failures at availability 0.90), run on the
frozen v5 code (results/rq2-rebase-v5/code/env, 6d13412a). Only hetero.params' due-date range changes: (1.75 S, 2.5 S).
Late share = jobs whose due date falls in the window [t0, t1] that exit after it or are still open at t1, over all
jobs due in the window. Modes:
  calib S          seeds 1000-1029, MDD-PTECT (quad, alpha 3): m50-none at S = 1 and m50-unif at scale S -> out/calib/
  pick             S* by linear interpolation where the failure late share equals the failure-free one -> cscale.json
  compare SET K    test seeds 10K..10K+9, the 6 v5 pairs + MDD-TECT; SET = none | unif | unif-cal (S*) -> out/compare/
  list-calib, list-compare   task lines for the cluster arrays
@par Usage
@code{.sh}
python results/rq4-mfail-ccal/mfail.py calib 1.5
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
os.environ.setdefault("PECT_CODE", str(REPO / "results" / "rq2-rebase-v5" / "code" / "env"))
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
        set_scale(1.0)
        for cond, want in (("m50-none", 57.74296801405038), ("m50-unif", 172.8389618594804)):
            t = time.time()
            r = run(cond, 0, REF)
            print(f"check {cond} seed 0 MDD_PTECT:quad:3: {r['tard']:.6f} (dev-pect-rule {want:.6f}) "
                  f"{'OK' if abs(r['tard'] - want) < 1e-3 else 'MISMATCH'}; late share {r['late']}/{r['due_in']} "
                  f"({time.time() - t:.0f} s)")


if __name__ == "__main__":
    main(sys.argv[1:])
