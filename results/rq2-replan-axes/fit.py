"""
@file fit.py
@brief rq2-replan-axes, §8's fit on axes 1, 2 and 6 as README §17 sets it out (approved by the user 11-09 as proposed).
       A censored normal (Tobit) regression of log2(Δ* / p̄) over the 35 distinct conditions (rule 9 upper bounds
       left-censored) becomes a re-plan rule Δ_rec = p̄ 2^(xᵀβ − h). Its margin h is the smallest that keeps >= 85% of
       the best gain at >= 90% of the training conditions. The rule is validated condition-out (each condition predicted
       from the other 34) by coverage (held-out conditions that keep >= 85%) and compute (60 s / Δ_rec), under a joint
       seed bootstrap, and read by F1-F3.
@par Usage (the lab entry rq2-replan-axes-fit runs launch_fit.sh; every step skips what is already written)
@code{.sh}
.venv/bin/python results/rq2-replan-axes/fit.py table        # per-seed table of the 35 conditions -> fit_out/table.pkl
.venv/bin/python results/rq2-replan-axes/fit.py tasks        # unfinished tasks (var-00..01, read-00..19, point)
.venv/bin/python results/rq2-replan-axes/fit.py run <task>   # one task -> fit_out/<task>.json
.venv/bin/python results/rq2-replan-axes/fit.py report       # fit.out and fit_loco.csv; prints the readings
.venv/bin/python results/rq2-replan-axes/fit.py test         # tests on synthetic tables (reads no outputs)
@endcode
"""
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402

OUT = Path(os.environ.get("FIT_OUT", HERE / "fit_out"))  # FIT_OUT: a scratch folder (tests of the launcher)
X_BAND = A.X_PRIMARY        # README §3: a rule keeps a condition at >= 1 - X of its best gain (X 15%)
TARGET = 0.90               # margin: >= 90% of the training conditions keep (§17 decision 3)
PASS_N, N_COND = 30, 35     # pass line: >= 30 of the 35 held-out conditions keep
C_MAX = 0.5                 # F3: at most half the compute of re-planning every 60 s
D1_MIN, D1_MAX = 60.0, 3600.0
H_STEP = 0.01               # margin grid (octaves)
SIGMA_MIN = 0.01            # numerical floor on σ (octaves); fits where it binds are counted
B_BOUND = 50.0              # bound on a standardized coefficient (octaves per SD); fits that reach it are counted
B_READ, B_VAR, CHUNK = 2000, 200, 100
STREAM = 1709               # the fit's own bootstrap stream (axes 1, 2 and 6 keep theirs)
N_SEEDS = 40

COVS = ("tau_p", "treg_p", "c", "agv", "u", "surge", "short")
READ = {"M0": (), "M1": ("m",), "M2": ("m",) + COVS}
VAR = {**{f"M1 + {c}": ("m", c) for c in COVS},
       "M2 without tau_p": tuple(t for t in READ["M2"] if t != "tau_p"),
       "M2 without agv": tuple(t for t in READ["M2"] if t != "agv"),
       "M3": READ["M2"] + ("u*c", "u*treg_p")}
TERM = {"m": "log2(m/15)", "tau_p": "log2 τ/p̄", "treg_p": "log2 T_reg/p̄", "c": "log2 c", "agv": "AGV on duty",
        "u": "u", "surge": "surge share", "short": "short share", "u*c": "u × log2 c", "u*treg_p": "u × log2 T_reg/p̄"}


def task_names():
    return ([f"var-{i:02d}" for i in range(B_VAR // CHUNK)] + [f"read-{i:02d}" for i in range(B_READ // CHUNK)]
            + ["point"])


def resamples():
    return np.random.default_rng(STREAM).integers(0, N_SEEDS, (B_READ, N_SEEDS))


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ── The table: per-seed arrays of the 35 conditions ──────────────────────────────────────────────────────────────────

def build_table():
    """Per condition: the planner's tardiness per D1 and seed, the plan-once baseline, and the per-seed covariates
    (fit_inputs.py's, README §17)."""
    import fit_inputs as FI
    data, base = A.load(HERE)
    conds = []
    for key in FI.distinct_keys(data, base):
        arm, k = key
        by_d = data[key]
        ds = sorted(by_d)
        seeds = sorted(set.intersection(*(set(by_d[d]) for d in ds)) & set(base[key]))
        assert seeds == list(range(N_SEEDS)), (key, seeds)
        descs = [base[key][s]["describe"] for s in seeds]
        terms = [FI.instance_terms(arm, k, s, descs[s]) for s in seeds]

        def col(f):
            return np.array([f(i) for i in seeds], float)

        conds.append({
            "key": repr(key), "label": A.label(*key), "scale": float(k) if arm == "proc" else 1.0,
            "m": FI.machines(arm, k), "ds": [float(d) for d in ds],
            "real": np.array([[by_d[d][s]["real"] for s in seeds] for d in ds], float),
            "ref": col(lambda s: base[key][s]["best"]),
            "p": col(lambda s: descs[s]["p_mean_plan"]), "p_fast": col(lambda s: descs[s]["p_fast_plan"]),
            "tau": col(lambda s: descs[s]["transport_mean"]), "block": col(lambda s: descs[s]["block_seconds"]),
            "agv": col(lambda s: terms[s]["agv_recorded"] if terms[s]["agv_recorded"] is not None
                       else terms[s]["agv_rebuilt"]),
            "u": col(lambda s: descs[s]["machine_busy_share"]), "c": col(lambda s: terms[s]["c"]),
            "surge": col(lambda s: terms[s]["surge"]), "short": col(lambda s: terms[s]["short"]),
            "tc": col(lambda s: by_d[ds[0]][s]["plan_seconds"] / by_d[ds[0]][s]["replans"])})
    assert len(conds) == N_COND, len(conds)
    return {"conds": conds, "built": time.strftime("%Y-%m-%d %H:%M"), "analysis_out_md5": md5(HERE / "analysis.out")}


# ── One resample: curves, Δ*, flags and covariates ───────────────────────────────────────────────────────────────────

def rows_of(table, idx, x=X_BAND, pdef="p"):
    """Per condition under the seed resample idx: pooled gains, Δ* (D1) at band x, y = log2(Δ* / p̄), the censoring flag
    (0 measured, -1 upper bound by rule 9, +1 right-censored by rule 6, 2 both: uninformative) and the covariates."""
    rows = []
    for c in table["conds"]:
        g = -100.0 * (c["real"][:, idx].sum(1) / c["ref"][idx].sum() - 1.0)
        d = A.dstar(c["ds"], list(g), x)
        p = float(np.mean(c[pdef][idx]))
        r = {"label": c["label"], "ds": np.array(c["ds"]), "g": g, "scale": c["scale"], "p": p, "flat": d["flat"],
             "m": c["m"], "tau_p": float(np.mean(c["tau"][idx])) / p, "treg_p": float(np.mean(c["block"][idx])) / p,
             **{k: float(np.mean(c[k][idx])) for k in ("c", "agv", "u", "surge", "short", "tc")}}
        if not d["flat"]:
            upper, right = d["best_d"] == c["ds"][0], bool(d["cens_right"])
            r.update(dstar_d1=d["dstar"], y=math.log2(d["dstar"] * c["scale"] / p),
                     cens=2 if upper and right else (-1 if upper else (1 if right else 0)))
        rows.append(r)
    return rows


def design(rows, terms):
    cols = [np.ones(len(rows))]
    for t in terms:
        if t == "m":
            v = [math.log2(r["m"] / 15.0) for r in rows]
        elif t in ("tau_p", "treg_p", "c"):
            v = [math.log2(r[t]) for r in rows]
        elif t == "u*c":
            v = [r["u"] * math.log2(r["c"]) for r in rows]
        elif t == "u*treg_p":
            v = [r["u"] * math.log2(r["treg_p"]) for r in rows]
        else:
            v = [r[t] for r in rows]
        cols.append(np.array(v, float))
    return np.column_stack(cols)


# ── Tobit ────────────────────────────────────────────────────────────────────────────────────────────────────────────

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def tobit_nll(th, Z, y, cens):
    """Negative log-likelihood and gradient in θ = (b, log σ); cens 0 exact, -1 y_true <= y, +1 y_true >= y."""
    b, ls = th[:-1], th[-1]
    s = math.exp(ls)
    z = (y - Z @ b) / s
    gmu = np.zeros(len(y))
    f, gls = 0.0, 0.0
    ex = cens == 0
    if ex.any():
        f += 0.5 * float(np.sum(z[ex] ** 2)) + ex.sum() * (ls + LOG_SQRT_2PI)
        gmu[ex] = -z[ex] / s
        gls += float(np.sum(1.0 - z[ex] ** 2))
    for side in (-1, 1):
        m = cens == side
        if m.any():
            a = z[m] if side == -1 else -z[m]          # P(y_true <= y) = Φ(z); P(y_true >= y) = Φ(-z)
            lp = log_ndtr(a)
            lam = np.exp(-0.5 * a * a - LOG_SQRT_2PI - lp)
            f -= float(np.sum(lp))
            gmu[m] = (lam if side == -1 else -lam) / s
            gls += float(np.sum(lam * a))
    return f, np.concatenate([Z.T @ gmu, [gls]])


def tobit(X, y, cens):
    """Censored normal regression by maximum likelihood (standardized columns, σ >= SIGMA_MIN, |b| <= B_BOUND per SD).
    Uninformative rows (cens 2) are left out. Returns beta in the columns' own units."""
    keep = cens != 2
    X, y, cens = X[keep], y[keep], cens[keep]
    mx, sx = X[:, 1:].mean(0), X[:, 1:].std(0)
    live = sx > 1e-12
    Z = np.column_stack([np.ones(len(X)), (X[:, 1:][:, live] - mx[live]) / sx[live]])
    b0 = np.linalg.lstsq(Z, y, rcond=None)[0]
    s0 = max(0.3, float(np.std(y - Z @ b0)))
    th0 = np.concatenate([np.clip(b0, -B_BOUND, B_BOUND), [math.log(s0)]])
    bounds = [(None, None)] + [(-B_BOUND, B_BOUND)] * (Z.shape[1] - 1) + [(math.log(SIGMA_MIN), math.log(50.0))]
    res = minimize(tobit_nll, th0, args=(Z, y, cens), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    if not res.success:                              # a line-search stop at the optimum: restart once from there
        res = minimize(tobit_nll, res.x, args=(Z, y, cens), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    _, grad = tobit_nll(res.x, Z, y, cens)
    lo_b = np.array([-np.inf if lb is None else lb for lb, _ in bounds])
    hi_b = np.array([np.inf if hb is None else hb for _, hb in bounds])
    pg = np.where((res.x <= lo_b + 1e-9) & (grad > 0) | (res.x >= hi_b - 1e-9) & (grad < 0), 0.0, grad)
    converged = bool(res.success) or float(np.max(np.abs(pg))) < 1e-4
    b, ls = res.x[:-1], res.x[-1]
    beta = np.zeros(X.shape[1])
    beta[1:][live] = b[1:] / sx[live]
    beta[0] = b[0] - float(np.sum(b[1:] * mx[live] / sx[live]))
    return {"beta": beta, "sigma": math.exp(ls), "ok": converged, "nll": float(res.fun),
            "floor": bool(ls <= math.log(SIGMA_MIN) + 1e-6), "edge": bool(np.any(np.abs(b[1:]) >= B_BOUND - 1e-6))}


# ── From a fit to a rule, and its score ──────────────────────────────────────────────────────────────────────────────

def kept(r, d1):
    """Kept share of the best gain at B2-equivalent interval(s) d1, the gain linear in log D1 between grid points."""
    return np.interp(np.log(d1), np.log(r["ds"]), r["g"]) / float(np.max(r["g"]))


def d1_rec(r, mu, h):
    """The rule's interval Δ_rec = p̄ 2^(mu - h) in B2-equivalent D1, clipped to the grid."""
    return np.clip(r["p"] * 2.0 ** (mu - np.asarray(h)) / r["scale"], D1_MIN, D1_MAX)


def tune_h(rows, mus, keep):
    """The smallest margin h >= 0 (H_STEP grid) at which >= TARGET of `rows` keep >= `keep` at their own Δ_rec. At the
    grid's top every row sits at D1 60 s; if even that misses the target, h is the top."""
    top = max(0.0, max(mu - math.log2(D1_MIN * r["scale"] / r["p"]) for r, mu in zip(rows, mus)))
    hs = np.arange(0.0, top + 1.5 * H_STEP, H_STEP)
    n_ok = np.zeros(len(hs))
    for r, mu in zip(rows, mus):
        n_ok += kept(r, d1_rec(r, mu, hs)) >= keep
    hit = np.nonzero(n_ok >= TARGET * len(rows) - 1e-9)[0]
    return float(hs[hit[0]] if len(hit) else hs[-1])


def fit_rows(rows, terms):
    X = design(rows, terms)
    return X, np.array([r["y"] for r in rows]), np.array([r["cens"] for r in rows])


def evaluate(rows, terms, keep, detail=False):
    """Full fit and condition-out validation of one model on the non-flat rows."""
    use = [r for r in rows if not r["flat"]]
    X, y, cens = fit_rows(use, terms)
    full = tobit(X, y, cens)
    h_full = tune_h(use, X @ full["beta"], keep)
    out = {"beta": full["beta"].tolist(), "sigma": full["sigma"], "h": h_full, "n": len(use),
           "fits": 1, "fail": int(not full["ok"]), "floor": int(full["floor"]), "edge": int(full["edge"])}
    per = []
    for i in range(len(use)):
        tr = np.arange(len(use)) != i
        f = tobit(X[tr], y[tr], cens[tr])
        h = tune_h([u for j, u in enumerate(use) if j != i], X[tr] @ f["beta"], keep)
        mu = float(X[i] @ f["beta"])
        d1 = float(d1_rec(use[i], mu, h))
        k = float(kept(use[i], d1))
        per.append({"label": use[i]["label"], "mu": mu, "h": h, "d1": d1, "kept": k, "keeps": bool(k >= keep - 1e-12),
                    "C": D1_MIN / d1})
        out["fits"] += 1
        out["fail"] += int(not f["ok"])
        out["floor"] += int(f["floor"])
        out["edge"] += int(f["edge"])
    out["cover"] = int(sum(p["keeps"] for p in per))
    out["C"] = float(np.mean([p["C"] for p in per]))
    if detail:
        out["loco"] = per
        out["full_d1"] = [float(d1_rec(u, float(xu @ full["beta"]), h_full)) for u, xu in zip(use, X)]
        out["labels"] = [u["label"] for u in use]
    return out


# ── Tasks ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def point_task(table):
    """All 40 seeds: every model in detail, the reported quantities, the sensitivities."""
    ident = np.arange(N_SEEDS)
    rows = rows_of(table, ident)
    assert not any(r["flat"] for r in rows), "a condition is flat at the point estimate"
    out = {"models": {name: evaluate(rows, terms, 1.0 - X_BAND, detail=True)
                      for name, terms in {**READ, **VAR}.items()}}
    out["conditions"] = [{"label": r["label"], "y": r["y"], "cens": r["cens"], "dstar_d1": r["dstar_d1"],
                          "p": r["p"], "scale": r["scale"], "tc": r["tc"],
                          "kept_2x": float(kept(r, min(D1_MAX, 2.0 * r["dstar_d1"]))),
                          "width_lb": r["dstar_d1"] / D1_MIN,
                          "kept_grid": [float(kept(r, d)) for d in r["ds"]], "ds": r["ds"].tolist()} for r in rows]
    sens = {}
    for name, x in (("X 10%", 0.10), ("X 20%", 0.20)):
        rr = rows_of(table, ident, x=x)
        sens[name] = {m: evaluate(rr, t, 1.0 - x) for m, t in READ.items()} if not any(r["flat"] for r in rr) else None
    rr = [dict(r, cens=0) for r in rows]
    sens["bounds as exact"] = {m: evaluate(rr, t, 1.0 - X_BAND) for m, t in READ.items()}
    rr = rows_of(table, ident, pdef="p_fast")
    sens["p̄ fastest machine"] = {m: evaluate(rr, t, 1.0 - X_BAND) for m, t in READ.items()}
    out["sensitivity"] = sens
    return out


def boot_task(table, models, rows_idx):
    idx = resamples()
    out = []
    for b in rows_idx:
        rows = rows_of(table, idx[b])
        out.append({"b": int(b), **{name: evaluate(rows, terms, 1.0 - X_BAND) for name, terms in models.items()}})
    return {"resamples": out}


def run_task(task, out_dir=OUT):
    path = out_dir / f"{task}.json"
    if path.exists():
        print(f"{task}: done already, skipped")
        return
    t0 = time.time()
    table = pickle.loads((out_dir / "table.pkl").read_bytes())
    if task == "point":
        res = point_task(table)
    else:
        kind, i = task.split("-")
        rng = range(int(i) * CHUNK, (int(i) + 1) * CHUNK)
        res = boot_task(table, READ if kind == "read" else VAR, rng)
    res["stamp"] = {"task": task, "fit_py_md5": md5(__file__), "table_md5": md5(out_dir / "table.pkl"),
                    "seconds": time.time() - t0, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, path)
    print(f"{task}: {time.time() - t0:.0f} s")


def write_table(out_dir=OUT, table=None):
    path = out_dir / "table.pkl"
    if path.exists():
        print("table: done already, skipped")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_table() if table is None else table
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(table))
    os.replace(tmp, path)
    print(f"table: {len(table['conds'])} conditions")


# ── Report ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def report_dir(out_dir):
    """fit.out and fit_loco.csv go beside the README for the real run, into the scratch folder otherwise."""
    return HERE if Path(out_dir).resolve() == (HERE / "fit_out").resolve() else Path(out_dir)


def q90(v):
    v = np.asarray(v, float)
    return float(np.percentile(v, 5)), float(np.percentile(v, 95))


def report(out_dir=OUT, n_read=B_READ, n_var=B_VAR, n_cond=N_COND, write=True, label="lab rq2-replan-axes-fit"):
    """fit.out and fit_loco.csv from every task output; returns the summary lines (the readings) for stdout."""
    def load(name):
        return json.loads((out_dir / f"{name}.json").read_text())

    names = [n for n in task_names() if (out_dir / f"{n}.json").exists()]
    point = load("point")
    read = [r for n in names if n.startswith("read") for r in load(n)["resamples"]][:n_read]
    var = [r for n in names if n.startswith("var") for r in load(n)["resamples"]][:n_var]
    assert len(read) == n_read and len(var) == n_var, (len(read), len(var))
    fit_md5 = {load(n)["stamp"]["fit_py_md5"] for n in names}
    assert len(fit_md5) == 1, "task outputs come from different fit.py versions"
    assert len({load(n)["stamp"]["table_md5"] for n in names}) == 1, "task outputs come from different tables"
    pm = point["models"]
    line_frac = PASS_N / N_COND
    L = []
    pr = L.append
    pr(f"§8's fit on axes 1, 2 and 6 (README §17), {label}: {n_cond} conditions, "
       f"{sum(c['cens'] == -1 for c in point['conditions'])} upper bounds (rule 9), "
       f"{sum(c['cens'] == 1 for c in point['conditions'])} right-censored; X {X_BAND:.0%}, margin target "
       f"{TARGET:.0%}, pass line {PASS_N} of {N_COND}; bootstrap {n_read} resamples (M0-M2), {n_var} (variants), "
       f"stream {STREAM}; fit.py md5 {next(iter(fit_md5))[:8]}")
    diag = {k: sum(m[k] for m in pm.values()) for k in ("fits", "fail", "floor", "edge")}
    bdiag = {k: sum(r[m][k] for r in read + var for m in r if m != "b") for k in ("fits", "fail", "floor", "edge")}
    pr(f"   fits at the point estimate: {diag['fits']}, not converged {diag['fail']}, σ at its floor {diag['floor']}, "
       f"a coefficient at its bound {diag['edge']}; in the bootstrap: {bdiag['fits']}, {bdiag['fail']}, "
       f"{bdiag['floor']}, {bdiag['edge']}")

    # (1) coefficients
    pr("")
    pr("(1) coefficients of log2(Δ* / p̄) (per octave of a ratio, per unit of a share; full fit on all conditions), "
       "90% bootstrap intervals; σ = residual scatter (octaves); h = the margin of the rule")
    for name, terms in {**READ, **VAR}.items():
        boot = read if name in READ else var
        bs = np.array([r[name]["beta"] for r in boot])
        ss = [r[name]["sigma"] for r in boot]
        cells = [f"{'intercept' if j == 0 else TERM[terms[j - 1]]} {pm[name]['beta'][j]:+.2f} "
                 f"[{q90(bs[:, j])[0]:+.2f}, {q90(bs[:, j])[1]:+.2f}]" for j in range(len(terms) + 1)]
        pr(f"   {name:18s} σ {pm[name]['sigma']:.2f} [{q90(ss)[0]:.2f}, {q90(ss)[1]:.2f}], h {pm[name]['h']:.2f}: "
           + "; ".join(cells))

    # (2) condition-out validation
    pr("")
    pr(f"(2) condition-out: each condition predicted from the other {n_cond - 1}; coverage = held-out conditions keeping "
       f">= {1 - X_BAND:.0%} of the best gain; compute C = mean of 60 s / Δ_rec (1 = always D1 60 s)")
    for name in {**READ, **VAR}:
        boot = read if name in READ else var
        cs = [r[name]["C"] for r in boot]
        cov = [r[name]["cover"] / r[name]["n"] for r in boot]
        hs = [p["h"] for p in pm[name]["loco"]]
        pr(f"   {name:18s} coverage {pm[name]['cover']:2d} of {pm[name]['n']} [{q90(cov)[0]:.0%}, {q90(cov)[1]:.0%}]   "
           f"C {pm[name]['C']:.3f} [{q90(cs)[0]:.3f}, {q90(cs)[1]:.3f}]   margin h {min(hs):.2f}-{max(hs):.2f}")
    conds = point["conditions"]
    pr("   reference, untuned and in sample (fit_inputs.out (4)): constant D1 "
       + "; ".join(f"{d:g} s keeps {sum(c['kept_grid'][c['ds'].index(d)] >= 1 - X_BAND - 1e-12 for c in conds)} "
                   f"of {len(conds)} at C {D1_MIN / d:.2f}" for d in (60.0, 120.0, 180.0, 300.0)))

    # (3) readings
    def step(a, b):
        diff = [r[a]["C"] - r[b]["C"] for r in read]
        lo, hi = q90(diff)
        ok = hi < 0 and pm[a]["cover"] >= PASS_N
        share = float(np.mean([(r[a]["C"] < r[b]["C"]) and (r[a]["cover"] >= line_frac * r[a]["n"]) for r in read]))
        return ok, (f"C({a}) - C({b}) {pm[a]['C'] - pm[b]['C']:+.3f} [90% {lo:+.3f}, {hi:+.3f}], coverage({a}) "
                    f"{pm[a]['cover']} of {pm[a]['n']} (line {PASS_N}); in {share:.0%} of resamples {a} beats {b} at "
                    f"the line")

    f1, t1 = step("M1", "M0")
    ref2 = "M1" if f1 else "M0"
    f2, t2 = step("M2", ref2)
    sel = "M2" if f2 else ("M1" if f1 else "M0")
    f3 = pm[sel]["cover"] >= PASS_N and pm[sel]["C"] <= C_MAX
    share3 = float(np.mean([(r[sel]["cover"] >= line_frac * r[sel]["n"]) and (r[sel]["C"] <= C_MAX) for r in read]))
    S = [f"F1 (the size law carries condition-out, M1 vs M0): {'PASS' if f1 else 'FAIL'}. {t1}",
         f"F2 (the covariates carry condition-out, M2 vs {ref2}): {'PASS' if f2 else 'FAIL'}. {t2}",
         f"F3 (coarser than the finest interval is safe; selected rule {sel}): {'PASS' if f3 else 'FAIL'}. coverage "
         f"{pm[sel]['cover']} of {pm[sel]['n']} (line {PASS_N}), C {pm[sel]['C']:.3f} (line {C_MAX}); both hold in "
         f"{share3:.0%} of resamples"]
    pr("")
    pr("(3) READINGS (pre-declared, README §17)")
    for s in S:
        pr("   " + s)

    # (4) reported
    pr("")
    pr("(4) reported beside them")
    k2 = [c["kept_2x"] for c in conds]
    pr(f"   overshoot: kept share at 2 Δ* median {np.median(k2):.1%} (range {min(k2):.1%}-{max(k2):.1%}); plateau "
       f"width lower bound Δ* / 60 s median {np.median([c['width_lb'] for c in conds]):.2f} "
       f"(range {min(c['width_lb'] for c in conds):.2f}-{max(c['width_lb'] for c in conds):.2f})")
    fr = [(c["label"], c["tc"] / (d1 * c["scale"])) for c, d1 in zip(conds, pm[sel]["full_d1"])]
    worst = sorted(fr, key=lambda t: -t[1])[:3]
    pr(f"   real time, selected rule {sel} fitted on all conditions: T_c / Δ_rec (serial re-plan) median "
       f"{np.median([t[1] for t in fr]):.3f}; highest " + ", ".join(f"{a} {b:.2f}" for a, b in worst))
    for sname, res in point["sensitivity"].items():
        if res is None:
            pr(f"   sensitivity {sname}: a condition is flat, not evaluated")
            continue
        pr(f"   sensitivity {sname}: " + "; ".join(f"{m} coverage {v['cover']} of {v['n']}, C {v['C']:.3f}"
                                                 for m, v in res.items()))
    if write:
        dest = report_dir(out_dir)
        (dest / "fit.out").write_text("\n".join(L) + "\n")
        with open(dest / "fit_loco.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "condition", "y", "cens", "mu_loco", "h_loco", "d1_rec", "kept", "keeps", "C",
                        "d1_rec_full"])
            for name, m in pm.items():
                for p, c, fd in zip(m["loco"], conds, m["full_d1"]):
                    w.writerow([name, p["label"], f"{c['y']:.4f}", c["cens"], f"{p['mu']:.4f}", f"{p['h']:.2f}",
                                f"{p['d1']:.1f}", f"{p['kept']:.4f}", int(p["keeps"]), f"{p['C']:.4f}", f"{fd:.1f}"])
    return S, L


# ── Tests (synthetic tables only) ────────────────────────────────────────────────────────────────────────────────────

def synthetic_table(seed=0, n=N_COND, slope_m=-0.5, noise=0.25):
    """A table shaped like the real one: 35 conditions, the 9-point grid, 40 seeds. True log2(Δ* / p̄) = 0.2 + slope_m
    log2(m / 15) + noise; gain g(Δ) = G (1 - 0.15 (Δ / Δ*)^0.8), with a rise to an interior best at a third of them."""
    rng = np.random.default_rng(seed)
    ds = [60.0, 120.0, 180.0, 300.0, 450.0, 600.0, 900.0, 1800.0, 3600.0]
    conds = []
    for i in range(n):
        m = [15, 15, 15, 20, 30, 50, 100][i % 7]
        scale = [1.0, 1.0, 1.0, 2.0][i % 4]
        p = 330.0 * scale
        y = 0.2 + slope_m * math.log2(m / 15) + noise * rng.standard_normal()
        dstar = p * 2.0 ** y
        G = rng.uniform(4, 15)
        dd = np.array(ds) * scale
        g = G * (1.0 - 0.15 * (dd / dstar) ** 0.8)
        if i % 3 == 0:
            g = g - G * 0.04 * np.maximum(0.0, np.log2(2.5 * 60 * scale / dd))
        ref = 100.0 * np.exp(0.3 * rng.standard_normal(N_SEEDS))
        real = ref[None, :] * (1.0 - g[:, None] / 100.0) * (1.0 + 0.02 * rng.standard_normal((len(ds), N_SEEDS)))
        conds.append({"key": f"syn{i}", "label": f"syn {i} m{m} k{scale:g}", "scale": scale, "m": m, "ds": ds,
                      "real": real, "ref": ref, "p": np.full(N_SEEDS, p), "p_fast": np.full(N_SEEDS, 0.85 * p),
                      "tau": np.full(N_SEEDS, 30.0 + m), "block": np.full(N_SEEDS, 5400.0 * scale),
                      "agv": np.clip(rng.uniform(0.2, 0.8) + 0.02 * rng.standard_normal(N_SEEDS), 0.01, 0.99),
                      "u": np.clip(rng.uniform(0.5, 0.8) + 0.01 * rng.standard_normal(N_SEEDS), 0.01, 0.99),
                      "c": np.full(N_SEEDS, rng.choice([1.5, 2.125, 3.0])),
                      "surge": np.clip(0.25 + 0.05 * rng.standard_normal(N_SEEDS), 0, 1),
                      "short": np.clip(0.25 + 0.05 * rng.standard_normal(N_SEEDS), 0, 1),
                      "tc": np.full(N_SEEDS, 5.0 * m / 15)})
    return {"conds": conds, "built": "synthetic", "analysis_out_md5": "synthetic"}


def test():
    rng = np.random.default_rng(3)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"   {'ok  ' if cond else 'FAIL'} {name} {detail}")

    print("fit.py tests (synthetic data only; no task output is read)")
    # 1. no censoring = OLS (σ the ML estimate)
    n = 200
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    y = X @ np.array([0.5, -0.4, 0.2, 1.0]) + 0.6 * rng.standard_normal(n)
    f = tobit(X, y, np.zeros(n, int))
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    check("uncensored fit equals OLS", np.allclose(f["beta"], b, atol=1e-5)
          and abs(f["sigma"] - math.sqrt(np.mean((y - X @ b) ** 2))) < 1e-5,
          f"max |Δβ| {np.max(np.abs(f['beta'] - b)):.1e}")
    # 2. analytic gradient
    cens = rng.choice([0, -1, 1], n)
    th = np.concatenate([rng.standard_normal(4) * 0.3, [math.log(0.7)]])
    _, gr = tobit_nll(th, X, y, cens)
    num = np.array([(tobit_nll(th + e, X, y, cens)[0] - tobit_nll(th - e, X, y, cens)[0]) / 2e-6
                    for e in np.eye(len(th)) * 1e-6])
    check("gradient matches finite differences", np.allclose(gr, num, rtol=1e-4, atol=1e-4),
          f"max diff {np.max(np.abs(gr - num)):.1e}")
    # 3. recovery under ~70% left-censoring (observed as an upper bound when y_true <= its threshold)
    n = 4000
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2)), rng.uniform(0, 1, n)])
    beta, sig = np.array([0.0, -0.5, 0.3, 0.8]), 0.5
    yt = X @ beta + sig * rng.standard_normal(n)
    thr = X @ beta + 0.26 + 0.3 * rng.standard_normal(n)
    up = yt <= thr
    f = tobit(X, np.where(up, thr, yt), np.where(up, -1, 0))
    check(f"recovery with {up.mean():.0%} upper bounds", np.max(np.abs(f["beta"] - beta)) < 0.08
          and abs(f["sigma"] - sig) < 0.05, f"β {np.round(f['beta'], 3)}, σ {f['sigma']:.3f}")
    naive = np.linalg.lstsq(X, np.where(up, thr, yt), rcond=None)[0]
    check("  (bounds treated as exact are biased up)", naive[0] > beta[0] + 0.2, f"intercept {naive[0]:+.2f}")
    # 4. right-censoring
    lo = yt >= X @ beta + 0.2
    f = tobit(X, np.where(lo, X @ beta + 0.2, yt), np.where(lo, 1, 0))
    check(f"recovery with {lo.mean():.0%} lower bounds", np.max(np.abs(f["beta"] - beta)) < 0.08
          and abs(f["sigma"] - sig) < 0.05, f"β {np.round(f['beta'], 3)}, σ {f['sigma']:.3f}")
    # 5. kept share, rule and margin on a synthetic table
    tab = synthetic_table(1)
    rows = rows_of(tab, np.arange(N_SEEDS))
    r = rows[0]
    check("kept share at grid points = g / max g", np.allclose(kept(r, r["ds"]), r["g"] / r["g"].max()))
    i = 3 if rows[3]["scale"] == 2.0 else next(j for j, u in enumerate(rows) if u["scale"] == 2.0)
    u = rows[i]
    check("y uses Δ = k D1 in proc-like rows", abs(u["y"] - math.log2(u["dstar_d1"] * 2.0 / u["p"])) < 1e-12)
    check("Δ_rec at mu = y, h = 0 is Δ* (when inside the grid)",
          abs(float(d1_rec(u, u["y"], 0.0)) - min(D1_MAX, max(D1_MIN, u["dstar_d1"]))) < 1e-6)
    mus = [rr["y"] + 0.3 * rng.standard_normal() for rr in rows]
    h = tune_h(rows, mus, 0.85)
    frac = np.mean([kept(rr, d1_rec(rr, mu, h)) >= 0.85 for rr, mu in zip(rows, mus)])
    frac_less = np.mean([kept(rr, d1_rec(rr, mu, h - H_STEP)) >= 0.85 for rr, mu in zip(rows, mus)]) if h > 0 else 0
    check("margin reaches the target and is the smallest on its grid", frac >= TARGET and (h == 0 or frac_less < TARGET),
          f"h {h:.2f}, kept at {frac:.0%} (at h - step {frac_less:.0%})")
    check("synthetic table has upper bounds and measured rows",
          0 < sum(rr["cens"] == -1 for rr in rows) < len(rows), f"{sum(rr['cens'] == -1 for rr in rows)} bounds")
    # 6. end to end in a scratch folder: table, point, one read and one var chunk (3 + 2 resamples), report, resume
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        write_table(d, tab)
        res = {"point": point_task(tab), "read-00": boot_task(tab, READ, range(3)),
               "var-00": boot_task(tab, VAR, range(2))}
        for name, v in res.items():
            v["stamp"] = {"task": name, "fit_py_md5": md5(__file__), "table_md5": md5(d / "table.pkl")}
            (d / f"{name}.json").write_text(json.dumps(v))
        S, L = report(d, n_read=3, n_var=2, write=False, label="synthetic")
        b_m = res["point"]["models"]["M1"]["beta"][1]
        check("end to end: report prints F1-F3", len(S) == 3 and all(s.startswith(("F1", "F2", "F3")) for s in S))
        check("end to end: M1 finds the synthetic size slope (-0.5)", -1.0 < b_m < 0.0, f"{b_m:+.2f}")
        t0 = time.time()
        run_task("point", d)
        check("resume: a finished task is skipped", time.time() - t0 < 2.0)
        print("   synthetic report (synthetic data, not a reading):")
        for s in L:
            print("      " + s)
    print("all tests passed" if ok else "SOME TESTS FAILED")
    return ok


def main(argv):
    cmd = argv[0] if argv else "report"
    if cmd == "table":
        write_table()
    elif cmd == "tasks":
        for t in task_names():
            if not (OUT / f"{t}.json").exists():
                print(t)
    elif cmd == "run":
        run_task(argv[1])
    elif cmd == "report":
        S, _ = report()
        print("\n".join(S))
        print(f"(fit.out, fit_loco.csv written in {report_dir(OUT)})")
    elif cmd == "test":
        sys.exit(0 if test() else 1)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
