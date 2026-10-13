"""
@file workup.py
@brief Bad-instance workup of the 400k slot policies (train-due-twin-slot-long): is there a scenario property that
       separates the instances where the policies lose to MDD-TECT ("bad") from the ones where they win ("good")?

Per instance it computes the policies' gap to MDD-TECT (label) and about 40 features of the generated scenario inside
the 6 h agent window: regime mix and where it falls, offered load and its peaks (per 900 s slot), fluid backlogs per
machine type and for transport (arriving work vs capacity), due-date tightness, op mix, carry-over from the warm-up.
Outcome-side features (fixed-pair results: MDD-TECT vs ATC-ECT, gap to own best pair, oracle gain) and the policy's
own slot choices are kept apart: they explain a loss but the generator cannot set them.

Two instance sets: seeds 0-39 (eval-due-twin-slot-long + -actions, fixed pairs + oracle from rq2-twin-fleet B2) are the
discovery set; seeds 1000-1199 (eval-due-twin-slot-long-ext: s0, s3, MDD-TECT, ATC-ECT) are the replication set. The
label on both is the mean gap of checkpoints s0 and s3 (the only two run on both sets); the 0-39 analysis also reports
the s0-s3 mean.

@par Usage
@code{.sh}
.venv/bin/python results/eval-due-twin-slot-long/workup/workup.py > results/eval-due-twin-slot-long/workup/workup.out
@endcode
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.stats import mannwhitneyu, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MAIN = REPO / "results/eval-due-twin-slot-long"
ACTIONS = REPO / "results/eval-due-twin-slot-long-actions"
EXT = REPO / "results/eval-due-twin-slot-long-ext"
TASKS = REPO / "results/rq2-twin-fleet/tasks"
WINDOW, SLOT, STEP = 21600.0, 900.0, 300.0
TYPES = ["Mill", "Lathe", "Weld", "Inspect", "Assemble"]
CK = "ckpt:train-due-twin-slot-long_s{}/checkpoint"
LABEL_SEEDS = (0, 3)

# Reference palette (dataviz skill, references/palette.md), light mode.
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10})


# ---------------------------------------------------------------------------------------------------------- features
def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def scenario_features(sc):
    """Scenario-side features inside the agent window [w0, w0 + 6 h]; everything here the generator controls."""
    w0 = float(sc["stochastic"].get("warmupSeconds") or 0.0)
    w1 = w0 + WINDOW
    p = sc["_meta"]["params"]
    blocks = sc["_meta"]["blocks"]
    k = p["machines_per_type"]
    f = {"warmup_h": w0 / 3600}

    # Regime blocks: shares, where they fall, how often the regime changes.
    third = WINDOW / 3
    for prof in ("normal", "surge", "short"):
        f[f"{prof}_share"] = sum(overlap(b["start"], b["end"], w0, w1) for b in blocks if b["load_profile"] == prof) / WINDOW
        for i, part in enumerate(("first", "mid", "last")):
            f[f"{prof}_{part}3"] = sum(overlap(b["start"], b["end"], w0 + i * third, w0 + (i + 1) * third)
                                       for b in blocks if b["load_profile"] == prof) / third
    inwin = [b for b in blocks if overlap(b["start"], b["end"], w0, w1) > 0]
    f["regime_changes"] = sum(a["load_profile"] != b["load_profile"] for a, b in zip(inwin, inwin[1:]))
    wts = np.array([overlap(b["start"], b["end"], w0, w1) for b in inwin]) / WINDOW
    f["c_mean"] = float(np.dot(wts, [b["due_date_allowance"] for b in inwin]))
    f["c_min"] = min(b["due_date_allowance"] for b in inwin)
    f["op_mean_block"] = float(np.dot(wts, [b["op_mean"] for b in inwin]))
    f["op_mean_spread"] = max(b["op_mean"] for b in inwin) / min(b["op_mean"] for b in inwin)
    f["pre_short"] = float(any(b["load_profile"] == "short" and b["end"] > w0 - 5400 and b["start"] < w0 for b in blocks))

    # Segments: nominal utilization and kinds, time-weighted over the window.
    ph = sc["_phases"]
    pw = np.array([overlap(x["start"], x["end"], w0, w1) for x in ph]) / WINDOW
    u = np.array([x["utilization"] for x in ph])
    f["util_nominal"] = float(np.dot(pw, u))
    f["util_nominal_max"] = float(u[pw > 0].max())
    for kind in ("balanced", "skewed", "bimodal", "lull", "burst"):
        f[f"seg_{kind}"] = float(sum(w for w, x in zip(pw, ph) if x["kind"] == kind))
    f["transport_capped"] = float(sum(w for w, x in zip(pw, ph) if x["transport_capped"]))

    # Jobs: realized work per type on a 300 s grid from t=0 (warm-up included), fluid backlogs.
    agv_at = lambda t: next((s["agvCount"] for s in reversed(sc.get("agvSchedule", [])) if s["start"] <= t), sc["agvCount"])
    grid = np.arange(0.0, w1 + STEP, STEP)
    n = len(grid) - 1
    work = np.zeros((len(TYPES), n))
    moves = np.zeros(n)
    jobs_in, ops_in, c_job, long_ops, ops_total = 0, [], [], 0, 0
    for j in sc["jobs"]:
        a = j["arrivalTime"]
        if a >= w1:
            continue
        i = min(int(a // STEP), n - 1)
        tw = 0.0
        for op in j["operations"]:
            d = float(np.mean(op["duration"]))
            work[TYPES.index(op["machineType"]), i] += d
            tw += min(op["duration"])
        moves[i] += len(j["operations"]) + 1
        if a >= w0:
            jobs_in += 1
            ops_in.append(len(j["operations"]))
            if "dueDate" in j:
                c_job.append((j["dueDate"] - a) / tw)
            blk = next(b for b in blocks if b["start"] <= a < b["end"]) if blocks else None
            ref = blk["op_mean"] if blk else sc["_meta"]["op_mean_seconds"]
            for op in j["operations"]:
                ops_total += 1
                long_ops += min(op["duration"]) > 1.5 * ref
    cap = k * STEP
    q = np.zeros((len(TYPES), n + 1))
    for t in range(n):
        q[:, t + 1] = np.maximum(0.0, q[:, t] + work[:, t] - cap)
    agv = np.array([agv_at(g) for g in grid[:-1]])
    tcap = agv * p["agv_max_utilization"] * STEP / p["agv_seconds_per_move"]
    tq = np.zeros(n + 1)
    for t in range(n):
        tq[t + 1] = max(0.0, tq[t] + moves[t] - tcap[t])
    iw0, iw1 = int(round(w0 / STEP)), n
    win = slice(iw0, iw1)
    load = work[:, win].sum(axis=0) / (cap * len(TYPES))                       # floor load per 300 s step
    tload = moves[win] / tcap[win]
    per_slot = int(SLOT / STEP)
    m = (len(load) // per_slot) * per_slot
    slot_load = load[:m].reshape(-1, per_slot).mean(axis=1)
    slot_tload = tload[:m].reshape(-1, per_slot).mean(axis=1)
    type_load = work[:, win].sum(axis=1) / (cap * (iw1 - iw0))
    f.update({
        "jobs": jobs_in,
        "util_real": float(load.mean()),
        "util_real_last3": float(slot_load[-8:].mean()),
        "util_real_first3": float(slot_load[:8].mean()),
        "slot_load_max": float(slot_load.max()),
        "slot_load_cv": float(slot_load.std() / slot_load.mean()),
        "slots_over_1": int((slot_load > 1.0).sum()),
        "type_load_max": float(type_load.max()),
        "type_imbalance": float(type_load.max() / type_load.mean()),
        "backlog_start_h": float(q[:, iw0].sum() / (k * len(TYPES) * 3600)),       # machine-hours queued at w0 (fluid)
        "backlog_mean_h": float(q[:, iw0:iw1 + 1].sum(axis=0).mean() / (k * len(TYPES) * 3600)),
        "backlog_max_h": float(q[:, iw0:iw1 + 1].sum(axis=0).max() / (k * len(TYPES) * 3600)),
        "backlog_end_h": float(q[:, iw1].sum() / (k * len(TYPES) * 3600)),
        "type_backlog_max_h": float((q[:, iw0:iw1 + 1] / (k * 3600)).max()),
        "transport_load": float(tload.mean()),
        "transport_load_max_slot": float(slot_tload.max()),
        "transport_backlog_mean": float(tq[iw0:iw1 + 1].mean()),
        "transport_backlog_max": float(tq[iw0:iw1 + 1].max()),
        "ops_per_job": float(np.mean(ops_in)),
        "long_op_share": long_ops / max(1, ops_total),
        "c_job_mean": float(np.mean(c_job)),
        "c_job_p10": float(np.percentile(c_job, 10)),
    })
    timeline = {"slot_load": slot_load, "slot_tload": slot_tload,
                "regime": [next((b["load_profile"] for b in blocks if b["start"] <= w0 + (s + 0.5) * SLOT < b["end"]), "normal")
                           for s in range(int(WINDOW / SLOT))]}
    return f, timeline


SCEN_GROUPS = {
    "regime": ["normal_share", "surge_share", "short_share", "normal_first3", "normal_mid3", "normal_last3",
               "surge_first3", "surge_mid3", "surge_last3", "short_first3", "short_mid3", "short_last3",
               "regime_changes", "pre_short", "warmup_h"],
    "load": ["util_nominal", "util_nominal_max", "util_real", "util_real_first3", "util_real_last3", "slot_load_max",
             "slot_load_cv", "slots_over_1", "jobs", "type_load_max", "type_imbalance", "seg_balanced", "seg_skewed",
             "seg_bimodal", "seg_lull", "seg_burst"],
    "queues": ["backlog_start_h", "backlog_mean_h", "backlog_max_h", "backlog_end_h", "type_backlog_max_h",
               "transport_load", "transport_load_max_slot", "transport_backlog_mean", "transport_backlog_max",
               "transport_capped"],
    "jobs/due": ["c_mean", "c_min", "c_job_mean", "c_job_p10", "op_mean_block", "op_mean_spread", "ops_per_job",
                 "long_op_share"],
}
SCEN = [c for g in SCEN_GROUPS.values() for c in g]


# ------------------------------------------------------------------------------------------------------------- data
def tard_table(path):
    e = pd.read_csv(path / "episodes.csv")
    return e.pivot_table(index="seed", columns="policy", values="window_tardiness") / 1000


def build(path, seeds, ck_seeds, tasks=False, actions=None):
    t = tard_table(path)
    rows, tl = [], {}
    for s in seeds:
        sc = json.loads((path / "scenarios" / f"s{s}.json").read_text())
        f, tl[s] = scenario_features(sc)
        mt = t.loc[s, "MDD-TECT"]
        r = {"seed": s, **f, "mddtect": mt}
        for k in ck_seeds:
            r[f"gap_s{k}"] = 100 * (t.loc[s, CK.format(k)] - mt) / mt
        r["gap"] = np.mean([r[f"gap_s{k}"] for k in LABEL_SEEDS])
        if "ATC-ECT" in t:
            r["atc_vs_mddtect"] = 100 * (t.loc[s, "ATC-ECT"] - mt) / mt
        if tasks:
            tk = json.loads((TASKS / f"B2_s{s}.json").read_text())
            fx = {kk: v["tard"] for kk, v in tk["fixed"].items()}
            r["atc_vs_mddtect"] = 100 * (fx["ATC-ECT"] - mt) / mt
            r["mddect_vs_mddtect"] = 100 * (fx["MDD-ECT"] - mt) / mt
            r["mddtect_vs_own_best"] = 100 * (mt - min(fx.values())) / min(fx.values())
            r["oracle_gain"] = 100 * (mt - tk["oracle"]) / mt
            r["own_best"] = min(fx, key=fx.get)
            r["fixed_spread"] = float(np.std(list(fx.values())) / np.mean(list(fx.values())))
            tl[s]["oracle"] = tk["schedule"]
        rows.append(r)
    d = pd.DataFrame(rows).set_index("seed")
    if actions is not None:
        a = pd.read_csv(actions / "decisions.csv")
        a = a[a.policy.str.startswith("ckpt")].sort_values(["policy", "seed", "step"])
        a["ck"] = a.policy.str.extract(r"_s(\d)/")[0].astype(int)
        for s in seeds:
            for k in LABEL_SEEDS:
                rr = a[(a.seed == s) & (a.ck == k)].rule.tolist()
                tl[s][f"policy_s{k}"] = rr
        lab = a[a.ck.isin(LABEL_SEEDS)]
        g = lab.groupby("seed")
        d["pol_mddtect_share"] = g.rule.apply(lambda r: (r == "MDD-TECT").mean())
        d["pol_atc_share"] = g.rule.apply(lambda r: r.str.startswith("ATC").mean())
        d["pol_tect_share"] = g.rule.apply(lambda r: r.str.endswith("TECT").mean())
        d["pol_switches"] = lab.groupby(["seed", "ck"]).rule.apply(lambda r: (r.values[1:] != r.values[:-1]).sum()).groupby("seed").mean()
    return d, tl


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1.0)
    return out


def feature_table(d, cols, q=0.25):
    lo, hi = d.gap.quantile(q), d.gap.quantile(1 - q)
    bad, good = d[d.gap >= hi], d[d.gap <= lo]
    rows = []
    for c in cols:
        x = d[c].astype(float)
        if x.nunique() < 2:
            continue
        rho, p = spearmanr(x, d.gap)
        mw = mannwhitneyu(bad[c], good[c]).pvalue if bad[c].nunique() + good[c].nunique() > 2 else np.nan
        rows.append({"feature": c, "rho": rho, "p": p, "bad_med": bad[c].median(), "good_med": good[c].median(),
                     "all_med": x.median(), "mw_p": mw})
    t = pd.DataFrame(rows).set_index("feature")
    t["q_bh"] = bh(t.p.values)
    return t.sort_values("p")


# ----------------------------------------------------------------------------------------------------------- models
def predict(d_train, d_test, cols):
    """Can scenario features alone tell bad from not-bad? LOO AUC on the training set, and train->test AUC."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = {}
    y_tr = (d_train.gap > 0).astype(int)
    y_te = (d_test.gap > 0).astype(int)
    models = {"logreg_L2": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
              "forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=0)}
    for name, m in models.items():
        cv_tr = LeaveOneOut() if len(d_train) <= 60 else StratifiedKFold(5, shuffle=True, random_state=0)
        p_tr = cross_val_predict(m, d_train[cols], y_tr, cv=cv_tr, method="predict_proba")[:, 1]
        p_te_cv = cross_val_predict(m, d_test[cols], y_te, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                    method="predict_proba")[:, 1]
        m.fit(d_train[cols], y_tr)
        p_te = m.predict_proba(d_test[cols])[:, 1]
        out[name] = {"cv_auc_discovery": roc_auc_score(y_tr, p_tr), "transfer_auc": roc_auc_score(y_te, p_te),
                     "cv_auc_replication": roc_auc_score(y_te, p_te_cv)}
        if name == "forest":
            m.fit(pd.concat([d_train[cols], d_test[cols]]), pd.concat([y_tr, y_te]))
            out["forest_importance"] = pd.Series(m.feature_importances_, cols).sort_values(ascending=False).head(10)
    return out


# ------------------------------------------------------------------------------------------------------------ plots
REG_COLORS = {"normal": "#e4e3df", "surge": RED, "short": VIOLET}


def plot_timelines(d, tl, path, title):
    order = d.sort_values("gap", ascending=False).index.tolist()
    n = len(order)
    has_pol = "policy_s0" in tl[order[0]]
    has_or = "oracle" in tl[order[0]]
    ncol = 2 + has_pol + has_or
    fig, axes = plt.subplots(1, ncol + 1, figsize=(3.0 * ncol + 2.2, 0.17 * n + 1.6),
                             gridspec_kw={"width_ratios": [3] * ncol + [1.6]}, sharey=True)
    reg_idx = {"normal": 0, "surge": 1, "short": 2}
    R = np.array([[reg_idx[r] for r in tl[s]["regime"]] for s in order])
    axes[0].imshow(R, aspect="auto", cmap=ListedColormap([REG_COLORS[k] for k in reg_idx]), vmin=0, vmax=2,
                   interpolation="nearest")
    axes[0].set_title("regime per 900 s slot")
    L = np.array([tl[s]["slot_load"][:24] for s in order])
    im = axes[1].imshow(L, aspect="auto", cmap="Blues", vmin=0.3, vmax=1.6, interpolation="nearest")
    axes[1].set_title("offered machine load per slot")
    cb = fig.colorbar(im, ax=axes[1], fraction=0.05, pad=0.02)
    cb.outline.set_visible(False)
    rules = ["MDD-TECT", "MDD-ECT", "ATC-TECT", "ATC-ECT", "SRT-TECT", "SRT-ECT", "EDD-TECT", "EDD-ECT", "other"]
    rcol = [BLUE, "#86b6ef", ORANGE, YELLOW, AQUA, "#7fd7b5", MAGENTA, "#f3b6cc", "#c9c8c3"]
    ridx = lambda r: rules.index(r) if r in rules else len(rules) - 1
    col = 2
    if has_pol:
        P = np.array([[ridx(r) for r in (tl[s]["policy_s0"] + ["other"] * 24)[:24]] for s in order])
        axes[col].imshow(P, aspect="auto", cmap=ListedColormap(rcol), vmin=0, vmax=len(rules) - 1, interpolation="nearest")
        axes[col].set_title("policy s0: pair per slot")
        col += 1
    if has_or:
        O = np.array([[ridx(r) for r in (tl[s]["oracle"] + ["other"] * 24)[:24]] for s in order])
        axes[col].imshow(O, aspect="auto", cmap=ListedColormap(rcol), vmin=0, vmax=len(rules) - 1, interpolation="nearest")
        axes[col].set_title("oracle: pair per slot")
    g = d.loc[order, "gap"].values
    axes[-1].barh(range(n), g, color=[RED if v > 0 else BLUE for v in g], height=0.75)
    axes[-1].axvline(0, color=MUTED, lw=0.8)
    axes[-1].set_title("gap to MDD-TECT, %")
    axes[-1].set_xlabel("mean of s0, s3")
    for a in axes[:-1]:
        a.set_xticks([0, 8, 16, 23])
        a.set_xticklabels(["0 h", "2 h", "4 h", "6 h"])
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([f"s{s}" for s in order], fontsize=6 if n > 60 else 7)
    if n > 60:
        axes[0].set_yticks(range(0, n, 10))
        axes[0].set_yticklabels([f"#{i}" for i in range(0, n, 10)])
        axes[0].set_ylabel("instances, worst gap at top")
    from matplotlib.patches import Patch
    handles = [Patch(color=REG_COLORS[k], label=k + (" (2 AGVs)" if k == "short" else "")) for k in reg_idx]
    fig.legend(handles=handles, loc="lower left", ncol=3, frameon=False, fontsize=7.5, title="regime",
               title_fontsize=7.5, bbox_to_anchor=(0.01, 0.0))
    if has_pol or has_or:
        fig.legend(handles=[Patch(color=c, label=r) for r, c in zip(rules, rcol)], loc="lower right", ncol=len(rules),
                   frameon=False, fontsize=7.5, title="rule pair", title_fontsize=7.5, bbox_to_anchor=(0.99, 0.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_features(sets, feats, path, title):
    """Strip plots: each feature's distribution in bad (top quartile gap) / middle / good (bottom quartile) instances."""
    nf = len(feats)
    ncols = 4
    nrows = int(np.ceil(nf / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(0)
    groups = [("bad", RED), ("middle", MUTED), ("good", BLUE)]
    for a, c in zip(axes, feats):
        for si, (sname, d) in enumerate(sets):
            lo, hi = d.gap.quantile(0.25), d.gap.quantile(0.75)
            grp = np.where(d.gap >= hi, "bad", np.where(d.gap <= lo, "good", "middle"))
            for gi, (gname, color) in enumerate(groups):
                v = d.loc[grp == gname, c].astype(float).values
                x = gi + (si - 0.5) * 0.36 * (len(sets) > 1)
                a.scatter(x + rng.uniform(-0.1, 0.1, len(v)), v, s=9 if sname == "0-39" else 4, color=color,
                          alpha=0.85 if sname == "0-39" else 0.35, linewidths=0,
                          marker="o" if sname == "0-39" else "s")
                if len(v):
                    a.hlines(np.median(v), x - 0.15, x + 0.15, color=INK, lw=1.6)
        a.set_xticks(range(3))
        a.set_xticklabels([g for g, _ in groups])
        a.set_title(c, fontsize=9)
    for a in axes[nf:]:
        a.axis("off")
    if len(sets) > 1:
        fig.text(0.99, 0.005, "left column of each group: seeds 0-39 (circles); right: seeds 1000-1199 (squares); "
                 "black bar = median", ha="right", fontsize=7.5, color=INK2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(sets, feats, path, title):
    fig, axes = plt.subplots(1, len(feats), figsize=(3.4 * len(feats), 3.0), sharey=True)
    for a, c in zip(np.atleast_1d(axes), feats):
        for sname, d in sets:
            a.scatter(d[c], d.gap, s=12 if sname == "0-39" else 6, color=BLUE if sname == "0-39" else ORANGE,
                      alpha=0.9 if sname == "0-39" else 0.5, linewidths=0, label=f"seeds {sname}")
        a.axhline(0, color=MUTED, lw=0.8)
        a.set_xlabel(c)
    np.atleast_1d(axes)[0].set_ylabel("policy gap to MDD-TECT, %")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------- main
def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    d0, tl0 = build(MAIN, range(40), (0, 1, 2, 3, 4), tasks=True,
                    actions=ACTIONS if (ACTIONS / "DONE").exists() else None)
    d0["gap_s0to3"] = d0[[f"gap_s{k}" for k in range(4)]].mean(axis=1)
    have_ext = (EXT / "DONE").exists()
    if have_ext:
        seeds = sorted(pd.read_csv(EXT / "episodes.csv").seed.unique())
        d1, tl1 = build(EXT, seeds, LABEL_SEEDS, actions=EXT)
    print(f"label: mean gap of checkpoints {LABEL_SEEDS} to MDD-TECT (window tardiness, %); > 0 = policy worse")
    print(f"seeds 0-39: {(d0.gap > 0).sum()}/40 lost; label vs s0-s3 mean Spearman "
          f"{spearmanr(d0.gap, d0.gap_s0to3)[0]:.2f}")
    if have_ext:
        print(f"seeds {seeds[0]}-{seeds[-1]}: {(d1.gap > 0).sum()}/{len(d1)} lost; mean gap {d1.gap.mean():+.2f}%, "
              f"median {d1.gap.median():+.2f}%; MDD-TECT vs ATC-ECT median {d1.atc_vs_mddtect.median():+.2f}%")

    print("\n== scenario features, seeds 0-39 (Spearman with the gap; bad = top quartile, good = bottom quartile; "
          "q_bh = Benjamini-Hochberg over all scenario features)")
    t0 = feature_table(d0, SCEN)
    print(t0.round(3).head(20).to_string())
    if have_ext:
        print(f"\n== same features on the replication set ({len(d1)} seeds)")
        t1 = feature_table(d1, SCEN)
        j = t0[["rho", "p", "q_bh"]].join(t1[["rho", "p", "q_bh", "bad_med", "good_med"]], lsuffix="_0_39", rsuffix="_ext")
        j["replicates"] = (np.sign(j.rho_0_39) == np.sign(j.rho_ext)) & (j.q_bh_ext < 0.1)
        print(j.sort_values("p_ext").round(3).head(25).to_string())

    out_cols = ["atc_vs_mddtect", "mddect_vs_mddtect", "mddtect_vs_own_best", "oracle_gain", "fixed_spread", "mddtect"]
    pol_cols = [c for c in ("pol_mddtect_share", "pol_atc_share", "pol_tect_share", "pol_switches") if c in d0]
    print("\n== outcome-side (fixed pairs / oracle) and policy-choice features, seeds 0-39 (not generator knobs)")
    print(feature_table(d0, out_cols + pol_cols).round(3).to_string())
    print("\nbest fixed pair per instance, bad vs good quartile (seeds 0-39):")
    lo, hi = d0.gap.quantile(0.25), d0.gap.quantile(0.75)
    print(pd.DataFrame({"bad": d0[d0.gap >= hi].own_best.value_counts(), "good": d0[d0.gap <= lo].own_best.value_counts()})
          .fillna(0).astype(int).to_string())
    if have_ext:
        print("\n== outcome / policy features on the replication set")
        print(feature_table(d1, ["atc_vs_mddtect", "mddtect", "pol_mddtect_share", "pol_atc_share", "pol_tect_share",
                                 "pol_switches"]).round(3).to_string())
        # Does "how good ATC-ECT is on this instance" explain the loss, and can the scenario predict that?
        print("\n== can the scenario predict ATC-ECT vs MDD-TECT? (replication set, top Spearman)")
        dd = d1.assign(gap=d1.atc_vs_mddtect)
        print(feature_table(dd, SCEN).round(3).head(10).to_string())

        print("\n== prediction: bad (gap > 0) from scenario features only")
        res = predict(d0, d1, SCEN)
        for k, v in res.items():
            if k != "forest_importance":
                print(f"  {k:10s} " + "  ".join(f"{a} {b:.2f}" for a, b in v.items()))
        print("  forest importance (both sets):\n" + res["forest_importance"].round(3).to_string())

    # Figures.
    plot_timelines(d0, tl0, HERE / "timeline_0_39.png",
                   "Seeds 0-39, sorted by policy gap to MDD-TECT (worst at top)")
    sets = [("0-39", d0)] + ([("1000-1199", d1)] if have_ext else [])
    top = (j.sort_values("p_ext").index[:12].tolist() if have_ext else t0.index[:12].tolist())
    plot_features(sets, top, HERE / "features_bad_vs_good.png",
                  "Scenario features in bad / middle / good instances (quartiles of the policy gap)")
    plot_features([("0-39", d0)], out_cols[:4] + pol_cols, HERE / "outcome_features_0_39.png",
                  "Outcome-side and policy-choice features, seeds 0-39")
    sc_feats = ["atc_vs_mddtect"] + top[:3]
    plot_scatter(sets, sc_feats, HERE / "gap_scatter.png", "Policy gap vs the strongest features")
    if have_ext:
        plot_timelines(d1, tl1, HERE / "timeline_ext.png",
                       f"Seeds {seeds[0]}-{seeds[-1]}, sorted by policy gap to MDD-TECT (worst at top)")
        d1.to_csv(HERE / "features_ext.csv", float_format="%.4f")
    d0.drop(columns=[]).to_csv(HERE / "features_0_39.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
