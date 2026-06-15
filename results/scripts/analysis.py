"""
DFJSP machine-failure validation — visualization suite.

Compares deterministic vs stochastic (low disruption) runs to diagnose why
"low" machine failures produce 2-10x makespan inflation.

Supports single-regime analysis if only --det is provided.


python analysis.py --det-results time_scale/none/results.csv --stoch-results time_scale/low/results.csv --out time_scale/figs --det time_scale/none/machine_utilization.csv --stoch time_scale/low/machine_utilization.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REGIME_COLORS = {"deterministic": "#2E8B57", "stochastic_low": "#DC143C"}


def _instance_order(col) -> list:
    """Return instances sorted by trailing number, then alphabetically."""
    vals = pd.Series(col).dropna().unique().tolist()
    def _key(x):
        m = re.search(r"(\d+)$", str(x))
        return (int(m.group(1)) if m else 0, str(x))
    return sorted(vals, key=_key)


def _strip_regime_suffix(df: pd.DataFrame) -> pd.DataFrame:
    """Strip stochastic-tag suffixes from instance names in-place (returns copy)."""
    if "regime" not in df.columns:
        return df
    det_insts = set(df.loc[df["regime"] == "deterministic", "instance"].astype(str).unique())
    sto_insts = set(df.loc[df["regime"] == "stochastic_low", "instance"].astype(str).unique())
    if not (det_insts and sto_insts and det_insts != sto_insts):
        return df
    suffix = None
    for inst in sorted(sto_insts):
        m = re.match(r"^(.+?)(_[^_0-9][^_]*)$", inst)
        if m and m.group(1) in det_insts:
            suffix = m.group(2)
            break
    if not suffix:
        return df
    df = df.copy()
    df["instance"] = df["instance"].astype(str).apply(
        lambda x: x[: -len(suffix)] if x.endswith(suffix) else x)
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(det_path: str | None, stoch_path: str | None, combined: str | None) -> pd.DataFrame:
    """Load CSVs and tag each row with `regime`. Robust to missing stochastic paths."""
    if combined:
        df = pd.read_csv(combined)
        if "regime" not in df.columns:
            raise ValueError("--combined CSV must have a `regime` column (values: deterministic, stochastic_low).")
    else:
        if not det_path and not stoch_path:
            raise ValueError("Provide either --combined, --det, or both --det and --stoch.")
        
        pieces = []
        if det_path:
            det = pd.read_csv(det_path)
            det["regime"] = "deterministic"
            pieces.append(det)
        if stoch_path:
            sto = pd.read_csv(stoch_path)
            sto["regime"] = "stochastic_low"
            pieces.append(sto)
            
        df = pd.concat(pieces, ignore_index=True)

    df = _strip_regime_suffix(df)
    df["instance"] = pd.Categorical(df["instance"], categories=_instance_order(df["instance"]), ordered=True)

    for c in ["utilization_rate", "idle_rate", "availability_rate",
              "time_processing", "time_operational", "idle_time",
              "total_repair_time", "failure_count", "ops_completed", "makespan"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def merge_results(df: pd.DataFrame, det_results: str | None, stoch_results: str | None) -> pd.DataFrame:
    """Merge episode-level results CSV(s) into the per-machine dataframe."""
    pieces = []
    if det_results:
        r = pd.read_csv(det_results); r["regime"] = "deterministic"; pieces.append(r)
    if stoch_results:
        r = pd.read_csv(stoch_results); r["regime"] = "stochastic_low"; pieces.append(r)
    if not pieces:
        return df

    results = pd.concat(pieces, ignore_index=True)
    results = _strip_regime_suffix(results)
    join_keys = ["timestamp", "instance", "rule", "seed", "regime"]
    join_keys = [k for k in join_keys if k in df.columns and k in results.columns]
    
    overlap = (set(df.columns) & set(results.columns)) - set(join_keys)
    results_clean = results.drop(columns=list(overlap))

    merged = df.merge(results_clean, on=join_keys, how="left")

    if "agvCount" in results_clean.columns and merged["agvCount"].isna().mean() > 0.5:
        fallback_keys = [k for k in join_keys if k != "timestamp"]
        results_dedup = results_clean.drop_duplicates(subset=fallback_keys, keep="first")
        merged = df.merge(results_dedup.drop(columns=["timestamp"], errors="ignore"),
                          on=fallback_keys, how="left")
        print(f"[merge_results] timestamp join failed; fell back to {fallback_keys}")

    return merged


# ---------------------------------------------------------------------------
# Figure 1 — per-machine utilization heatmap
# ---------------------------------------------------------------------------

def fig_utilization_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    g = df.groupby(["regime", "instance", "machine_id"], observed=True)["utilization_rate"].mean().reset_index()
    active_regimes = g["regime"].unique()

    fig, axes = plt.subplots(1, len(active_regimes), figsize=(10 * len(active_regimes), 7), sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, regime in zip(axes, active_regimes):
        sub = g[g["regime"] == regime]
        pivot = sub.pivot(index="machine_id", columns="instance", values="utilization_rate")
        sns.heatmap(pivot, ax=ax, cmap="viridis", vmin=0, vmax=1,
                    cbar_kws={"label": "Utilization"}, linewidths=0.2, linecolor="white")
        ax.set_title(f"Per-machine utilization — {regime}")
        ax.set_xlabel("Instance"); ax.set_ylabel("Machine ID")
        
    fig.suptitle("Where the work actually goes (averaged across PDRs and seeds)", y=1.01)
    fig.savefig(out_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — utilization distribution per instance
# ---------------------------------------------------------------------------

def fig_utilization_distribution(df: pd.DataFrame, out_path: Path) -> None:
    active_regimes = df["regime"].unique()
    should_split = len(active_regimes) > 1
    current_palette = {r: REGIME_COLORS[r] for r in active_regimes}

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.violinplot(data=df, x="instance", y="utilization_rate", hue="regime",
                   split=should_split, inner="quartile", cut=0, ax=ax,
                   palette=current_palette, density_norm="width")
    ax.axhline(0.05, color="red",   linestyle="--", alpha=0.6, label="5% starvation threshold")
    ax.axhline(0.25, color="orange", linestyle="--", alpha=0.4, label="25% threshold")
    ax.set_title("Distribution of per-machine utilization (averaged across PDRs/seeds)")
    ax.set_xlabel("Instance"); ax.set_ylabel("Utilization rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", ncol=3)
    fig.savefig(out_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — machine-time budget stacked bar
# ---------------------------------------------------------------------------

def fig_time_budget(df: pd.DataFrame, out_path: Path) -> None:
    g = (df.groupby(["regime", "instance"], observed=True)
           .agg(processing=("time_processing", "sum"),
                idle=("idle_time", "sum"),
                repair=("total_repair_time", "sum"))
           .reset_index())
    totals = g[["processing", "idle", "repair"]].sum(axis=1).replace(0, np.nan)
    for c in ["processing", "idle", "repair"]:
        g[c] = g[c] / totals

    active_regimes = g["regime"].unique()
    fig, axes = plt.subplots(1, len(active_regimes), figsize=(9 * len(active_regimes), 6), sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, regime in zip(axes, active_regimes):
        sub = g[g["regime"] == regime].set_index("instance")[["processing", "idle", "repair"]]
        sub.plot(kind="bar", stacked=True, ax=ax,
                 color=["#2E8B57", "#C0C0C0", "#DC143C"], width=0.8, edgecolor="white")
        ax.set_title(f"Machine-time breakdown — {regime}")
        ax.set_ylabel("Fraction of total machine-time")
        ax.set_xlabel("Instance"); ax.set_ylim(0, 1)
        ax.legend(title="State", loc="upper right")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Where machine-time goes: green = useful, grey = waiting for work, red = down", y=1.02)
    fig.savefig(out_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — failures and repair time vs makespan inflation
# ---------------------------------------------------------------------------

def fig_failures_vs_makespan(df: pd.DataFrame, out_path: Path) -> None:
    if "stochastic_low" not in df["regime"].values:
        print("[fig_failures_vs_makespan] Stochastic records missing; skipping relative cross-comparison scatter.")
        return

    sto = df[df["regime"] == "stochastic_low"]
    det = df[df["regime"] == "deterministic"]

    sto_agg = (sto.groupby(["instance", "rule", "seed"], observed=True)
                  .agg(failures=("failure_count", "sum"),
                       repair_time=("total_repair_time", "sum"),
                       makespan=("makespan", "first"))
                  .reset_index())
    det_agg = (det.groupby(["instance", "rule", "seed"], observed=True)
                  .agg(makespan_det=("makespan", "first"))
                  .reset_index())
                  
    if det_agg["makespan_det"].notna().sum() == 0:
        det_agg = (det.groupby(["instance", "rule"], observed=True)
                      .agg(makespan_det=("makespan", "mean")).reset_index())
        merged = sto_agg.merge(det_agg, on=["instance", "rule"], how="left")
    else:
        merged = sto_agg.merge(det_agg, on=["instance", "rule", "seed"], how="left")
        if merged["makespan_det"].isna().any():
            fb = (det.groupby(["instance", "rule"], observed=True)
                     .agg(makespan_det_fb=("makespan", "mean")).reset_index())
            merged = merged.merge(fb, on=["instance", "rule"], how="left")
            merged["makespan_det"] = merged["makespan_det"].fillna(merged["makespan_det_fb"])
            merged.drop(columns=["makespan_det_fb"], inplace=True)

    merged["makespan_ratio"] = merged["makespan"] / merged["makespan_det"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(data=merged, x="failures", y="makespan_ratio", hue="instance", style="rule", ax=axes[0], alpha=0.75, s=70)
    axes[0].axhline(1, color="black", linestyle="--", alpha=0.4)
    axes[0].set_title("Inflation vs total failures")
    axes[0].set_xlabel("Total machine failures in episode")
    axes[0].set_ylabel("Makespan ratio (stochastic / deterministic)")
    if axes[0].get_legend() is not None:
        sns.move_legend(axes[0], "upper left", bbox_to_anchor=(1.02, 1), ncol=2, fontsize=7)

    sns.scatterplot(data=merged, x="repair_time", y="makespan_ratio", hue="instance", style="rule", ax=axes[1], alpha=0.75, s=70, legend=False)
    axes[1].axhline(1, color="black", linestyle="--", alpha=0.4)
    axes[1].set_title("Inflation vs total repair time")
    axes[1].set_xlabel("Total repair time (sim-seconds, summed across machines)")
    axes[1].set_ylabel("")
    fig.suptitle("Direct downtime cost vs makespan inflation — gap = cascade", y=1.02)
    fig.savefig(out_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — worst-inflation / worst-case instance deep dive
# ---------------------------------------------------------------------------

def fig_mk13_deep_dive(df: pd.DataFrame, out_path: Path, target: str | None = None) -> None:
    if target is None:
        means = (df.groupby(["regime", "instance"], observed=True)["makespan"].mean().unstack("regime"))
        if "stochastic_low" in means.columns and "deterministic" in means.columns:
            ratio = (means["stochastic_low"] / means["deterministic"]).dropna()
            target = str(ratio.idxmax()) if not ratio.empty else str(means["stochastic_low"].dropna().idxmax())
        elif not means.empty:
            target = str(means.iloc[:, -1].dropna().idxmax())
        else:
            print("[fig_mk13_deep_dive] No usable data; skipping deep dive."); return

    sub = df[df["instance"].astype(str) == str(target)].copy()
    if sub.empty:
        print(f"[fig_mk13_deep_dive] Instance '{target}' not found; skipping.")
        return

    g = (sub.groupby(["regime", "rule", "machine_id"], observed=True)
            .agg(utilization=("utilization_rate", "mean"),
                 failures=("failure_count", "mean"),
                 repair=("total_repair_time", "mean"),
                 ops=("ops_completed", "mean"))
            .reset_index())

    has_stochastic = "stochastic_low" in g["regime"].values
    fig, axes = plt.subplots(2, 2 if has_stochastic else 1, figsize=(17 if has_stochastic else 9, 11), squeeze=False)

    # (a) Utilization heatmap
    util = g.pivot_table(index="machine_id", columns=["regime", "rule"], values="utilization")
    sns.heatmap(util, ax=axes[0, 0], cmap="viridis", vmin=0, vmax=1, cbar_kws={"label": "Utilization"})
    axes[0, 0].set_title(f"{target} — utilization layout map")
    axes[0, 0].set_xlabel(""); axes[0, 0].set_ylabel("Machine ID")

    # If deterministic only, secondary panels map to alternate absolute indicators
    if has_stochastic:
        stoch_only = g[g["regime"] == "stochastic_low"]
        # (b) Failure tracking
        fail_p = stoch_only.pivot(index="machine_id", columns="rule", values="failures")
        sns.heatmap(fail_p, ax=axes[0, 1], cmap="Reds", annot=True, fmt=".1f", cbar_kws={"label": "Mean failures"})
        axes[0, 1].set_title(f"{target} — failures per machine (stochastic)")
        
        # (c) Repair duration layout
        rep_p = stoch_only.pivot(index="machine_id", columns="rule", values="repair")
        sns.heatmap(rep_p, ax=axes[1, 0], cmap="Reds", annot=True, fmt=".0f", cbar_kws={"label": "Total repair time (sim-s)"})
        axes[1, 0].set_title(f"{target} — total repair time (stochastic)")
        axes[1, 0].set_xlabel("PDR"); axes[1, 0].set_ylabel("Machine ID")
        
        # (d) Throughput comparison
        ops = g.pivot_table(index="machine_id", columns="regime", values="ops")
        ops[["deterministic", "stochastic_low"]].plot(
            kind="bar", ax=axes[1, 1], color=[REGIME_COLORS["deterministic"], REGIME_COLORS["stochastic_low"]],
            edgecolor="white", width=0.8)
        axes[1, 1].set_title(f"{target} — ops completed per machine")
        axes[1, 1].set_ylabel("Mean ops completed"); axes[1, 1].set_xlabel("Machine ID")
    else:
        # For deterministic data, use standard bar layouts on row 2
        ops = g.pivot_table(index="machine_id", columns="regime", values="ops")
        ops.plot(kind="bar", ax=axes[1, 0], color=[REGIME_COLORS["deterministic"]], edgecolor="white", width=0.8, legend=False)
        axes[1, 0].set_title(f"{target} — ops completed per machine")
        axes[1, 0].set_ylabel("Mean ops completed"); axes[1, 0].set_xlabel("Machine ID")

    fig.suptitle(f"{target} deep dive breakdown metrics", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — starvation summary
# ---------------------------------------------------------------------------

def fig_starvation(df: pd.DataFrame, out_path: Path) -> None:
    thresholds = [0.05, 0.10, 0.25]
    rows = []
    for (regime, inst), s in df.groupby(["regime", "instance"], observed=True):
        per_machine = s.groupby("machine_id")["utilization_rate"].mean()
        if per_machine.empty:
            continue
        for t in thresholds:
            rows.append({"regime": regime, "instance": inst,
                         "threshold": f"< {int(t*100)}%",
                         "fraction": float((per_machine < t).mean())})
    starv = pd.DataFrame(rows)

    active_regimes = starv["regime"].unique()
    fig, axes = plt.subplots(1, len(active_regimes), figsize=(9 * len(active_regimes), 6), sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, regime in zip(axes, active_regimes):
        sub = starv[starv["regime"] == regime]
        sns.barplot(data=sub, x="instance", y="fraction", hue="threshold", ax=ax, palette="rocket_r")
        ax.set_title(f"Fraction of machines below utilization threshold — {regime}")
        ax.set_xlabel("Instance"); ax.set_ylabel("Fraction of machines")
        ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Below threshold", loc="upper left")
    fig.savefig(out_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 — PDR sensitivity (CV of makespan across PDRs)
# ---------------------------------------------------------------------------

def fig_pdr_sensitivity(df: pd.DataFrame, out_path: Path) -> None:
    g = df.groupby(["regime", "instance", "rule"], observed=True)["makespan"].mean().reset_index()
    cv = (g.groupby(["regime", "instance"], observed=True)["makespan"]
            .agg(lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan)
            .reset_index(name="cv"))

    active_regimes = cv["regime"].unique()
    current_palette = {r: REGIME_COLORS[r] for r in active_regimes}

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=cv, x="instance", y="cv", hue="regime", ax=ax, palette=current_palette)
    ax.set_title("PDR sensitivity — coefficient of variation of makespan across PDRs")
    ax.set_xlabel("Instance"); ax.set_ylabel("CV(makespan) across PDRs")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(out_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8 — lambda calibration check
# ---------------------------------------------------------------------------

def fig_lambda_validation(df: pd.DataFrame, out_path: Path) -> None:
    needed = {"mean_ttf_theoretical", "episode_failures", "machines"}
    if not needed.issubset(df.columns) or "stochastic_low" not in df["regime"].values:
        print("[fig_lambda_validation] Theoretical context or stochastic records missing; skipping lambda analysis.")
        return

    sto = df[df["regime"] == "stochastic_low"]
    ep = (sto.groupby(["instance", "rule", "seed"], observed=True)
             .agg(makespan=("makespan", "first"),
                  episode_failures=("episode_failures", "first"),
                  mean_ttf=("mean_ttf_theoretical", "first"),
                  n_machines=("machines", "first"))
             .reset_index())
    ep = ep.dropna(subset=["mean_ttf", "episode_failures", "n_machines"])
    ep = ep[ep["mean_ttf"] > 0]
    if ep.empty: return

    ep["expected_failures"] = ep["n_machines"] * ep["makespan"] / ep["mean_ttf"]
    ep["failure_ratio"] = ep["episode_failures"] / ep["expected_failures"].replace(0, np.nan)

    if isinstance(ep["instance"].dtype, pd.CategoricalDtype):
        ep["instance"] = ep["instance"].cat.remove_unused_categories()

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    sns.scatterplot(data=ep, x="expected_failures", y="episode_failures", hue="instance", style="rule", ax=axes[0], alpha=0.75, s=70)
    lim = float(max(ep["expected_failures"].max(), ep["episode_failures"].max())) * 1.1
    axes[0].plot([0, lim], [0, lim], "k--", alpha=0.5, label="1:1 Match")
    axes[0].set_xlim(0, lim); axes[0].set_ylim(0, lim)
    axes[0].set_title("Observed vs expected failures per episode")
    axes[0].set_xlabel(r"Expected: $n_{machines}\, \times\, makespan\, /\, \mathrm{mean\_TTF}$")
    axes[0].set_ylabel("Observed episode_failures")

    order = _instance_order(ep["instance"])
    sns.boxplot(data=ep, x="instance", y="failure_ratio", ax=axes[1], order=order, color="#DC143C", boxprops={"alpha": 0.6})
    axes[1].axhline(1, color="black", linestyle="--", alpha=0.5)
    axes[1].set_yscale("log")
    axes[1].set_title("Failure-rate calibration ratio (observed / expected)")
    axes[1].tick_params(axis="x", rotation=45)

    fig.savefig(out_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9 — AGV count effect
# ---------------------------------------------------------------------------

def fig_agv_count_effect(df: pd.DataFrame, out_path: Path) -> None:
    if "agvCount" not in df.columns:
        return
    if df["agvCount"].dropna().nunique() < 2:
        return

    g = (df.groupby(["regime", "instance", "rule", "seed", "agvCount"], observed=True)
            .agg(makespan=("makespan", "first"),
                 mean_util=("utilization_rate", "mean"),
                 frac_starved=("utilization_rate", lambda x: float((x < 0.05).mean())))
            .reset_index())

    active_regimes = g["regime"].unique()
    current_palette = {r: REGIME_COLORS[r] for r in active_regimes}

    fig, axes = plt.subplots(2, 2, figsize=(17, 10))

    sns.lineplot(data=g, x="agvCount", y="makespan", hue="regime", style="regime", markers=True, dashes=False, ax=axes[0, 0], palette=current_palette, errorbar=("ci", 95))
    axes[0, 0].set_title("Makespan vs AGV count")

    sns.lineplot(data=g, x="agvCount", y="mean_util", hue="regime", style="regime", markers=True, dashes=False, ax=axes[0, 1], palette=current_palette, errorbar=("ci", 95))
    axes[0, 1].set_title("Mean machine utilization vs AGV count")

    sns.lineplot(data=g, x="agvCount", y="frac_starved", hue="regime", style="regime", markers=True, dashes=False, ax=axes[1, 0], palette=current_palette, errorbar=("ci", 95))
    axes[1, 0].set_title("Starvation rate vs AGV count")

    all_inst = _instance_order(g["instance"])
    n = len(all_inst)
    indices = sorted({0, n // 4, n // 2, n - 1})
    pick = [all_inst[i] for i in indices if i < n]
    if pick:
        sub = g[g["instance"].isin(pick)]
        sns.lineplot(data=sub, x="agvCount", y="makespan", hue="instance", style="regime", markers=True, ax=axes[1, 1], errorbar=("ci", 95))
        axes[1, 1].set_title("Sampled instances sweep breakdown")
    else:
        axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10 — failures per unit sim-time (rate-normalized)
# ---------------------------------------------------------------------------

def fig_failure_rate_normalized(df: pd.DataFrame, out_path: Path) -> None:
    if not {"episode_failures", "machines"}.issubset(df.columns) or "stochastic_low" not in df["regime"].values:
        return

    sto = df[df["regime"] == "stochastic_low"]
    ep = (sto.groupby(["instance", "rule", "seed"], observed=True)
             .agg(makespan=("makespan", "first"),
                  episode_failures=("episode_failures", "first"),
                  n_machines=("machines", "first"))
             .reset_index().dropna())
    ep["rate_per_machine_per_simsec"] = ep["episode_failures"] / (ep["n_machines"] * ep["makespan"])

    if "mean_ttf_theoretical" in sto.columns:
        ep_tt = sto.groupby(["instance", "rule", "seed"], observed=True)["mean_ttf_theoretical"].first().reset_index()
        ep = ep.merge(ep_tt, on=["instance", "rule", "seed"], how="left")
        ep["theoretical_rate"] = 1.0 / ep["mean_ttf_theoretical"]

    if isinstance(ep["instance"].dtype, pd.CategoricalDtype):
        ep["instance"] = ep["instance"].cat.remove_unused_categories()

    fig, ax = plt.subplots(figsize=(13, 6))
    order = _instance_order(ep["instance"])
    sns.boxplot(data=ep, x="instance", y="rate_per_machine_per_simsec", ax=ax, order=order, color="#DC143C", boxprops={"alpha": 0.6})
    
    if "theoretical_rate" in ep.columns:
        tr = ep[ep["mean_ttf_theoretical"] > 0].groupby("instance", observed=True)["theoretical_rate"].mean()
        for i, inst in enumerate(order):
            if inst in tr.index:
                ax.hlines(tr.loc[inst], i - 0.4, i + 0.4, colors="black", linestyles="--", linewidth=1.5)

    ax.set_title("Empirical failure rate per machine per sim-second")
    ax.set_ylabel("Failures / (n_machines × makespan)")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(out_path); plt.close(fig)


# ---------------------------------------------------------------------------
# Numerical summary table
# ---------------------------------------------------------------------------

def summary_table(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    for (regime, inst), s in df.groupby(["regime", "instance"], observed=True):
        per_m = s.groupby("machine_id").agg(util=("utilization_rate", "mean"),
                                            failures=("failure_count", "sum"),
                                            repair=("total_repair_time", "sum"))
        rows.append({
            "regime": regime,
            "instance": str(inst),
            "mean_util":           round(per_m["util"].mean(), 3),
            "median_util":         round(per_m["util"].median(), 3),
            "frac_below_5pct":     round((per_m["util"] < 0.05).mean(), 3),
            "frac_below_25pct":    round((per_m["util"] < 0.25).mean(), 3),
            "mean_makespan":       round(s["makespan"].mean(), 1),
            "total_failures":      int(per_m["failures"].sum()),
            "total_repair_time":   round(per_m["repair"].sum(), 1),
            "repair_frac_of_make": round(per_m["repair"].sum() / max(s["makespan"].sum(), 1), 4),
        })
    summary = pd.DataFrame(rows).sort_values(["instance", "regime"])
    summary.to_csv(out_path, index=False)
    return summary


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="DFJSP machine-failure visualization suite")
    p.add_argument("--det",   help="Deterministic machine_utilization CSV path")
    p.add_argument("--stoch", help="Stochastic (low) machine_utilization CSV path")
    p.add_argument("--combined", help="Single machine_utilization CSV with `regime` column")
    p.add_argument("--det-results",   dest="det_results", help="Deterministic results.csv")
    p.add_argument("--stoch-results", dest="stoch_results", help="Stochastic results.csv")
    p.add_argument("--out", default="./figs", help="Output directory")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.det, args.stoch, args.combined)
    df = merge_results(df, args.det_results, args.stoch_results)

    fig_utilization_heatmap(df,      out / "01_utilization_heatmap.png")
    fig_utilization_distribution(df, out / "02_utilization_distribution.png")
    fig_time_budget(df,              out / "03_time_budget.png")
    fig_failures_vs_makespan(df,     out / "04_failures_vs_makespan.png")
    fig_mk13_deep_dive(df,           out / "05_mk13_deep_dive.png")
    fig_starvation(df,               out / "06_starvation.png")
    fig_pdr_sensitivity(df,          out / "07_pdr_sensitivity.png")
    fig_lambda_validation(df,        out / "08_lambda_validation.png")
    fig_agv_count_effect(df,         out / "09_agv_count_effect.png")
    fig_failure_rate_normalized(df,  out / "10_failure_rate_normalized.png")

    summary = summary_table(df, out / "summary_table.csv")
    print("\n=== Data Execution Summary ===")
    with pd.option_context("display.max_rows", None, "display.width", 180):
        print(summary.to_string(index=False))
    print(f"\nWrote figures + summary_table.csv to {out}/")


if __name__ == "__main__":
    main()