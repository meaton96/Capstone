"""
@file dev_b2_slot_analyze.py
@brief Analysis for the 2026-10-07 learning-stack dev runs (results/scripts/dev_b2_slot_train_eval.sh): learning curves,
       training diagnostics, the held-out comparison table and whether the policy leaves the MDD-TECT prior where it
       should, each next to the reference prior run (train-due-twin-slot-prior-kl s0-1, scale init, 10,800 s horizon).

@par Usage
@code{.sh}
.venv/bin/python results/scripts/dev_b2_slot_analyze.py dev-prior-offset
@endcode
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

RES = Path(__file__).resolve().parents[1]
REF = "train-due-twin-slot-prior-kl"
REF_EVAL = RES / "eval-due-twin-slot-prior"


def curve(run: Path) -> str:
    d = pd.read_csv(run / "episodes.csv")
    d["fifth"] = pd.qcut(d["global_step"], 5, labels=False, duplicates="drop")
    c = d.groupby("fifth")["return"].mean()
    return "  ".join(f"{m:+.1f}" for m in c.values) + f"   ({len(d)} episodes)"


def scalars(run: Path, tags) -> dict:
    """Mean of each TensorBoard scalar over the last quarter of training and over the first update."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in tags:
        if tag not in acc.Tags()["scalars"]:
            continue
        v = np.array([e.value for e in acc.Scalars(tag)])
        out[tag] = (v[0], v[-max(1, len(v) // 4):].mean())
    return out


def actor_out(run: Path) -> str:
    sd = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)["model_state_dict"]
    w = sd["actor_critic.actor.net.3.weight"]
    off = sd.get("actor_critic.actor.prior_offset")
    b = sd["actor_critic.actor.net.3.bias"]
    lg = (b + off) if off is not None else b
    return (f"output weight norm {w.norm():.2f}, logit offset+bias MDD {lg[2]:+.2f} ATC {lg[4]:+.2f} "
            f"(prior offset {'on' if off is not None and off.abs().sum() > 0 else 'off'})")


def flips(dec: pd.DataFrame, pairs: pd.DataFrame, policy: str) -> str:
    """How the policy leaves MDD: share of slots per job rule, and whether p(ATC) / ATC slots go where ATC-ECT wins."""
    a = dec[dec.policy == policy]
    if a.empty:
        return "no decisions"
    share = a.rule.value_counts(normalize=True)
    atc_gap = 100 * (pairs["ATC-ECT"] - pairs["MDD-TECT"]) / pairs["MDD-TECT"]     # < 0: ATC-ECT wins
    per = a.groupby("seed").agg(p_atc=("p_job_ATC", "mean"), off=("job_head", lambda h: float((h != 2).mean())))
    per = per.join(atc_gap.rename("atc_gap"), how="inner")
    rho = spearmanr(per.p_atc, per.atc_gap).statistic
    win = per[per.atc_gap < 0]
    lose = per[per.atc_gap >= 0]
    return (f"slots: " + ", ".join(f"{k} {v:.1%}" for k, v in share.head(4).items())
            + f"\n      mean p(ATC) {a.p_job_ATC.mean():.3f}; Spearman(p(ATC), ATC-ECT gap) {rho:+.2f}; "
            f"non-MDD slots where ATC-ECT wins ({len(win)} inst) {win.off.mean():.1%} vs elsewhere ({len(lose)}) "
            f"{lose.off.mean():.1%}")


def main():
    name = sys.argv[1]
    seeds = sorted(RES.glob(f"{name}_s*"))
    print(f"=== {name}: {len(seeds)} training seeds\n")
    print("== learning curve: mean episode return (= -window tardiness / 1000) by fifth of training")
    for run in seeds:
        ref = RES / f"{REF}_{run.name.rsplit('_', 1)[1]}"
        print(f"  {run.name:34s} {curve(run)}")
        if ref.exists():
            print(f"  {'(ref) ' + ref.name:34s} {curve(ref)}")
    tags = ["charts/explained_variance", "losses/prior_kl", "losses/entropy", "charts/adv_var_before",
            "charts/adv_var_after", "charts/shared_baseline_share", "charts/paired_share", "charts/paired_nonzero_share",
            "charts/adv_var_gae", "charts/adv_var_paired"]
    print("\n== training diagnostics (first update -> mean of the last quarter)")
    for run in seeds:
        ref = RES / f"{REF}_{run.name.rsplit('_', 1)[1]}"
        for r in [run] + ([ref] if ref.exists() else []):
            s = scalars(r, tags)
            print(f"  {r.name:34s} " + "  ".join(f"{k.split('/')[1]} {a:.3f}->{b:.3f}" for k, (a, b) in s.items()))
            print(f"  {'':34s} {actor_out(r)}")

    ev = RES / name / "eval"
    summary = pd.read_csv(ev / "summary.csv")
    cols = ["policy", "return_mean", "vs_best_fixed_pct", "wins_vs_best_fixed", "losses_vs_best_fixed",
            "vs_hindsight_fixed_pct", "vs_oracle_pct", "oracle_gain_captured_pct", "discounted_return_mean"]
    show = summary[(summary.kind == "checkpoint") | summary.policy.isin(["MDD-TECT", "ATC-ECT", "SRT-ECT"])]
    pd.set_option("display.width", 220)
    print("\n== held-out B2 seeds 0-39 (cost = window tardiness / 1000; % change in mean cost, < 0 better)")
    print(show[[c for c in cols if c in show]].round(3).to_string(index=False))
    print("   vs best = MDD-TECT (named); vs hindsight = each instance's best of the 15 pairs; oracle = rq2-twin-fleet B2")

    e = pd.read_csv(ev / "episodes.csv")
    pairs = e[e.kind == "pdr"].assign(tard=lambda d: -d["return"]).pivot(index="seed", columns="policy", values="tard")
    dec = pd.read_csv(ev / "decisions.csv")
    print("\n== leaving the prior (deterministic slot choices, job head; MDD = 2)")
    for pol in summary[summary.kind == "checkpoint"].policy:
        print(f"  {pol}\n      {flips(dec, pairs, pol)}")
    if (REF_EVAL / "decisions.csv").exists():
        rd = pd.read_csv(REF_EVAL / "decisions.csv")
        for s in (0, 1):
            pol = f"ckpt:{REF}_s{s}/checkpoint"
            print(f"  (ref) {pol}\n      {flips(rd, pairs, pol)}")


if __name__ == "__main__":
    main()
