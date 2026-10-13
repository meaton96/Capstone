"""
@file dev_b2_slot_analyze.py
@brief Analysis for the 2026-10-07 learning-stack dev runs (results/scripts/dev_b2_slot_train_eval.sh): learning curves,
       training diagnostics, the held-out comparison table and whether the policy leaves the MDD-TECT prior where it
       should, each next to the reference prior run (train-due-twin-slot-prior-kl s0-1, scale init, 10,800 s horizon).
       10-11: works for both action layouts (H15 heads v4 and the v5 heads {SRT, MDD, ATC} x {PECT, PTECT}): the head
       rule lists come from the decision log / checkpoint width, the reference pair is MDD-PECT when the evaluation
       has it (else MDD-TECT), and "leaving the prior" compares against the best ATC pair present.

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


# Job-head rule order per action layout, keyed by the actor's output width (job + machine logits).
JOB_HEADS = {8: ("SRT", "SPT", "MDD", "EDD", "ATC"), 5: ("SRT", "MDD", "ATC")}   # v4 (H15) / v5


def actor_out(run: Path) -> str:
    sd = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)["model_state_dict"]
    w = sd["actor_critic.actor.net.3.weight"]
    off = sd.get("actor_critic.actor.prior_offset")
    b = sd["actor_critic.actor.net.3.bias"]
    lg = (b + off) if off is not None else b
    jobs = JOB_HEADS.get(len(b))
    if jobs is None:
        return f"output weight norm {w.norm():.2f} (unknown head layout, {len(b)} logits)"
    return (f"output weight norm {w.norm():.2f}, logit offset+bias MDD {lg[jobs.index('MDD')]:+.2f} "
            f"ATC {lg[jobs.index('ATC')]:+.2f} "
            f"(prior offset {'on' if off is not None and off.abs().sum() > 0 else 'off'})")


def reference(pairs: pd.DataFrame) -> str:
    """The named reference pair: MDD-PECT (B2 reference since 10-10, dev-pect-rule) when evaluated, else MDD-TECT."""
    return "MDD-PECT" if "MDD-PECT" in pairs else "MDD-TECT"


def best_atc(pairs: pd.DataFrame) -> str:
    """The ATC pair with the lowest pooled tardiness among those evaluated (ATC-ECT for H15 runs)."""
    atc = [c for c in pairs.columns if c.startswith("ATC-")]
    return min(atc, key=lambda c: pairs[c].sum())


def flips(dec: pd.DataFrame, pairs: pd.DataFrame, policy: str) -> str:
    """How the policy leaves MDD: share of slots per job rule, and whether p(ATC) / ATC slots go where the best ATC pair
    beats the reference pair."""
    a = dec[dec.policy == policy]
    if a.empty:
        return "no decisions"
    share = a.rule.value_counts(normalize=True)
    jobs = [c[len("p_job_"):] for c in a.columns if c.startswith("p_job_") and a[c].notna().any()]
    mdd = jobs.index("MDD") if "MDD" in jobs else 2
    ref, atc = reference(pairs), best_atc(pairs)
    atc_gap = 100 * (pairs[atc] - pairs[ref]) / pairs[ref]     # < 0: the ATC pair wins
    per = a.groupby("seed").agg(p_atc=("p_job_ATC", "mean"), off=("job_head", lambda h: float((h != mdd).mean())))
    per = per.join(atc_gap.rename("atc_gap"), how="inner")
    rho = spearmanr(per.p_atc, per.atc_gap).statistic
    win = per[per.atc_gap < 0]
    lose = per[per.atc_gap >= 0]
    return (f"slots: " + ", ".join(f"{k} {v:.1%}" for k, v in share.head(4).items())
            + f"\n      mean p(ATC) {a.p_job_ATC.mean():.3f}; Spearman(p(ATC), {atc} gap vs {ref}) {rho:+.2f}; "
            f"non-MDD slots where {atc} wins ({len(win)} inst) {win.off.mean():.1%} vs elsewhere ({len(lose)}) "
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
    show = summary[(summary.kind == "checkpoint") | summary.policy.isin(
        ["MDD-PECT", "MDD-PTECT", "ATC-PTECT", "SRT-PECT", "MDD-TECT", "ATC-ECT", "SRT-ECT"])]
    pd.set_option("display.width", 220)
    print("\n== held-out B2 seeds 0-39 (cost = window tardiness / 1000; % change in mean cost, < 0 better)")
    print(show[[c for c in cols if c in show]].round(3).to_string(index=False))
    print("   vs best = the named best fixed pair (MDD-PECT since 10-11, MDD-TECT before); vs hindsight = each instance's "
          "best of the evaluated pairs; oracle = rq2-twin-fleet B2 (H15)")

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
