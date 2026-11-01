"""
@file plot.py
@brief eval-v5-slot (2026-10-22): the joint "decision interval" figure. x = decision interval (min, log scale), y = %
       window tardiness vs MDD-PTECT on B2 test seeds 0-39 (< 0 better): the greedy switching ceiling with an exact
       forecast (../rq2-replan 1a), the planner on a published schedule with ±10-min arrival errors (1b), and PPO
       (family mean with 95% CI; dots = training seeds). -> slot_sweep.png
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

HERE = Path(__file__).resolve().parent
rp = pd.read_csv(HERE.parent / "rq2-replan" / "summary.csv")
ppo = pd.read_csv(HERE / "summary.csv")

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ex = rp[(rp.level == "exact") & rp.trigger.isna()].sort_values("interval_s")
ax.plot(ex.interval_s / 60, ex["vs_mdd_ptect_pooled_%"], "o-", color="#1f77b4", label="Planner, exact forecast (greedy oracle)")
ax.fill_between(ex.interval_s / 60, ex["vs_mdd_ptect_ci_lo"], ex["vs_mdd_ptect_ci_hi"], color="#1f77b4", alpha=0.12)
jt = rp[(rp.level == "jit10") & rp.trigger.isna()].sort_values("interval_s")
ax.plot(jt.interval_s / 60, jt["vs_mdd_ptect_pooled_%"], "s--", color="#ff7f0e", label="Planner, arrivals ±10 min in the forecast")
ppo = ppo.sort_values("slot_s")
x = ppo.slot_s / 60
ax.errorbar(x, ppo["family_pooled_%"], yerr=[ppo["family_pooled_%"] - ppo["family_ci_lo"], ppo["family_ci_hi"] - ppo["family_pooled_%"]],
            fmt="D-", color="#2ca02c", capsize=3, label="PPO, mean of training seeds (95% CI)")
for c in [c for c in ppo.columns if re.fullmatch(r"seed_s\d+_%", c)]:
    ax.scatter(x, ppo[c], s=12, color="#2ca02c", alpha=0.45, zorder=3)
ax.axhline(0, color="black", lw=0.8)
ax.set_xscale("log")
ax.set_xticks([1, 2, 3, 5, 10, 15, 20, 30, 60])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Decision interval (minutes, log scale)")
ax.set_ylabel("Window tardiness vs MDD-PTECT (%)\n(< 0 is better)")
ax.set_title("Decision interval: switching ceiling, planner with forecast error, and PPO\n(B2, twin, test seeds 0-39)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "slot_sweep.png", dpi=150)
print("wrote", HERE / "slot_sweep.png")
