"""
@file analyze.py
@brief train-due-twin-slot: learning curves of the twin-trained PPO runs (tardiness, 6 h regime-block episodes).
       Episode return (= -tardiness over the window / 1000) by fifths of training, per seed; plus timing.json.
       Held-out evaluation against the fixed pairs is a separate step (eval-due-twin).
@par Usage
@code{.sh}
.venv/bin/python results/train-due-twin/analyze.py
@endcode
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def main():
    runs = sorted(REPO.glob("results/train-due-twin-slot_s*/episodes.csv"))
    if not runs:
        print("no runs pulled yet")
        return
    for f in runs:
        d = pd.read_csv(f)
        d = d[~d["truncated"].astype(str).str.lower().isin(["nan"])] if "truncated" in d else d
        d["fifth"] = pd.qcut(d["global_step"], 5, labels=False, duplicates="drop")
        curve = d.groupby("fifth")["return"].agg(["mean", "count"])
        t = f.parent / "timing.json"
        tim = json.loads(t.read_text()) if t.exists() else {}
        print(f"== {f.parent.name}: {len(d)} episodes, last step {d['global_step'].max():,}"
              + (f", {tim['sps_loop']:.0f} steps/s, {tim['total_s'] / 3600:.1f} h" if tim else ""))
        print("   mean return by fifth of training: " + "  ".join(f"{m:+.2f} (n={int(n)})" for m, n in curve.values))


if __name__ == "__main__":
    main()
