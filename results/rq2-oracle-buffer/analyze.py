"""
@file analyze.py
@brief rq2-oracle-buffer: switching headroom per buffer regime, with deadlocks and blocking.

@details Same headroom columns as results/rq2-oracle-screen/analyze.py (time in system = -return of flow_time;
static_% / switch_% / total_% / spread_%), plus per regime:
  - fixed_deadlocks: fixed-pair episodes that deadlocked (of 12 x seeds); oracle_deadlocks: seeds whose final
    schedule deadlocked (the oracle ranks non-deadlocked runs first, see switch_oracle.py);
  - blocked_ms / wait_js: mean output-blocked machine-seconds and job-seconds waiting on full input buffers,
    over the stage-1 (fixed-pair) runs.
A regime whose fixed pairs deadlock often is not usable for training until the grid is loosened.

@par Usage
@code{.sh}
.venv/bin/python results/rq2-oracle-buffer/analyze.py [results/rq2-oracle-buffer]
@endcode
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("screen", HERE.parent / "rq2-oracle-screen" / "analyze.py")
screen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(screen)


def buffer_rows(root: Path) -> pd.DataFrame:
    rows = []
    for result in sorted(root.glob("*/s*/result.json")):
        r = json.loads(result.read_text())
        stages = pd.read_csv(result.parent / "stages.csv")
        fixed = stages[stages["stage"] == 1]
        rows.append({
            "regime": result.parent.parent.name, "seed": r["seed"],
            "fixed_deadlocks": len(r.get("fixed_deadlocks", [])),
            "oracle_deadlock": int(bool(r.get("oracle_deadlock"))),
            "blocked_ms": pd.to_numeric(fixed.get("output_blocked_machine_seconds"), errors="coerce").mean(),
            "wait_js": pd.to_numeric(fixed.get("buffer_wait_job_seconds"), errors="coerce").mean(),
        })
    return pd.DataFrame(rows)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    df = screen.regime_rows(root)
    if df.empty:
        print("no finished seeds yet")
        return
    summary = screen.summarize(df)
    extra = buffer_rows(root).groupby("regime").agg(
        fixed_deadlocks=("fixed_deadlocks", "sum"), oracle_deadlocks=("oracle_deadlock", "sum"),
        blocked_ms=("blocked_ms", "mean"), wait_js=("wait_js", "mean")).reset_index()
    summary = summary.merge(extra, on="regime", how="left")
    summary.to_csv(root / "summary.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 240, "display.max_columns", 24):
        print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
