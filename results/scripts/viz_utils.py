"""
viz_utils.py — Shared helpers for FJSP result visualization scripts.
"""

import pandas as pd
import sys

# ── Column schemas ──────────────────────────────────────────────────────────

BRANDIMARTE_COLS = {
    "timestamp", "rule", "seed", "makespan",
    "jobs", "machines", "total_ops", "decisions", "total_reward", "timescale",
}

RANDOM_GEN_COLS = BRANDIMARTE_COLS  # same schema

SENSITIVITY_COLS = {
    "timestamp", "rule", "seed", "makespan",
    "jobs", "machines", "total_ops", "agvCount",
    "decisions", "total_reward", "timescale",
}

# ── PDR display names ────────────────────────────────────────────────────────

PDR_ORDER = [
    "SPT_SMPT", "SPT_SRWT",
    "LPT_MMUR", "LPT_SMPT",
    "SRT_SRWT", "SRT_SMPT",
    "LRT_MMUR", "SDT_SRWT",
    "RANDOM",
]

PDR_LABELS = {r: r.replace("_", "\n") for r in PDR_ORDER}


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_csv(path: str, required_cols: set) -> pd.DataFrame:
    """Load a results CSV and assert required columns are present."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] {path} is missing columns: {missing}")
    if df["makespan"].isnull().any():
        sys.exit(f"[ERROR] {path} contains null makespan values.")
    if (df["makespan"] <= 0).any():
        sys.exit(f"[ERROR] {path} contains non-positive makespan values.")
    print(f"[OK] Loaded {len(df)} rows from '{path}'")
    return df


def load_brandimarte(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Brandimarte CSV and split into:
      - pdr_df  : one row per instance per deterministic PDR
      - rand_df : repeated rows for RANDOM rule

    Instances are inferred by ordering the deterministic block by (jobs, total_ops).
    """
    df = load_csv(path, BRANDIMARTE_COLS)
    rand_df = df[df["rule"].str.upper() == "RANDOM"].copy()
    pdr_df  = df[df["rule"].str.upper() != "RANDOM"].copy()

    # Assign instance label from row order within each rule group
    # (assumes rows appear in the same problem order for every PDR)
    pdr_df = pdr_df.sort_values(["rule", "timestamp"])
    instance_ids = (
        pdr_df.groupby("rule").cumcount() + 1
    )
    pdr_df["instance"] = "MK" + instance_ids.astype(str).str.zfill(2)

    # For random, assign instance by order of appearance
    rand_df = rand_df.sort_values("timestamp").reset_index(drop=True)
    n_instances = pdr_df["instance"].nunique()
    rand_df["instance"] = "MK" + (
        (rand_df.index % n_instances + 1).astype(str).str.zfill(2)
    )

    return pdr_df, rand_df


def load_random_gen(path: str) -> pd.DataFrame:
    """Load the random-generated-jobs baseline CSV."""
    df = load_csv(path, RANDOM_GEN_COLS)
    # Create a human-readable config label
    df["config"] = df["jobs"].astype(str) + "j/" + df["machines"].astype(str) + "m"
    return df


def load_sensitivity(path: str) -> pd.DataFrame:
    """Load the AGV sensitivity analysis CSV."""
    df = load_csv(path, SENSITIVITY_COLS)
    df["config"] = df["jobs"].astype(str) + "j/" + df["machines"].astype(str) + "m"
    return df
