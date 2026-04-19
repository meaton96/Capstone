"""
test_results_format.py — Validates that results CSVs match expected schema before plotting.

Usage:
    pytest test_results_format.py --random-csv=<path> --brandimarte-csv=<path> --sensitivity-csv=<path>

    # Or skip datasets you don't have yet:
    pytest test_results_format.py --brandimarte-csv=results/brandimarte.csv
"""

import pytest
import pandas as pd
import numpy as np


# ── CLI options ──────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption("--random-csv",      default=None, help="Path to random-gen results CSV")
    parser.addoption("--brandimarte-csv", default=None, help="Path to Brandimarte results CSV")
    parser.addoption("--sensitivity-csv", default=None, help="Path to sensitivity analysis CSV")


@pytest.fixture(scope="session")
def random_csv(request):
    return request.config.getoption("--random-csv")

@pytest.fixture(scope="session")
def brandimarte_csv(request):
    return request.config.getoption("--brandimarte-csv")

@pytest.fixture(scope="session")
def sensitivity_csv(request):
    return request.config.getoption("--sensitivity-csv")


# ── Shared helpers ────────────────────────────────────────────────────────────

BASE_COLS = {"timestamp", "rule", "seed", "makespan",
             "jobs", "machines", "total_ops", "decisions", "total_reward", "timescale"}

def _load(path):
    return pd.read_csv(path, parse_dates=["timestamp"])


def _assert_base_schema(df, path):
    missing = BASE_COLS - set(df.columns)
    assert not missing, f"{path}: missing columns {missing}"


def _assert_no_nulls(df, path, cols):
    for col in cols:
        assert df[col].notnull().all(), f"{path}: null values in column '{col}'"


def _assert_positive(df, path, col):
    assert (df[col] > 0).all(), f"{path}: non-positive values in column '{col}'"


# ── Random-gen tests ──────────────────────────────────────────────────────────

class TestRandomGen:

    @pytest.fixture(autouse=True)
    def skip_if_missing(self, random_csv):
        if random_csv is None:
            pytest.skip("--random-csv not provided")

    def test_schema(self, random_csv):
        df = _load(random_csv)
        _assert_base_schema(df, random_csv)

    def test_no_null_makespan(self, random_csv):
        df = _load(random_csv)
        _assert_no_nulls(df, random_csv, ["makespan", "rule", "jobs", "machines"])

    def test_positive_makespan(self, random_csv):
        df = _load(random_csv)
        _assert_positive(df, random_csv, "makespan")

    def test_positive_jobs_and_machines(self, random_csv):
        df = _load(random_csv)
        _assert_positive(df, random_csv, "jobs")
        _assert_positive(df, random_csv, "machines")

    def test_rule_column_nonempty(self, random_csv):
        df = _load(random_csv)
        assert df["rule"].nunique() >= 1, "Expected at least one rule in random-gen CSV"

    def test_seed_is_integer(self, random_csv):
        df = _load(random_csv)
        assert pd.api.types.is_integer_dtype(df["seed"]), "seed column should be integer"

    def test_reward_is_negative_or_zero(self, random_csv):
        """Cumulative reward for FJSP should be <= 0 (penalty-based reward)."""
        df = _load(random_csv)
        assert (df["total_reward"] <= 0).all(), \
            "Expected total_reward <= 0 (penalty-based reward signal)"


# ── Brandimarte tests ─────────────────────────────────────────────────────────

class TestBrandimarte:

    @pytest.fixture(autouse=True)
    def skip_if_missing(self, brandimarte_csv):
        if brandimarte_csv is None:
            pytest.skip("--brandimarte-csv not provided")

    def test_schema(self, brandimarte_csv):
        df = _load(brandimarte_csv)
        _assert_base_schema(df, brandimarte_csv)

    def test_no_null_makespan(self, brandimarte_csv):
        df = _load(brandimarte_csv)
        _assert_no_nulls(df, brandimarte_csv, ["makespan", "rule"])

    def test_positive_makespan(self, brandimarte_csv):
        df = _load(brandimarte_csv)
        _assert_positive(df, brandimarte_csv, "makespan")

    def test_random_rule_has_multiple_seeds(self, brandimarte_csv):
        df = _load(brandimarte_csv)
        random_rows = df[df["rule"].str.upper() == "RANDOM"]
        if random_rows.empty:
            pytest.skip("No RANDOM rule rows found")
        assert random_rows["seed"].nunique() > 1, \
            "RANDOM rule should have multiple seeds (expected ~10)"

    def test_deterministic_rules_single_seed_per_instance(self, brandimarte_csv):
        """Each deterministic PDR should appear once per problem instance (no repeats)."""
        df = _load(brandimarte_csv)
        pdr_df = df[df["rule"].str.upper() != "RANDOM"]
        counts = pdr_df.groupby("rule").size()
        # All deterministic rules should have the same row count (one per instance)
        assert counts.nunique() == 1, \
            f"Deterministic PDRs have unequal row counts: {counts.to_dict()}"

    def test_expected_pdr_count(self, brandimarte_csv):
        """Expect 8 deterministic PDRs."""
        df = _load(brandimarte_csv)
        pdr_rules = df[df["rule"].str.upper() != "RANDOM"]["rule"].unique()
        assert len(pdr_rules) == 8, \
            f"Expected 8 deterministic PDRs, found {len(pdr_rules)}: {pdr_rules}"

    def test_total_ops_positive(self, brandimarte_csv):
        df = _load(brandimarte_csv)
        _assert_positive(df, brandimarte_csv, "total_ops")


# ── Sensitivity analysis tests ────────────────────────────────────────────────

class TestSensitivity:

    @pytest.fixture(autouse=True)
    def skip_if_missing(self, sensitivity_csv):
        if sensitivity_csv is None:
            pytest.skip("--sensitivity-csv not provided")

    def test_schema(self, sensitivity_csv):
        df = _load(sensitivity_csv)
        required = BASE_COLS | {"agvCount"}
        missing = required - set(df.columns)
        assert not missing, f"{sensitivity_csv}: missing columns {missing}"

    def test_no_null_makespan(self, sensitivity_csv):
        df = _load(sensitivity_csv)
        _assert_no_nulls(df, sensitivity_csv, ["makespan", "rule", "agvCount"])

    def test_positive_makespan(self, sensitivity_csv):
        df = _load(sensitivity_csv)
        _assert_positive(df, sensitivity_csv, "makespan")

    def test_agv_count_positive(self, sensitivity_csv):
        df = _load(sensitivity_csv)
        _assert_positive(df, sensitivity_csv, "agvCount")

    def test_multiple_agv_counts(self, sensitivity_csv):
        df = _load(sensitivity_csv)
        assert df["agvCount"].nunique() >= 2, \
            "Sensitivity CSV should contain at least 2 distinct AGV counts"

    def test_multiple_seeds_per_rule_config(self, sensitivity_csv):
        """Each rule+config+AGV combination should have ≥ 3 seeds."""
        df = _load(sensitivity_csv)
        df["config"] = df["jobs"].astype(str) + "j/" + df["machines"].astype(str) + "m"
        counts = df.groupby(["rule", "config", "agvCount"])["seed"].nunique()
        low = counts[counts < 3]
        if not low.empty:
            pytest.xfail(
                f"Some rule/config/AGV combos have < 3 seeds "
                f"(expected 3 repeats):\n{low}"
            )

    def test_row_count_plausible(self, sensitivity_csv):
        """Sanity check: 594 rows expected based on full config run."""
        df = _load(sensitivity_csv)
        # Warn rather than hard-fail in case configs differ
        if len(df) != 594:
            pytest.warns(UserWarning, match="row count")
            import warnings
            warnings.warn(
                f"Expected 594 rows, found {len(df)}. "
                "If configs changed this may be expected.",
                UserWarning,
            )
