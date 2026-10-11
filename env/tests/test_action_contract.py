"""
@file test_action_contract.py
@brief The action heads and rule catalog agree across C# (DispatchingEngine.cs, DispatchingRule.cs), env/config.py,
       the scenario schema and the DES twin.
"""
import re
from pathlib import Path

from channels.config_schema import _JOB_RULES, _MACHINE_RULES
from config import (ACTION_LAYOUT, CATALOG_JOB_RULES, CATALOG_MACHINE_RULES, JOB_HEAD_RULES, MACHINE_HEAD_RULES,
                    PTIME_RULE_PARAMS)
from des_twin.engine import TwinConfig
from des_twin.rules import JOB_RULES as TWIN_JOB_RULES, MACHINE_RULES as TWIN_MACHINE_RULES, PTIME_RULES

SIM = Path(__file__).resolve().parents[2] / "Capstone" / "Assets" / "Scripts" / "Simulation"
ENGINE = (SIM / "DispatchingEngine.cs").read_text()
RULE_ENUM = (SIM / "Types" / "DispatchingRule.cs").read_text()


def _head(name, kind):
    block = re.search(rf"{kind}\[\] {name}\s*=\s*\{{([^}}]*)\}}", ENGINE).group(1)
    return re.findall(rf"{kind}\.(\w+)", block)


def test_heads_match_csharp_in_order():
    assert _head("JobHead", "JobRule") == JOB_HEAD_RULES
    assert _head("MachineHead", "MachineRule") == MACHINE_HEAD_RULES


def test_every_rule_pair_exists_in_the_csharp_catalog():
    """@brief config_schema accepts every JOB_MACHINE pair of its halves, so the C# enum must define each."""
    enum = set(re.findall(r"\b([A-Z]+_[A-Z]+)\b", RULE_ENUM.split("enum DispatchingRule", 1)[1]))
    missing = [f"{j}_{m}" for j in _JOB_RULES for m in _MACHINE_RULES if f"{j}_{m}" not in enum]
    assert not missing
    job_enum = re.search(r"enum JobRule \{([^}]*)\}", ENGINE).group(1)
    assert set(_JOB_RULES) <= {w.strip() for w in job_enum.split(",")}


def test_twin_runs_every_head_rule():
    assert set(JOB_HEAD_RULES) <= set(TWIN_JOB_RULES)
    assert set(MACHINE_HEAD_RULES) <= set(TWIN_MACHINE_RULES) | set(PTIME_RULES)


def test_catalog_matches_csharp_and_schema():
    """@brief config.CATALOG_* (what evaluate.py --pdr accepts) = the C# job / machine halves = the scenario schema's."""
    job_enum = [w.strip() for w in re.search(r"enum JobRule \{([^}]*)\}", ENGINE).group(1).split(",")]
    machine_enum = [w.strip() for w in re.search(r"enum MachineRule \{([^}]*)\}", ENGINE).group(1).split(",")]
    assert set(CATALOG_JOB_RULES) == set(job_enum) == set(_JOB_RULES)
    assert set(CATALOG_MACHINE_RULES) == set(machine_enum) == set(_MACHINE_RULES)
    assert set(CATALOG_JOB_RULES) <= set(TWIN_JOB_RULES)
    assert set(CATALOG_MACHINE_RULES) <= set(TWIN_MACHINE_RULES) | set(PTIME_RULES)


def test_ptime_defaults_agree():
    """@brief PECT / PTECT defaults: config.PTIME_RULE_PARAMS = the twin's TwinConfig = C# PtimeRuleParams."""
    cfg = TwinConfig(rule="MDD_PECT")
    assert {k: getattr(cfg, f"ptime_{k}") for k in PTIME_RULE_PARAMS} == PTIME_RULE_PARAMS
    assert ACTION_LAYOUT["machine_head_params"] == PTIME_RULE_PARAMS
    src = (SIM / "Types" / "FJSSPConfig.cs").read_text().split("class PtimeRuleParams", 1)[1]
    cs = dict(re.findall(r"public (?:double|string) (\w+) = ([^;]+);", src))
    assert float(cs["Alpha"]) == PTIME_RULE_PARAMS["alpha"]
    assert cs["Shape"].strip('"') == PTIME_RULE_PARAMS["shape"]
    assert float(cs["Tau"]) == PTIME_RULE_PARAMS["tau"]
    assert float(cs["Reroute"]) == PTIME_RULE_PARAMS["reroute"]
    assert float(cs["Avail"]) == PTIME_RULE_PARAMS["avail"]


def test_v4_checkpoint_is_refused_with_a_clear_message():
    """@brief An H15 (action schema v4) checkpoint must not load into the v5 heads (its indices mean other rules)."""
    import pytest
    from train import check_action_layout
    v4 = {"action_layout": {"schema_version": 4, "job_head": ["SRT", "SPT", "MDD", "EDD", "ATC"],
                            "machine_head": ["ECT", "TECT", "SRWT"]}}
    with pytest.raises(ValueError, match="action schema v4.*v5"):
        check_action_layout(v4, "old.pt")
    check_action_layout({"action_layout": dict(ACTION_LAYOUT)}, "new.pt")
    other = dict(ACTION_LAYOUT, machine_head_params=dict(PTIME_RULE_PARAMS, alpha=4.0))
    with pytest.raises(ValueError, match="action layout differs"):
        check_action_layout({"action_layout": other}, "alpha4.pt")
