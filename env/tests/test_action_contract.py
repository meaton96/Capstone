"""
@file test_action_contract.py
@brief The action heads and rule catalog agree across C# (DispatchingEngine.cs, DispatchingRule.cs), env/config.py,
       the scenario schema and the DES twin.
"""
import re
from pathlib import Path

from channels.config_schema import _JOB_RULES, _MACHINE_RULES
from config import JOB_HEAD_RULES, MACHINE_HEAD_RULES
from des_twin.rules import JOB_RULES as TWIN_JOB_RULES, MACHINE_RULES as TWIN_MACHINE_RULES

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
    assert set(MACHINE_HEAD_RULES) <= set(TWIN_MACHINE_RULES)
