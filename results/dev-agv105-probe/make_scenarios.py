"""
@file make_scenarios.py
@brief dev-agv105-probe (2026-10-15): the first hour of arrivals of two 105-machine agv-study-rerun scenarios, as
       short Unity batch scenarios that run to empty. They measure how fast the player simulates a 105-machine floor
       before more 30 h cells are spent. Everything but the job list is the full scenario's (tiling, layout, rules
       come from the run's flags). Writes linux_server/BatchConfigs/Scenarios/agv5_*_arr1h_s0.json.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SC = REPO / "linux_server" / "BatchConfigs" / "Scenarios"
CUT = 3600.0

for src in ("agv5_b2t_t7open_s0", "agv5_b2m_t7iso_s0"):
    s = json.loads((SC / f"{src}.json").read_text())
    n_all = len(s["jobs"])
    s["jobs"] = [j for j in s["jobs"] if j["arrivalTime"] < CUT]
    s["name"] = src.replace("_s0", "_arr1h_s0")
    s["_comment"] = (f"dev-agv105-probe: arrivals < {CUT:.0f} s of {src} ({len(s['jobs'])} of {n_all} jobs). "
                     + s.get("_comment", ""))
    out = SC / f"{s['name']}.json"
    out.write_text(json.dumps(s))
    print(f"{out.relative_to(REPO)}: {len(s['jobs'])} of {n_all} jobs, last arrival "
          f"{max(j['arrivalTime'] for j in s['jobs']):.0f} s")
